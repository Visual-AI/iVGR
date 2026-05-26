#!/usr/bin/env python3
"""
Script for generating SFT datasets (using offline-generated CoT samples).

Workflow:
1. Read all rollouts and their rewards saved during one epoch.
2. Filter for questions where group B or single-group accuracy <= 0.4.
3. For group B questions, look up the matching group A rollouts by original_idx.
4. Keep rollouts that pass: format reward=1, acc reward=1, box valid=1, box reward >= 0.3 (max).
5. Use the judge model (qwen-72B) to rewrite that rollout 5 times (drop boxes, keep image descriptions and logic).
6. Pull additional samples from the offline-generated SFT folder (instead of calling 235b online).
7. For single samples, source samples directly from the offline SFT folder.
8. Save as a llama-factory-formatted SFT JSON dataset.
"""

import json
import os
import re
import argparse
import base64
import threading
import random
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from jinja2 import Template
from PIL import Image

# Import math_verify if available
try:
    import math_verify
    parse = lambda x: math_verify.parse(x, parsing_timeout=None)
    verify = lambda x, y: math_verify.verify(x, y)
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("[Warning] math_verify not available, will use LLM for math verification")

# ============================================================================
# Judge model configuration (qwen-72B)
# ============================================================================
JUDGE_CLIENT = OpenAI(
    base_url="http://10.20.33.239:10931/v1",
    api_key="EMPTY"
)
JUDGE_MODEL_NAME = "judge"


# ============================================================================
# Rewrite prompt (revised version with diversity instructions)
# ============================================================================
REWRITE_PROMPT = """You are an expert at rewriting reasoning processes while maintaining factual accuracy.

## Task
Rewrite the following reasoning process by:
1. **Removing all bounding boxes**: Delete all <box>[x1,y1,x2,y2]</box> tags completely
2. **Preserving ALL visual descriptions**: Keep every single detail about what is seen in the image - do NOT add, remove, or change any visual observations
3. **Maintaining logical flow**: The reasoning steps should follow the same logic
4. **Varying expression style**: Use different sentence structures, word choices, and phrasing while keeping the same meaning

## Important Rules
- Do NOT hallucinate or invent any new visual details
- Do NOT omit any visual observations from the original
- Every factual claim about the image must be preserved exactly
- Only the way of expressing these facts can vary

## Original Reasoning (with boxes to remove):
{original_reasoning}

## Rewritten Reasoning (without boxes, same facts, different expression):
"""


# ============================================================================
# Format-checking functions
# ============================================================================
def check_format(response: str) -> bool:
    """Check whether the response format is correct."""
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, response.strip())
    return format_match is not None


def extract_think_content(response: str) -> Optional[str]:
    """Extract the content inside <think>...</think>."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer_content(response: str) -> Optional[str]:
    """Extract the content inside <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else None


# ============================================================================
# Data loading helpers
# ============================================================================
def load_json_logs(json_log_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON log files."""
    all_samples = []
    json_files = sorted([f for f in os.listdir(json_log_dir) if f.startswith("step_") and f.endswith(".json")])
    
    print(f"Found {len(json_files)} JSON log files")
    
    for json_file in tqdm(json_files, desc="Loading JSON logs"):
        json_path = os.path.join(json_log_dir, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
                all_samples.extend(log_data.get("samples", []))
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    print(f"Loaded {len(all_samples)} total samples")
    return all_samples


def filter_low_accuracy_samples(samples: List[Dict[str, Any]], threshold: float = 0.4, max_samples: int = 200) -> List[Dict[str, Any]]:
    """Filter questions where group B accuracy <= threshold."""
    filtered = []
    for sample in samples:
        group = sample.get("group", "")
        rewards = sample.get("rewards", {})
        accuracy = rewards.get("accuracy", 0.0)
        
        if (group == "B") and accuracy <= threshold:
            filtered.append(sample)
        if max_samples > 0 and len(filtered) >= max_samples:
            break
    print(f"Filtered {len(filtered)} samples with accuracy <= {threshold} (group B)")
    return filtered


def find_best_group_a_rollout(samples: List[Dict[str, Any]], original_idx: int) -> Optional[Dict[str, Any]]:
    """Find a qualifying Group A rollout."""
    group_a_samples = [
        s for s in samples
        if s.get("group") == "A" and s.get("original_idx") == original_idx
    ]

    if not group_a_samples:
        return None

    # Filter qualifying rollouts
    qualified = []
    for sample in group_a_samples:
        rewards = sample.get("rewards", {})
        if (rewards.get("format", 0) == 1.0 and
            rewards.get("accuracy", 0) == 1.0 and
            rewards.get("box_valid", 0) == 1.0 and
            rewards.get("box_reward", 0) >= 0.3):
            qualified.append(sample)

    if not qualified:
        return None

    # Pick the one with the highest box_reward
    best = max(qualified, key=lambda x: x.get("rewards", {}).get("box_reward", 0))
    return best


def _filter_unused_data(data):
    def doc2len(doc):
        if doc["datasource"] == "treevgr" or doc["datasource"] in ["ours_arxivqa", "ours_mmk12",
    "ours_thinklite",
    "ours_tqa",
    "ours_virl",
    "ours_wemath_pro",
    "ours_wemath_std", "ours_docqa", "ours_infoqa"]:
            return True
        else:
            return False

    dataframe = data.filter(
        lambda doc: doc2len(doc),
        num_proc=16,
        desc=f"Filtering textcot task",
    )
    return dataframe


def load_original_dataset(data_path: str, image_dir: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """Load the original dataset; returns an original_idx -> data mapping."""
    dataset_dict = {}
    
    if data_path.endswith(".parquet"):
        from datasets import load_dataset
        df = load_dataset("parquet", data_files=data_path, split="train")
        df = _filter_unused_data(df)
        for idx in range(len(df)):
            row = df[idx]
            dataset_dict[idx] = {
                "question": row.get("problem", ""),
                "query": row.get("query", row.get("problem", "")),  # Some datasets use `query`
                "answer": row.get("answer", ""),
                "images": row.get("images", []),
                "datasource": row.get("datasource", ""),
            }
    else:
        # Assume a JSON file
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for idx, item in enumerate(data):
                dataset_dict[idx] = {
                    "question": item.get("problem", item.get("question", "")),
                    "query": item.get("query", item.get("problem", item.get("question", ""))),
                    "answer": item.get("answer", ""),
                    "images": item.get("images", []),
                    "datasource": item.get("datasource", ""),
                }
    
    print(f"Loaded {len(dataset_dict)} samples from original dataset")
    return dataset_dict


def load_offline_sft_samples(offline_sft_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load offline-generated SFT samples.

    Args:
        offline_sft_dir: Offline SFT sample directory; each file is named {idx}.json.

    Returns:
        Mapping from original_idx -> List[sft_sample].
    """
    offline_samples = {}
    
    if not os.path.exists(offline_sft_dir):
        print(f"Warning: Offline SFT directory not found: {offline_sft_dir}")
        return offline_samples
    
    json_files = [f for f in os.listdir(offline_sft_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} offline SFT files")
    
    for json_file in tqdm(json_files, desc="Loading offline SFT samples"):
        try:
            idx = int(json_file.replace(".json", ""))
            json_path = os.path.join(offline_sft_dir, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
                if samples:  # Only keep non-empty sample lists
                    offline_samples[idx] = samples
        except (ValueError, json.JSONDecodeError) as e:
            # Filename isn't numeric or JSON parsing failed
            continue
    
    print(f"Loaded offline SFT samples for {len(offline_samples)} indices")
    return offline_samples


def get_format_prompt(group: str, format_prompt_dir: str = "./examples/format_prompt") -> str:
    """Pick the appropriate format prompt by group."""
    if group == "B":
        # Group B uses the math prompt
        format_file = os.path.join(format_prompt_dir, "adagrounding_sft_data_examples.jinja")
    else:  # single
        format_file = os.path.join(format_prompt_dir, "adagrounding_thyme_mathdata.jinja")

    if os.path.exists(format_file):
        with open(format_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Default format
        return "{{ content | trim }}"


def remove_boxes_from_reasoning(reasoning: str) -> str:
    """Strip all box tags from the reasoning."""
    # Remove <box>...</box>
    reasoning = re.sub(r"<box>.*?</box>", "", reasoning, flags=re.DOTALL)
    # Collapse extra whitespace/newlines
    reasoning = re.sub(r"\s+", " ", reasoning)
    reasoning = reasoning.strip()
    return reasoning


# ============================================================================
# CoT rewriting (uses the judge model)
# ============================================================================
def rewrite_reasoning_with_judge(original_reasoning: str, num_rewrites: int = 5) -> List[str]:
    """
    Rewrite the reasoning with the judge model (qwen-72B).
    Uses higher temperature to increase diversity.
    """
    rewritten_list = []
    seen = set()  # Dedup set

    prompt = REWRITE_PROMPT.format(original_reasoning=original_reasoning)

    # Use varying temperatures to increase diversity
    temperatures = [0.3, 0.5, 0.7, 0.8, 0.9]

    attempts = 0
    max_attempts = num_rewrites * 3  # Try up to 3x the target count

    while len(rewritten_list) < num_rewrites and attempts < max_attempts:
        temp = temperatures[attempts % len(temperatures)]
        attempts += 1

        try:
            completion = JUDGE_CLIENT.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
            )
            rewritten = completion.choices[0].message.content.strip()

            # Dedup
            if rewritten not in seen:
                seen.add(rewritten)
                rewritten_list.append(rewritten)
                print(f"  [Rewrite] Got unique CoT #{len(rewritten_list)} (temp={temp})")
        except Exception as e:
            print(f"Warning: Failed to rewrite (attempt {attempts}): {e}")

    # If still short, fall back to simple box removal
    if len(rewritten_list) < num_rewrites:
        fallback = remove_boxes_from_reasoning(original_reasoning)
        while len(rewritten_list) < num_rewrites:
            rewritten_list.append(fallback)

    return rewritten_list[:num_rewrites]


# ============================================================================
# SFT sample creation
# ============================================================================
def create_sft_sample(
    original_data: Dict[str, Any],
    rewritten_reasoning: str,
    answer: str,
    group: str,
    format_prompt_template: str,
    image_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Create an SFT sample in llama-factory format."""
    question = original_data.get("question", "")
    images = original_data.get("images", [])

    # Resolve image paths
    image_paths = []
    if images:
        if isinstance(images[0], str):
            # Path strings
            if image_dir:
                image_paths = [os.path.join("", img) if not os.path.isabs(img) else img for img in images]
            else:
                image_paths = images
        elif isinstance(images[0], dict) and "bytes" in images[0]:
            # bytes format: would need to write the image out
            # (simplified here — assume a path already exists)
            pass

    # Build the full response
    response = f"<think>{rewritten_reasoning}</think> <answer>{answer}</answer>"

    # Build the user prompt
    if group in ["B", "single"]:
        question_with_image = f"{question}"
    else:
        question_with_image = question

    template = Template(format_prompt_template)
    user_prompt = template.render(content=question_with_image)

    # Build conversations
    conversations = [
        {
            "from": "human",
            "value": user_prompt
        },
        {
            "from": "gpt",
            "value": response
        }
    ]
    
    sample = {
        "images": image_paths if image_paths else [],
        "conversations": conversations,
        "system": "You are a helpful assistant."
    }
    
    return sample


def select_offline_samples(offline_samples: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
    """
    Select a target number of samples from the offline pool.

    Args:
        offline_samples: Offline sample list.
        num_samples: Number of samples to select.

    Returns:
        Selected samples.
        - If the offline pool has <= num_samples, return all samples.
        - If it has > num_samples, randomly select num_samples.
    """
    if len(offline_samples) <= num_samples:
        return offline_samples
    else:
        return random.sample(offline_samples, num_samples)


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate SFT JSON samples from training rollouts (using offline CoT samples)")
    parser.add_argument("--json_log_dir", type=str, required=True, help="Directory containing JSON log files from training")
    parser.add_argument("--original_data_path", type=str, required=True, help="Path to original dataset (parquet or json)")
    parser.add_argument("--offline_sft_dir", type=str, required=True, help="Directory containing offline generated SFT samples")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for SFT JSON file")
    parser.add_argument("--accuracy_threshold", type=float, default=0.4, help="Accuracy threshold for filtering")
    parser.add_argument("--max_filtered_samples", type=int, default=200, help="Maximum number of filtered samples to process (-1 for all)")
    parser.add_argument("--num_rewrites", type=int, default=5, help="Number of CoT rewrites by judge model")
    parser.add_argument("--num_offline_samples", type=int, default=3, help="Number of samples from offline generated data for group B")
    parser.add_argument("--num_single_samples", type=int, default=8, help="Number of samples for single group")
    parser.add_argument("--format_prompt_dir", type=str, default="./examples/format_prompt", help="Directory containing format prompt templates")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # 1. Load all JSON logs
    print("=" * 60)
    print("Step 1: Loading JSON logs...")
    print("=" * 60)
    all_samples = load_json_logs(args.json_log_dir)

    # 2. Filter low-accuracy samples
    print("\n" + "=" * 60)
    print("Step 2: Filtering low accuracy samples...")
    print("=" * 60)
    filtered_samples = filter_low_accuracy_samples(all_samples, args.accuracy_threshold, args.max_filtered_samples)

    # 3. Load the original dataset
    print("\n" + "=" * 60)
    print("Step 3: Loading original dataset...")
    print("=" * 60)
    original_dataset = load_original_dataset(args.original_data_path, args.image_dir)

    # 4. Load offline-generated SFT samples
    print("\n" + "=" * 60)
    print("Step 4: Loading offline SFT samples...")
    print("=" * 60)
    offline_sft_samples = load_offline_sft_samples(args.offline_sft_dir)

    # 5. Collect every original_idx that appears in the JSON logs
    print("\n" + "=" * 60)
    print("Step 5: Collecting trained sample indices...")
    print("=" * 60)
    
    trained_indices = set()
    for sample in all_samples:
        original_idx = sample.get("original_idx", -1)
        if original_idx >= 0:
            trained_indices.add(original_idx)
    print(f"Found {len(trained_indices)} unique samples that have been trained")
    
    # 6. Process each sample
    print("\n" + "=" * 60)
    print("Step 6: Processing samples...")
    print("=" * 60)

    sft_samples = []
    processed_count = 0
    skipped_count = 0

    # Group by original_idx
    idx_to_samples = defaultdict(list)
    for sample in filtered_samples:
        original_idx = sample.get("original_idx", -1)
        if original_idx >= 0:
            idx_to_samples[original_idx].append(sample)

    # Per-sample processing function
    def process_single_sample(original_idx: int, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Process all samples for a single original_idx.

        Returns:
            (generated SFT samples, processed count, skipped count).
        """
        result_samples = []
        local_processed = 0
        local_skipped = 0

        # Skip if it didn't appear in the JSON logs (i.e. wasn't trained)
        if original_idx not in trained_indices:
            print(f"[{original_idx}] Warning: was not trained, skipping")
            return result_samples, local_processed, 1

        if original_idx not in original_dataset:
            print(f"[{original_idx}] Warning: not found in original dataset")
            return result_samples, local_processed, 1

        original_data = original_dataset[original_idx]
        ground_truth = original_data.get("answer", "")
        question = original_data.get("question", "") or original_data.get("query", "")
        question = question.replace("<image>", "").strip()

        # Look up offline-generated samples for this idx
        offline_samples_for_idx = offline_sft_samples.get(original_idx, [])

        # For group B, we need to find the matching Group A rollout
        group_b_samples = [s for s in samples if s.get("group") == "B"]
        single_samples = [s for s in samples if s.get("group") == "single"]

        # ====================================================================
        # Handle Group B samples
        # ====================================================================
        if group_b_samples:
            print(f"[{original_idx}] Processing Group B...")

            # Load the format prompt
            format_prompt_template = get_format_prompt("B", args.format_prompt_dir)

            # Find the best Group A rollout
            best_group_a = find_best_group_a_rollout(all_samples, original_idx)

            if best_group_a is None:
                # No qualifying Group A rollout — fall back to offline samples only
                print(f"[{original_idx}] No qualified Group A rollout, using offline samples...")
                # total_samples_needed = args.num_rewrites + args.num_offline_samples
                total_samples_needed = args.num_offline_samples

                if len(offline_samples_for_idx) == 0:
                    print(f"[{original_idx}] [ABANDON] No offline samples available")
                    local_skipped += 1
                else:
                    # Use offline samples (random or all)
                    selected_offline = select_offline_samples(offline_samples_for_idx, total_samples_needed)
                    for offline_sample in selected_offline:
                        result_samples.append(offline_sample)
                        local_processed += 1

                    print(f"[{original_idx}] Group B (all offline): {len(selected_offline)} samples (available: {len(offline_samples_for_idx)})")
            else:
                # Extract Group A's think content and answer
                group_a_response = best_group_a.get("response", "")
                think_content = extract_think_content(group_a_response)
                answer_match = re.search(r"<answer>(.*?)</answer>", group_a_response, re.DOTALL)
                answer = answer_match.group(1).strip() if answer_match else ""

                if think_content is None:
                    # Failed to extract think content — fall back to offline samples
                    print(f"[{original_idx}] No think content in Group A, using offline samples...")
                    total_samples_needed = args.num_offline_samples

                    if len(offline_samples_for_idx) == 0:
                        print(f"[{original_idx}] [ABANDON] No offline samples available")
                        local_skipped += 1
                    else:
                        # Use offline samples (random or all)
                        selected_offline = select_offline_samples(offline_samples_for_idx, total_samples_needed)
                        for offline_sample in selected_offline:
                            result_samples.append(offline_sample)
                            local_processed += 1

                        print(f"[{original_idx}] Group B (all offline): {len(selected_offline)} samples (available: {len(offline_samples_for_idx)})")
                else:
                    # Part 1: rewrite CoT 5 times with the judge model
                    rewritten_list = rewrite_reasoning_with_judge(
                        think_content,
                        num_rewrites=args.num_rewrites
                    )

                    for rewritten_reasoning in rewritten_list:
                        sft_sample = create_sft_sample(
                            original_data,
                            rewritten_reasoning,
                            answer,
                            "B",
                            format_prompt_template,
                            args.image_dir
                        )
                        result_samples.append(sft_sample)
                        local_processed += 1

                    # Part 2: pull offline samples (random or all)
                    selected_offline = select_offline_samples(offline_samples_for_idx, args.num_offline_samples)
                    for offline_sample in selected_offline:
                        result_samples.append(offline_sample)
                        local_processed += 1

                    print(f"[{original_idx}] Group B: {len(rewritten_list)} rewritten + {len(selected_offline)} offline (available: {len(offline_samples_for_idx)})")

        # ====================================================================
        # Handle single samples
        # ====================================================================
        if single_samples:
            print(f"[{original_idx}] Processing Single...")

            # Use offline-generated SFT samples (random or all)
            if len(offline_samples_for_idx) == 0:
                print(f"[{original_idx}] [ABANDON] No offline samples available for single")
                local_skipped += 1
            else:
                selected_offline = select_offline_samples(offline_samples_for_idx, args.num_single_samples)
                for offline_sample in selected_offline:
                    result_samples.append(offline_sample)
                    local_processed += 1

                print(f"[{original_idx}] Single: {len(selected_offline)} offline samples (available: {len(offline_samples_for_idx)})")

        return result_samples, local_processed, local_skipped

    # Run in parallel with a thread pool
    print(f"Using {args.num_workers} threads for parallel processing...")

    # Lock to protect result aggregation
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_sample, original_idx, samples): original_idx
            for original_idx, samples in idx_to_samples.items()
        }

        # Collect results
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing samples"):
            original_idx = future_to_idx[future]
            try:
                result_samples, local_processed, local_skipped = future.result()
                with results_lock:
                    sft_samples.extend(result_samples)
                    processed_count += local_processed
                    skipped_count += local_skipped
            except Exception as e:
                print(f"[{original_idx}] Error: {e}")
                with results_lock:
                    skipped_count += 1

    # 7. Save results
    print("\n" + "=" * 60)
    print("Step 7: Saving results...")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nSummary:")
    print(f"  Total samples processed: {processed_count}")
    print(f"  Samples skipped: {skipped_count}")
    print(f"  Total SFT samples generated: {len(sft_samples)}")
    print(f"  Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()

