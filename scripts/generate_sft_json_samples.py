#!/usr/bin/env python3
"""
Script for generating SFT datasets.

Workflow:
1. Read all rollouts and their rewards saved during one epoch.
2. Filter for questions where group B or single-group accuracy <= 0.4.
3. For group B questions, look up the matching group A rollouts by original_idx.
4. Keep rollouts that pass: format reward=1, acc reward=1, box valid=1, box reward >= 0.3 (max).
5. Use the judge model (qwen-72B) to rewrite that rollout 5 times (drop boxes, keep image descriptions and logic).
6. Generate 3 more samples with the local 235b model (with format and correctness checks).
7. For single samples, use the 235b model to generate reasoning + answers until 8 are produced.
8. Save as a llama-factory-formatted SFT JSON dataset.
"""

import json
import os
import re
import argparse
import base64
import threading
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
# Locally deployed 235b model configuration
# ============================================================================
LOCAL_MODEL_CLIENT = OpenAI(
    base_url="http://localhost:12346/v1",
    api_key="EMPTY"
)
LOCAL_MODEL_NAME = "235b"

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

## Original Question:
{question}

## Original Reasoning (with boxes to remove):
{original_reasoning}

## Rewritten Reasoning (without boxes, same facts, different expression):
"""

# ============================================================================
# Math verification prompt
# ============================================================================
MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def get_chat_template():
    """Return the chat template for answer judgement."""
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    """Return the few-shot examples."""
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""
    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""
    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""
    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""
    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""
    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""
    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""
    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_prompt(predict_str, ground_truth, question):
    """Build the answer-judgement prompt."""
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt


# ============================================================================
# Format-checking helpers
# ============================================================================
def check_format(response: str) -> bool:
    """Check whether the response format is correct."""
    pattern = re.compile(r"<thought>.*</thought>.*<answer>.*</answer>", re.DOTALL)
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
# Answer-correctness helpers
# ============================================================================
def rule_math_verify(ground_truth: str, model_answer: str) -> bool:
    """Validate a math answer with rules."""
    if not MATH_VERIFY_AVAILABLE:
        return False
    try:
        gold = parse(ground_truth)
        answer = parse(model_answer)
        return verify(gold, answer)
    except Exception:
        return False


def generative_verify(query: str, ground_truth: str, model_answer: str) -> bool:
    """Validate an answer with an LLM."""
    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )
    response = ""
    for it in range(3):
        try:
            chat_response = JUDGE_CLIENT.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f'[ERROR] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        return False


def check_accuracy(model_answer: str, ground_truth: str, question: str, use_math_verify: bool = False) -> bool:
    """Check answer correctness."""
    if len(model_answer) > 1000:
        return False

    # Try rule-based verification first
    if use_math_verify and rule_math_verify(ground_truth, model_answer):
        return True

    # Then fall back to LLM verification
    full_prompt = get_prompt(model_answer, ground_truth, question)
    for attempt in range(3):
        try:
            completion = JUDGE_CLIENT.chat.completions.create(
                model=JUDGE_MODEL_NAME,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.01
            )
            response = completion.choices[0].message.content.strip()
            score = float(response)
            return score == 1.0
        except:
            continue
    
    return False


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


def filter_low_accuracy_samples(samples: List[Dict[str, Any]], threshold: float = 0.4) -> List[Dict[str, Any]]:
    """Filter questions where group B or single-group accuracy <= threshold."""
    filtered = []
    for sample in samples:
        group = sample.get("group", "")
        rewards = sample.get("rewards", {})
        accuracy = rewards.get("accuracy", 0.0)
        
        if (group == "B" or group == "single") and accuracy <= threshold:
            filtered.append(sample)
        if len(filtered) >= 200:
            break
    print(f"Filtered {len(filtered)} samples with accuracy <= {threshold} (group B or single)")
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


def get_format_prompt(group: str, format_prompt_dir: str = "./examples/format_prompt") -> str:
    """Pick the appropriate format prompt by group."""
    if group == "B":
        # Group B uses the math prompt
        format_file = os.path.join(format_prompt_dir, "adagrounding_thyme_v31_thinklite_prompt.jinja")
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
# Image helpers
# ============================================================================
def load_image_as_base64(image_source: Union[str, dict, bytes], image_dir: Optional[str] = None) -> Optional[str]:
    """
    Load an image and base64-encode it.

    Args:
        image_source: Image source — path string, dict with `bytes`, or raw bytes.
        image_dir: Image directory (used for relative paths).

    Returns:
        Base64-encoded image string, or None on failure.
    """
    try:
        if isinstance(image_source, str):
            # Path string
            if image_dir and not os.path.isabs(image_source):
                image_path = os.path.join(image_dir, image_source)
            else:
                image_path = image_source

            with open(image_path, "rb") as f:
                image_bytes = f.read()
        elif isinstance(image_source, dict) and "bytes" in image_source:
            # Dict with bytes
            image_bytes = image_source["bytes"]
        elif isinstance(image_source, bytes):
            # Raw bytes
            image_bytes = image_source
        else:
            print(f"Warning: Unknown image source type: {type(image_source)}")
            return None

        # Encode as base64
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return base64_str
    except Exception as e:
        print(f"Warning: Failed to load image: {e}")
        return None


def build_vl_message_content(text: str, images: List[Any], image_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build the message content for a VL model (text + images).

    Args:
        text: Text content.
        images: Image list.
        image_dir: Image directory.

    Returns:
        Message content list in OpenAI VL API format.
    """
    content = []

    # Add images
    for image in images:
        base64_str = load_image_as_base64(image, image_dir)
        if base64_str:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_str}"
                }
            })

    # Add text
    content.append({
        "type": "text",
        "text": text
    })

    return content


# ============================================================================
# CoT rewriting and reasoning-generation helpers
# ============================================================================
def rewrite_reasoning_with_judge(original_reasoning: str, question: str, num_rewrites: int = 5) -> List[str]:
    """
    Rewrite the reasoning with the judge model (qwen-72B).
    Uses higher temperature to increase diversity.

    Args:
        original_reasoning: Original reasoning trace.
        question: Original question.
        num_rewrites: Number of rewrites to produce.
    """
    rewritten_list = []
    seen = set()  # Dedup set

    prompt = REWRITE_PROMPT.format(question=question, original_reasoning=original_reasoning)

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


def generate_with_local_model(
    question: str,
    ground_truth: str,
    format_prompt_template: str,
    images: List[Any] = None,
    image_dir: Optional[str] = None,
    num_samples: int = 3,
    temperature: float = 0.6,
    max_attempts: int = 30
) -> List[Tuple[str, str]]:
    """
    Generate reasoning chains and answers with the local 235b VL model.

    Args:
        question: Question text.
        ground_truth: Reference answer.
        format_prompt_template: Format-prompt template.
        images: Image list.
        image_dir: Image directory.
        num_samples: Number of samples to generate.
        temperature: Sampling temperature.
        max_attempts: Maximum number of attempts.

    Returns:
        List of (think_content, answer) tuples.
    """
    valid_samples = []

    # Build the prompt
    template = Template(format_prompt_template)
    user_prompt = template.render(content=question)

    # For qwen3vl
    user_prompt = user_prompt.replace("<think>", "<thought>").replace("</think>", "</thought>")

    # Build the message content (with images)
    if images and len(images) > 0:
        # VL model: use image-containing message format
        message_content = build_vl_message_content(user_prompt, images, image_dir)
    else:
        # Text-only model: use a simple string
        message_content = user_prompt

    attempts = 0
    while len(valid_samples) < num_samples and attempts < max_attempts:
        attempts += 1

        try:
            completion = LOCAL_MODEL_CLIENT.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=[{"role": "user", "content": message_content}],
                temperature=temperature,
                max_tokens=4096,
            )
            response = completion.choices[0].message.content.strip()

            # Check format
            if not check_format(response):
                print(f"  [Local Model] Attempt {attempts}: Format check failed")
                continue

            # Extract the answer
            answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip()

            # Check answer correctness
            if not check_accuracy(answer, ground_truth, question):
                print(f"  [Local Model] Attempt {attempts}: Accuracy check failed")
                continue

            # Extract think content
            think_content = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL).group(1).strip()
            valid_samples.append((think_content, answer))
            print(f"  [Local Model] Got valid sample #{len(valid_samples)} (attempt {attempts})")

        except Exception as e:
            print(f"Warning: Local model generation failed (attempt {attempts}): {e}")

    return valid_samples


def generate_single_samples_with_local_model(
    question: str,
    ground_truth: str,
    format_prompt_template: str,
    images: List[Any] = None,
    image_dir: Optional[str] = None,
    num_samples: int = 8,
    temperature: float = 0.6,
    max_attempts: int = 50
) -> List[Tuple[str, str]]:
    """
    Generate reasoning chains and answers for `single` samples using the local 235b VL model.

    Args:
        question: Question text.
        ground_truth: Reference answer.
        format_prompt_template: Format-prompt template.
        images: Image list.
        image_dir: Image directory.
        num_samples: Number of samples to generate.
        temperature: Sampling temperature.
        max_attempts: Maximum number of attempts.

    Returns:
        List of (think_content, answer) tuples.
    """
    return generate_with_local_model(
        question=question,
        ground_truth=ground_truth,
        format_prompt_template=format_prompt_template,
        images=images,
        image_dir=image_dir,
        num_samples=num_samples,
        temperature=temperature,
        max_attempts=max_attempts
    )


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
    response = f"<think>{rewritten_reasoning}</think><answer>{answer}</answer>"

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


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate SFT JSON samples from training rollouts")
    parser.add_argument("--json_log_dir", type=str, required=True, help="Directory containing JSON log files")
    parser.add_argument("--original_data_path", type=str, required=True, help="Path to original dataset (parquet or json)")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for SFT JSON file")
    parser.add_argument("--accuracy_threshold", type=float, default=0.4, help="Accuracy threshold for filtering")
    parser.add_argument("--num_rewrites", type=int, default=5, help="Number of CoT rewrites by judge model")
    parser.add_argument("--num_local_samples", type=int, default=3, help="Number of samples from local model for group B")
    parser.add_argument("--num_single_samples", type=int, default=8, help="Number of samples for single group")
    parser.add_argument("--local_temperature", type=float, default=0.6, help="Temperature for local 235b model")
    parser.add_argument("--format_prompt_dir", type=str, default="./examples/format_prompt", help="Directory containing format prompt templates")
    
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
    filtered_samples = filter_low_accuracy_samples(all_samples, args.accuracy_threshold)

    # 3. Load the original dataset
    print("\n" + "=" * 60)
    print("Step 3: Loading original dataset...")
    print("=" * 60)
    original_dataset = load_original_dataset(args.original_data_path, args.image_dir)

    # 4. Collect every original_idx that appears in the JSON logs
    print("\n" + "=" * 60)
    print("Step 4: Collecting trained sample indices...")
    print("=" * 60)

    trained_indices = set()
    for sample in all_samples:
        original_idx = sample.get("original_idx", -1)
        if original_idx >= 0:
            trained_indices.add(original_idx)
    print(f"Found {len(trained_indices)} unique samples that have been trained")

    # 5. Process each sample
    print("\n" + "=" * 60)
    print("Step 5: Processing samples...")
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
                # No qualifying Group A rollout — generate 8 samples with the local model
                print(f"[{original_idx}] No qualified Group A rollout, using local model for all samples...")
                total_samples_needed = args.num_rewrites + args.num_local_samples
                local_samples_gen = generate_with_local_model(
                    question=question,
                    ground_truth=ground_truth,
                    format_prompt_template=format_prompt_template,
                    images=original_data.get("images", []),
                    image_dir=args.image_dir,
                    num_samples=total_samples_needed,
                    temperature=args.local_temperature,
                    max_attempts=16
                )
                
                # Skip this sample if no valid generations came back
                if len(local_samples_gen) == 0:
                    print(f"[{original_idx}] [ABANDON] No valid samples obtained")
                    local_skipped += 1
                else:
                    for think_content_local, answer_local in local_samples_gen:
                        sft_sample = create_sft_sample(
                            original_data,
                            think_content_local,
                            answer_local,
                            "B",
                            format_prompt_template,
                            args.image_dir
                        )
                        result_samples.append(sft_sample)
                        local_processed += 1

                    print(f"[{original_idx}] Group B (all local): {len(local_samples_gen)} samples")
            else:
                # Extract Group A's think content and answer
                group_a_response = best_group_a.get("response", "")
                think_content = extract_think_content(group_a_response)
                answer_match = re.search(r"<answer>(.*?)</answer>", group_a_response, re.DOTALL)
                answer = answer_match.group(1).strip() if answer_match else ""

                if think_content is None:
                    # Failed to extract think content — fall back to 8 local-model samples
                    print(f"[{original_idx}] No think content in Group A, using local model...")
                    total_samples_needed = args.num_rewrites + args.num_local_samples
                    local_samples_gen = generate_with_local_model(
                        question=question,
                        ground_truth=ground_truth,
                        format_prompt_template=format_prompt_template,
                        images=original_data.get("images", []),
                        image_dir=args.image_dir,
                        num_samples=total_samples_needed,
                        temperature=args.local_temperature,
                        max_attempts=20
                    )

                    # Skip this sample if no valid generations came back
                    if len(local_samples_gen) == 0:
                        print(f"[{original_idx}] [ABANDON] No valid samples obtained")
                        local_skipped += 1
                    else:
                        for think_content_local, answer_local in local_samples_gen:
                            sft_sample = create_sft_sample(
                                original_data,
                                think_content_local,
                                answer_local,
                                "B",
                                format_prompt_template,
                                args.image_dir
                            )
                            result_samples.append(sft_sample)
                            local_processed += 1

                        print(f"[{original_idx}] Group B (all local): {len(local_samples_gen)} samples")
                else:
                    # Part 1: rewrite CoT 5 times with the judge model
                    rewritten_list = rewrite_reasoning_with_judge(
                        think_content,
                        question=question,
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

                    # Part 2: generate 3 more samples with the local 235b model
                    local_samples_gen = generate_with_local_model(
                        question=question,
                        ground_truth=ground_truth,
                        format_prompt_template=format_prompt_template,
                        images=original_data.get("images", []),
                        image_dir=args.image_dir,
                        num_samples=args.num_local_samples,
                        temperature=args.local_temperature,
                        max_attempts=20
                    )

                    for think_content_local, answer_local in local_samples_gen:
                        sft_sample = create_sft_sample(
                            original_data,
                            think_content_local,
                            answer_local,
                            "B",
                            format_prompt_template,
                            args.image_dir
                        )
                        result_samples.append(sft_sample)
                        local_processed += 1

                    print(f"[{original_idx}] Group B: {len(rewritten_list)} rewritten + {len(local_samples_gen)} local")

        # ====================================================================
        # Handle single samples
        # ====================================================================
        if single_samples:
            print(f"[{original_idx}] Processing Single...")

            # Load the format prompt
            format_prompt_template = get_format_prompt("single", args.format_prompt_dir)

            # Generate 8 reasoning + answer pairs with the local 235b model
            local_samples_gen = generate_single_samples_with_local_model(
                question=question,
                ground_truth=ground_truth,
                format_prompt_template=format_prompt_template,
                images=original_data.get("images", []),
                image_dir=args.image_dir,
                num_samples=args.num_single_samples,
                temperature=args.local_temperature,
                max_attempts=20
            )

            # Skip this sample if no valid generations came back
            if len(local_samples_gen) == 0:
                print(f"[{original_idx}] [ABANDON] No valid samples obtained")
                local_skipped += 1
            else:
                for think_content_local, answer_local in local_samples_gen:
                    sft_sample = create_sft_sample(
                        original_data,
                        think_content_local,
                        answer_local,
                        "single",
                        format_prompt_template,
                        args.image_dir
                    )
                    result_samples.append(sft_sample)
                    local_processed += 1

                print(f"[{original_idx}] Single: {len(local_samples_gen)} samples")

        return result_samples, local_processed, local_skipped

    # Run in parallel with a thread pool
    num_workers = 32
    print(f"Using {num_workers} threads for parallel processing...")

    # Lock to protect result aggregation
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
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

    # 6. Save results
    print("\n" + "=" * 60)
    print("Step 6: Saving results...")
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
