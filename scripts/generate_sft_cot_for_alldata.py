#!/usr/bin/env python3

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
os.environ["NO_PROXY"] = "localhost,10.20.33.239,10.20.34.132,127.0.0.1"
os.environ["no_proxy"] = "localhost,10.20.33.239,10.20.34.132,127.0.0.1"
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
LOCAL_MODEL_CLIENT_1 = OpenAI(
    base_url="http://localhost:12346/v1",
    api_key="EMPTY"
)
LOCAL_MODEL_NAME_1 = "235b"

# ============================================================================
# Second deployed model configuration (judge2)
# ============================================================================
LOCAL_MODEL_CLIENT_2 = OpenAI(
    base_url="http://10.20.34.132:12347/v1",
    api_key="EMPTY"
)
LOCAL_MODEL_NAME_2 = "judge2"

# Model list — used for random selection
COT_MODEL_CLIENTS = [
    (LOCAL_MODEL_CLIENT_1, LOCAL_MODEL_NAME_1),
    (LOCAL_MODEL_CLIENT_2, LOCAL_MODEL_NAME_2),
]


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
        pass
    
    print(f"Loaded {len(dataset_dict)} samples from original dataset")
    return dataset_dict


def get_format_prompt(group: str, format_prompt_dir: str = "./examples/format_prompt") -> str:
    """Pick the appropriate format prompt by group."""
    if group == "B":
        # Group B uses the thinklite prompt
        format_file = os.path.join(format_prompt_dir, "adagrounding_thyme_v31_thinklite_prompt.jinja")
    else:  # mathdata
        format_file = os.path.join(format_prompt_dir, "adagrounding_thyme_mathdata.jinja")

    if os.path.exists(format_file):
        with open(format_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Default format
        return "{{ content | trim }}"



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
    question_with_image = f"{question}"

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


def generate_and_validate_samples(
    question: str,
    ground_truth: str,
    format_prompt_template: str,
    images: List[Any] = None,
    image_dir: Optional[str] = None,
    num_samples: int = 8,
    temperature: float = 0.6,
    max_attempts: int = 40
) -> List[Tuple[str, str]]:
    """
    Generate reasoning chains and answers with the local 235b VL model, and validate
    both format and correctness.

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

    # For qwen3vl, swap the tag style
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
            # Randomly pick a model
            client, model_name = random.choice(COT_MODEL_CLIENTS)

            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": message_content}],
                temperature=temperature,
                max_tokens=8192,
            )
            response = completion.choices[0].message.content.strip()

            # Check format
            if not check_format(response):
                continue

            # Extract the answer
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            if not answer_match:
                continue
            answer = answer_match.group(1).strip()

            # Check answer correctness
            if not check_accuracy(answer, ground_truth, question):
                continue

            # Extract think content
            think_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
            if not think_match:
                continue
            think_content = think_match.group(1).strip()

            valid_samples.append((think_content, answer))

        except Exception as e:
            print(f"Warning: Local model generation failed (attempt {attempts}): {e}")

    return valid_samples


def process_single_sample(
    idx: int,
    data: Dict[str, Any],
    output_dir: str,
    format_prompt_dir: str,
    image_dir: Optional[str],
    num_samples: int,
    temperature: float,
    max_attempts: int
) -> Tuple[int, int]:
    """
    Process a single sample.

    Args:
        idx: Sample index.
        data: Sample data.
        output_dir: Output directory.
        format_prompt_dir: Format-prompt template directory.
        image_dir: Image directory.
        num_samples: Number of samples to generate per data point.
        temperature: Sampling temperature.
        max_attempts: Maximum number of attempts.

    Returns:
        (number of successful samples, whether the data point was skipped).
    """
    # Read the data
    question = data.get("question", "") or data.get("query", "")
    question = question.replace("<image>", "").strip()
    ground_truth = data.get("answer", "")
    images = data.get("images", [])
    datasource = data.get("datasource", "")

    # Pick a group based on datasource
    if datasource == "treevgr":
        group = "B"
    else:
        group = "mathdata"

    # Load the matching format prompt
    format_prompt_template = get_format_prompt(group, format_prompt_dir)

    # Call the local API to generate samples
    valid_samples = generate_and_validate_samples(
        question=question,
        ground_truth=ground_truth,
        format_prompt_template=format_prompt_template,
        images=images,
        image_dir=image_dir,
        num_samples=num_samples,
        temperature=temperature,
        max_attempts=max_attempts
    )

    # Create SFT samples
    sft_samples = []
    for think_content, answer in valid_samples:
        sft_sample = create_sft_sample(
            data,
            think_content,
            answer,
            group,
            format_prompt_template,
            image_dir
        )
        sft_samples.append(sft_sample)

    # Save to JSON file
    output_path = os.path.join(output_dir, f"{idx}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_samples, f, ensure_ascii=False, indent=2)

    return len(sft_samples), 0 if len(sft_samples) > 0 else 1


# ============================================================================
# Main entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate SFT JSON samples from original dataset")
    parser.add_argument("--original_data_path", type=str, required=True, help="Path to original dataset (parquet or json)")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for individual JSON files")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate per data point")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for local model")
    parser.add_argument("--max_attempts", type=int, default=16, help="Maximum attempts per sample")
    parser.add_argument("--format_prompt_dir", type=str, default="./examples/format_prompt", help="Directory containing format prompt templates")
    parser.add_argument("--num_workers", type=int, default=64, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the original dataset
    print("=" * 60)
    print("Loading original dataset...")
    print("=" * 60)
    original_dataset = load_original_dataset(args.original_data_path, args.image_dir)

    # Check existing files so we can skip them
    print("\n" + "=" * 60)
    print("Checking existing files...")
    print("=" * 60)
    existing_files = set()
    if os.path.exists(args.output_dir):
        for filename in os.listdir(args.output_dir):
            if filename.endswith(".json"):
                try:
                    idx = int(filename.replace(".json", ""))
                    existing_files.add(idx)
                except ValueError:
                    pass

    # Drop already-processed samples
    samples_to_process = {idx: data for idx, data in original_dataset.items() if idx not in existing_files}
    print(f"  Total samples in dataset: {len(original_dataset)}")
    print(f"  Already processed: {len(existing_files)}")
    print(f"  To be processed: {len(samples_to_process)}")

    # Process each sample
    print("\n" + "=" * 60)
    print(f"Processing {len(samples_to_process)} samples with {args.num_workers} workers...")
    print(f"Using 2 model endpoints for load balancing...")
    print("=" * 60)

    total_samples = 0
    skipped_count = 0

    # Lock to protect result aggregation
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks (only the unprocessed ones)
        future_to_idx = {
            executor.submit(
                process_single_sample,
                idx,
                data,
                args.output_dir,
                args.format_prompt_dir,
                args.image_dir,
                args.num_samples,
                args.temperature,
                args.max_attempts
            ): idx
            for idx, data in samples_to_process.items()
        }

        # Collect results
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing samples"):
            idx = future_to_idx[future]
            try:
                num_generated, was_skipped = future.result()
                with results_lock:
                    total_samples += num_generated
                    skipped_count += was_skipped
            except Exception as e:
                print(f"[{idx}] Error: {e}")
                with results_lock:
                    skipped_count += 1
                # Save an empty list
                output_path = os.path.join(args.output_dir, f"{idx}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    # Print statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total data points in dataset: {len(original_dataset)}")
    print(f"  Already processed (skipped): {len(existing_files)}")
    print(f"  Processed this run: {len(samples_to_process)}")
    print(f"  Total SFT samples generated: {total_samples}")
    print(f"  Data points with no valid samples: {skipped_count}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
