"""
Dual-group training reward function.

Group A: uses grounding prompt and emits boxes inside <think>.
- Reward = ACC + format + box reward

Group B: uses math prompt and does not emit boxes.
- Reward = ACC + format + consistency reward
- consistency reward: compared against the reasoning chain of the best Group A rollout.

Features:
- A global cache stores the best reference rollout per sample.
- The cache is updated dynamically during training to accumulate higher-quality references.
"""

import re
import json
import os
from typing import Any, Dict, List, Optional
from openai import OpenAI
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy
from datetime import datetime

# Disable timeout on import to avoid signal issues in multi-threaded environments
import math_verify
parse = lambda x: math_verify.parse(x, parsing_timeout=None)
verify = lambda x, y: math_verify.verify(x, y)


# ============================================================================
# Global cache: stores the best reference rollout for each original_idx.
# Structure: {original_idx: {"think_content": str, "accuracy": float, "box_reward": float, "step": int}}
# ============================================================================
_reference_cache: Dict[int, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_global_step = 0  # Tracks the training step

# ============================================================================
# JSON logging configuration
# ============================================================================
_json_log_dir: Optional[str] = None  # JSON log directory, loaded from config
_json_log_enabled: bool = True  # Whether JSON logging is enabled, loaded from config


def set_json_log_dir(log_dir: str):
    """Set the JSON log directory."""
    global _json_log_dir
    _json_log_dir = log_dir
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        print(f"[JSON Log] Log directory set to: {log_dir}")


def enable_json_log(enabled: bool = True):
    """Enable or disable JSON logging."""
    global _json_log_enabled
    _json_log_enabled = enabled
    print(f"[JSON Log] Logging {'enabled' if enabled else 'disabled'}")


def init_json_log_config(json_log_dir: Optional[str] = None, json_log_enabled: bool = True):
    """Initialize the JSON logging configuration from config."""
    global _json_log_dir, _json_log_enabled
    if json_log_dir is not None:
        _json_log_dir = json_log_dir
        os.makedirs(_json_log_dir, exist_ok=True)
    _json_log_enabled = json_log_enabled
    print(f"[JSON Log] Initialized: dir={_json_log_dir}, enabled={_json_log_enabled}")


def update_reference_cache(original_idx: int, think_content: str, accuracy: float, box_reward: float) -> bool:
    """
    Try to update the reference cache.

    Update conditions:
    1. Cache is empty (first time we see this sample).
    2. The new accuracy is higher.
    3. Accuracy is the same but box_reward is higher.

    Args:
        original_idx: Original sample index.
        think_content: Content of the think segment.
        accuracy: Accuracy (0 or 1).
        box_reward: Box reward (0-1).

    Returns:
        True if the cache was updated, False otherwise.
    """
    global _reference_cache, _global_step

    if think_content is None:
        return False

    with _cache_lock:
        current_cache = _reference_cache.get(original_idx)

        should_update = False
        if current_cache is None:
            # Cache is empty, add directly
            should_update = True
        else:
            # Compare quality: accuracy first, then box_reward
            cached_acc = current_cache["accuracy"]
            cached_box = current_cache["box_reward"]
            
            if accuracy > cached_acc:
                should_update = True
            elif accuracy == cached_acc and box_reward > cached_box:
                should_update = True
        
        if should_update:
            _reference_cache[original_idx] = {
                "think_content": think_content,
                "accuracy": accuracy,
                "box_reward": box_reward,
                "step": _global_step,
            }
            return True
        
        return False


def get_cached_reference(original_idx: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve the cached reference rollout.

    Args:
        original_idx: Original sample index.

    Returns:
        Cached reference info, or None if not present.
    """
    with _cache_lock:
        return _reference_cache.get(original_idx)


def get_cache_stats() -> Dict[str, Any]:
    """Return cache statistics."""
    with _cache_lock:
        if not _reference_cache:
            return {"size": 0, "avg_accuracy": 0, "avg_box_reward": 0}
        
        accuracies = [v["accuracy"] for v in _reference_cache.values()]
        box_rewards = [v["box_reward"] for v in _reference_cache.values()]
        
        return {
            "size": len(_reference_cache),
            "avg_accuracy": sum(accuracies) / len(accuracies),
            "avg_box_reward": sum(box_rewards) / len(box_rewards),
        }


def clear_reference_cache():
    """Clear the cache (can be called at the start of training)."""
    global _reference_cache
    with _cache_lock:
        _reference_cache.clear()
    print("[Reference Cache] Cache cleared")


def save_reference_cache(cache_path: str):
    """Save the reference cache to disk."""
    global _reference_cache, _global_step
    with _cache_lock:
        cache_data = {
            "cache": _reference_cache,
            "global_step": _global_step,
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[Reference Cache] Saved cache to {cache_path} (size={len(_reference_cache)}, step={_global_step})")


def load_reference_cache(cache_path: str):
    """Load the reference cache from disk."""
    global _reference_cache, _global_step
    if not os.path.exists(cache_path):
        print(f"[Reference Cache] Cache file not found at {cache_path}, starting with empty cache")
        return
    
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        with _cache_lock:
            _reference_cache = cache_data.get("cache", {})
            _global_step = cache_data.get("global_step", 0)
        
        print(f"[Reference Cache] Loaded cache from {cache_path} (size={len(_reference_cache)}, step={_global_step})")
    except Exception as e:
        print(f"[Reference Cache] Failed to load cache from {cache_path}: {e}, starting with empty cache")
        with _cache_lock:
            _reference_cache = {}
            _global_step = 0


def increment_global_step():
    """Increment the global step counter."""
    global _global_step
    _global_step += 1


def save_step_json_log(
    reward_inputs: List[Dict[str, Any]],
    scores: List[Dict[str, float]],
    step: int
):
    """
    Save the question, rollout, and reward score for each training step to a JSON file.

    Args:
        reward_inputs: List of inputs containing question and response.
        scores: List of corresponding reward scores.
        step: Current training step.
    """
    global _json_log_dir, _json_log_enabled

    if not _json_log_enabled or _json_log_dir is None:
        return

    # Build the log payload
    log_data = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(reward_inputs),
        "samples": []
    }
    
    for i, (reward_input, score) in enumerate(zip(reward_inputs, scores)):
        sample_data = {
            "index": i,
            "question": reward_input.get("question", ""),
            "ground_truth": reward_input.get("ground_truth", ""),
            "response": reward_input.get("response", ""),
            "group": reward_input.get("group", "unknown"),
            "original_idx": reward_input.get("original_idx", -1),
            "datasource": reward_input.get("datasource", ""),
            "rewards": {k: v for k, v in score.items() if not k.startswith("_")}
        }
        log_data["samples"].append(sample_data)
    
    # Save to file
    json_filename = f"step_{step:06d}.json"
    json_path = os.path.join(_json_log_dir, json_filename)
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"[JSON Log] Saved step {step} log to: {json_path}")
    except Exception as e:
        print(f"[JSON Log] Failed to save log for step {step}: {e}")


def rule_math_verify(ground_truth, model_answer):
    try:
        gold = parse(ground_truth)
        answer = parse(model_answer)
        return verify(gold, answer)
    except Exception as e:
        return False


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
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


# Consistency-evaluation prompt for dual-group training (legacy version, deprecated)
DUAL_GROUP_CONSISTENCY_PROMPT_V1 = """You are an expert at evaluating visual descriptions in reasoning chains.

I will provide two reasoning processes for the same visual question:
1. **Reference CoT**: A reasoning chain with visual grounding (bounding boxes). Assume its image descriptions are CORRECT.
2. **Target CoT**: A reasoning chain without bounding boxes.

Your task is to compare the **image content descriptions** in Target CoT against Reference CoT.

## Scoring Rules (mutually exclusive, check in order):

1. **Score 0.0**: Target CoT contains ANY contradiction or conflict with Reference CoT's image descriptions.

2. **Score 0.3**: Target CoT has NO contradiction, BUT it BOTH:
   - Contains descriptions/details NOT present in Reference CoT, AND
   - Missing some descriptions/details that ARE present in Reference CoT.

3. **Score 0.7**: Target CoT has NO contradiction, BUT ONE of the following:
   - Contains descriptions/details NOT present in Reference CoT, OR
   - Missing some descriptions/details that ARE present in Reference CoT.

4. **Score 1.0**: Target CoT's image descriptions are FULLY CONSISTENT with Reference CoT - no contradiction, no extra details, no missing details.

## Input:

**Question**: {question}

**Reference CoT (with boxes, assume correct)**: 
{reference_think}

**Target CoT (without boxes)**:
{target_think}

## Output:
Output ONLY the score (0.0, 0.3, 0.7, or 1.0), nothing else.

Score:"""


# Consistency-evaluation prompt for dual-group training (V2, with few-shot examples)
# DUAL_GROUP_CONSISTENCY_PROMPT = """You are an expert at evaluating visual descriptions in reasoning chains.

# ## Task
# Compare the image descriptions/clues in Target CoT against Reference CoT (which is assumed to be CORRECT).

# ## Scoring Rules:
# - **Score 0.0**: Target CoT has ANY contradiction or conflict with Reference CoT's image descriptions.
# - **Score 0.5**: Target CoT has NO clear contradiction, but contains different information that cannot be verified as consistent or contradictory.
# - **Score 1.0**: Target CoT's image descriptions are FULLY CONSISTENT with Reference CoT.

# ## Few-Shot Examples:

# ### Example 1:
# **Question**: What color is the car on the left?
# **Reference CoT**: Looking at the image, I can see a red car <box>[100,200,300,400]</box> parked on the left side of the street. There is also a blue truck <box>[400,200,600,400]</box> on the right.
# **Target CoT**: The car on the left side of the street is red. I can also see a truck on the right side.
# **Score**: 1.0
# (Reason: Both describe the same visual content - red car on left, truck on right. Fully consistent.)

# ### Example 2:
# **Question**: What is the person doing?
# **Reference CoT**: The person <box>[150,100,250,350]</box> is sitting on a wooden bench and reading a book.
# **Target CoT**: The person is standing near the bench and looking at their phone.
# **Score**: 0.0
# (Reason: Clear contradiction - sitting vs standing, reading book vs looking at phone.)

# ### Example 3:
# **Question**: How many birds are in the image?
# **Reference CoT**: I can see three birds <box>[50,50,100,100]</box> <box>[120,60,170,110]</box> <box>[200,40,250,90]</box> flying in the sky.
# **Target CoT**: There are several birds flying in the blue sky with some clouds.
# **Score**: 0.5
# (Reason: No contradiction, but "several birds" and "clouds" are different/additional info that cannot be verified.)

# ### Example 4:
# **Question**: What is on the table?
# **Reference CoT**: On the table <box>[100,300,500,450]</box>, there is a red apple <box>[200,320,260,380]</box> and a glass of water <box>[300,310,350,390]</box>.
# **Target CoT**: The table has a green apple and a glass of juice on it.
# **Score**: 0.0
# (Reason: Contradiction - red apple vs green apple, water vs juice.)

# ### Example 5:
# **Question**: Where is the dog?
# **Reference CoT**: The dog <box>[300,400,450,550]</box> is lying on the grass near the tree <box>[100,200,200,500]</box>.
# **Target CoT**: I see a dog resting on the grass, close to a tree.
# **Score**: 1.0
# (Reason: Fully consistent - both describe dog on grass near tree.)

# ### Example 6:
# **Question**: What is the weather like?
# **Reference CoT**: The sky is clear and sunny based on the bright lighting in the image.
# **Target CoT**: It appears to be daytime. There might be some clouds in the distance.
# **Score**: 0.5
# (Reason: No direct contradiction about sunny weather, but "clouds in distance" is unverifiable additional info.)

# ## Your Task:

# **Question**: {question}

# **Reference CoT (assume correct)**: 
# {reference_think}

# **Target CoT**:
# {target_think}

# Output ONLY the score (0.0, 0.5, or 1.0), nothing else.

# Score:"""


# OpenAI client configuration (read from environment variables).
# Required: JUDGE_BASE_URL.
# Optional: JUDGE_API_KEY (default "EMPTY"), JUDGE_MODEL_NAME (default "judge").
# For a separate consistency-scoring endpoint, set CONSISTENCY_BASE_URL
# (and optionally CONSISTENCY_API_KEY); otherwise the judge client is reused.
client = OpenAI(
    base_url=os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("JUDGE_API_KEY", "EMPTY"),
)
model_name = os.environ.get("JUDGE_MODEL_NAME", "judge")

if "CONSISTENCY_BASE_URL" in os.environ:
    client_consistency = OpenAI(
        base_url=os.environ["CONSISTENCY_BASE_URL"],
        api_key=os.environ.get("CONSISTENCY_API_KEY", os.environ.get("JUDGE_API_KEY", "EMPTY")),
    )
else:
    client_consistency = client

def ours_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def get_number_of_boxes(predict_str: str, pattern: str) -> float:
    matches = re.findall(pattern, predict_str, re.DOTALL)
    return len(matches)


def ours_box_reward_grounding(predict_str: str, pattern: str) -> tuple:
    matches = re.findall(pattern, predict_str, re.DOTALL)
    
    valid_boxes_count = 0
    total_boxes_count = len(matches)
    all_boxes = []
    
    for match in matches:
        box = match.strip()
        coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
        coord_match = re.match(coord_pattern, box)
        
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            all_boxes.append((x1, y1, x2, y2))
            if x1 < x2 and y1 < y2:
                valid_boxes_count += 1
    
    if total_boxes_count == 0:
        return 0.0, 0.0

    def compute_iou(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    all_boxes = list(all_boxes)
    for i, t_coord in enumerate(all_boxes[:total_boxes_count // 2]):
        for j, p_coord in enumerate(all_boxes):
            iou = compute_iou(t_coord, p_coord) if i != j else 0
            if iou > 0.9:
                return 0.0, 0.0
    
    box_valid_reward = valid_boxes_count / total_boxes_count
    box_num_reward = total_boxes_count
    return box_valid_reward, box_num_reward


def ours_box_iou_reward_grounding(predict_str: str, target_instances: list, pattern: str) -> float:
    matches = re.findall(pattern, predict_str, re.DOTALL)
    all_boxes = []
    
    for match in matches:
        box = match.strip()
        coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
        coord_match = re.match(coord_pattern, box)
        
        if coord_match:
            x1, y1, x2, y2 = map(int, coord_match.groups())
            if x1 < x2 and y1 < y2:
                all_boxes.append([x1, y1, x2, y2])
    
    target_boxes = [instance["bbox"] for instance in target_instances if instance["bbox"] is not None]
    if len(target_boxes) == 0:
        return len(all_boxes) > 0

    def calculate_average_iou(pred_boxes, target_boxes):
        def compute_iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            inter_width = max(0, inter_x_max - inter_x_min)
            inter_height = max(0, inter_y_max - inter_y_min)
            inter_area = inter_width * inter_height
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area
            return inter_area / union_area if union_area > 0 else 0.0

        pred_coords = pred_boxes
        target_coords = target_boxes
        total_iou = 0.0
        num_targets = len(target_boxes)
        if num_targets == 0:
            return 0.0

        for t_coord in target_coords:
            best_iou = 0.0
            for p_coord in pred_coords:
                iou = compute_iou(t_coord, p_coord)
                if iou > best_iou:
                    best_iou = iou
            total_iou += best_iou

        return total_iou / num_targets

    iou_recall = calculate_average_iou(all_boxes, target_boxes)
    iou_precision = calculate_average_iou(target_boxes, all_boxes)
    return 0.5 * (iou_recall + iou_precision)


def get_prompt(predict_str, ground_truth, question):
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


def ours_acc_reward(predict_str: str, ground_truth: str, question: str, cot: str) -> float:
    while True:
        if len(predict_str) > 1000:
            return 0.0
        full_prompt = get_prompt(predict_str, ground_truth, question)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.01
        )
        try:
            score = float(completion.choices[0].message.content)
            return score
        except:
            print("[WARNING] Reward is not a scalar, call LLM again.")
            pass


def generative_verify(query, ground_truth, model_answer):
    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )
    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        return False


def ours_acc_reward_math(model_answer, ground_truth, question, cot):
    if rule_math_verify(ground_truth, model_answer):
        acc_reward = 1.0
    else:
        acc_reward = 1.0 if generative_verify(question, ground_truth, model_answer) else 0.0
    return acc_reward


def _parse_consistency_score(response: str, valid_scores: set) -> Optional[float]:
    """
    Parse the consistency score from an LLM response.

    Args:
        response: Response text from the LLM.
        valid_scores: Set of valid scores.

    Returns:
        The parsed score, or None if it cannot be parsed.
    """
    # Try to extract a number — matches the V1 scoring values: 0.0, 0.3, 0.7, 1.0
    score_match = re.search(r'(0\.0|0\.3|0\.7|1\.0|0|1)', response)
    if score_match:
        score_str = score_match.group(1)
        consistency_score = float(score_str)

        # Validate against the valid score set
        if consistency_score in valid_scores:
            return consistency_score
        elif consistency_score == 0:
            return 0.0
        elif consistency_score == 1:
            return 1.0
        else:
            # Round to the nearest valid score
            closest = min(valid_scores, key=lambda x: abs(x - consistency_score))
            return closest

    # If no valid score could be extracted, try a direct conversion
    try:
        consistency_score = float(response)
        return min(max(consistency_score, 0.0), 1.0)
    except ValueError:
        return None


def compute_dual_group_consistency(reference_think: str, target_think: str, question: str) -> float:
    """
    Compute the consistency reward between Group B's think segment and the best
    Group A rollout's think segment.

    Uses temperature 0.5 to generate 4 scores and returns their average.

    Scoring rubric (V1):
    - 0.0: Target CoT contradicts or conflicts with the reference CoT's image descriptions.
    - 0.3/0.7: Intermediate states.
    - 1.0: Fully consistent.

    Args:
        reference_think: Best Group A rollout's think segment (with boxes, assumed correct).
        target_think: Group B rollout's think segment (without boxes).
        question: Original question.

    Returns:
        consistency_score: Average of the 4 scores.
    """
    if reference_think is None or target_think is None:
        return 0.0

    full_prompt = DUAL_GROUP_CONSISTENCY_PROMPT_V1.format(
        question=question,
        reference_think=reference_think,
        target_think=target_think
    )

    valid_scores = {0.0, 0.3, 0.7, 1.0}  # Matches DUAL_GROUP_CONSISTENCY_PROMPT_V1
    num_samples = 4  # Generate 4 scores

    # Here we use the average of 4 scores to compute the consistency reward
    # Also, you may use 1 score to compute the consistency reward by setting num_samples = 1
    # with temperature 0.0 to generate 1 score, which is aligned with our submission
    for attempt in range(3):
        try:
            # Use n=4 to generate 4 responses in one call, temperature 0.5
            completion = client_consistency.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.5,
                n=num_samples,
            )

            # Parse the score from each response
            parsed_scores = []
            for choice in completion.choices:
                response = choice.message.content.strip()
                score = _parse_consistency_score(response, valid_scores)
                if score is not None:
                    parsed_scores.append(score)

            # Return the average if at least one score was parsed successfully
            if parsed_scores:
                avg_score = sum(parsed_scores) / len(parsed_scores)
                return avg_score

        except Exception as e:
            print(f"[WARNING] Consistency computation failed (attempt {attempt + 1}): {e}")
            continue

    return 0.0  # Return 0 if all attempts fail


def extract_think_content(predict_str: str) -> str:
    """Extract the content inside <think>...</think>."""
    match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer_content(predict_str: str) -> str:
    """Extract the content inside <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    return match.group(1).strip() if match else None


def _compute_group_a_score(reward_input: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the reward for Group A (grounding).

    Returns two parts:
    - Numeric fields used for metrics.
    - Internal fields (prefixed with `_`) used internally and stripped before metrics.
    """
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    ground_truth = reward_input["ground_truth"]
    format_reward = ours_format_reward(predict_str)
    question = reward_input["question"].replace("<image>", "").strip()

    # Extract think and answer
    think_cot = extract_think_content(predict_str)
    answer_ans = extract_answer_content(predict_str)

    # Compute box reward
    if len(reward_input.get("target_instances", [])) == 0:
        box_valid, box_num = 0.0, 0.0
        box_reward = 0.0
    else:
        box_valid, box_num = ours_box_reward_grounding(predict_str, pattern=r"<box>(.*?)</box>")
        box_reward = ours_box_iou_reward_grounding(predict_str, reward_input["target_instances"], pattern=r"<box>(.*?)</box>")

    # Compute accuracy
    accuracy = 0.0
    if answer_ans is not None:
        if reward_input.get("datasource", "") in ["thinklite"]:
            accuracy = ours_acc_reward_math(answer_ans, ground_truth, question, "")
        else:
            accuracy = ours_acc_reward(answer_ans, ground_truth, question, "")

    # Group A overall reward = format + box_valid + box_reward + accuracy
    overall_reward = format_reward + box_valid + box_reward + accuracy

    return {
        # Numeric fields - used for metrics
        "overall": overall_reward,
        "format": format_reward,
        "accuracy": accuracy,
        "box_reward": box_reward,
        "box_valid": box_valid,
        "box_num": box_num,
        "consistency": 0.0,  # Group A doesn't compute consistency, but the field is required
        "is_group_a": 1.0,  # Numeric group tag; 1.0 means Group A
        # Internal fields - used for consistency computation, removed before final return
        "_think_content": think_cot,
        "_answer_content": answer_ans,
    }


def _compute_group_b_score(reward_input: Dict[str, Any], reference_think: str = None) -> Dict[str, Any]:
    """Compute the reward for Group B (math / no-box).

    Returns two parts:
    - Numeric fields used for metrics.
    - Internal fields (prefixed with `_`) used internally and stripped before metrics.
    """
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    ground_truth = reward_input["ground_truth"]
    format_reward = ours_format_reward(predict_str)
    question = reward_input["question"].replace("<image>", "").strip()

    # Extract think and answer
    think_cot = extract_think_content(predict_str)
    answer_ans = extract_answer_content(predict_str)

    # Compute accuracy
    accuracy = 0.0
    if answer_ans is not None:
        if reward_input.get("datasource", "") in ["thinklite"]:
            accuracy = ours_acc_reward_math(answer_ans, ground_truth, question, "")
        else:
            accuracy = ours_acc_reward(answer_ans, ground_truth, question, "")

    # Compute consistency reward (only when reference_think is available)
    consistency_reward = 0.0
    consistency_from_llm = False  # Tracks whether the LLM was actually invoked
    if reference_think is not None and think_cot is not None:
        consistency_reward = compute_dual_group_consistency(reference_think, think_cot, question)
        consistency_from_llm = True  # The LLM was invoked

    # Group B overall reward = format + accuracy + consistency_reward
    # consistency_reward = consistency_reward * 0.0
    overall_reward = format_reward + accuracy + consistency_reward

    return {
        # Numeric fields - used for metrics
        "overall": overall_reward,
        "format": format_reward,
        "accuracy": accuracy,
        "consistency": consistency_reward,
        "box_reward": 0.0,
        "box_valid": 0.0,
        "box_num": 0.0,
        "is_group_a": 0.0,  # Numeric group tag; 0.0 means Group B
        # Internal fields - used for consistency computation, removed before final return
        "_think_content": think_cot,
        "_answer_content": answer_ans,
        "_consistency_from_llm": consistency_from_llm,  # Whether the LLM was invoked
    }


def _compute_single_score(reward_input: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the reward for `single` samples (e.g. thinklite).

    Only ACC + format are computed; no box reward or consistency reward.
    """
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
    ground_truth = reward_input["ground_truth"]
    format_reward = ours_format_reward(predict_str)
    question = reward_input["question"].replace("<image>", "").strip()

    # Extract think and answer
    think_cot = extract_think_content(predict_str)
    answer_ans = extract_answer_content(predict_str)

    # Compute accuracy
    accuracy = 0.0
    if answer_ans is not None:
        # thinklite data is validated as math
        accuracy = ours_acc_reward(answer_ans, ground_truth, question, "")

    # Overall reward for `single` = format + accuracy * 1.5 (matches the original math task)

    overall_reward = format_reward + accuracy * 1.5

    return {
        # Numeric fields - used for metrics
        "overall": overall_reward,
        "format": format_reward,
        "accuracy": accuracy,
        "consistency": 0.0,  # `single` doesn't compute consistency
        "box_reward": 0.0,
        "box_valid": 0.0,
        "box_num": 0.0,
        "is_group_a": -1.0,  # -1.0 indicates the `single` type
        # Internal fields
        "_think_content": think_cot,
        "_answer_content": answer_ans,
        "_consistency_from_llm": False,
    }


def ours_compute_score(reward_inputs: List[Dict[str, Any]], json_log_dir: Optional[str] = None, json_log_enabled: bool = True) -> List[Dict[str, float]]:
    """
    Reward computation function for dual-group training.

    For each sample (identified by original_idx):
    1. Group A rollouts: compute ACC + format + box reward.
    2. Find the Group A rollout with the highest reward and a correct answer.
    3. Group B rollouts: compute ACC + format + consistency reward (compared against the best Group A rollout).
    """
    # Initialize JSON logging config (passed in via kwargs)
    init_json_log_config(json_log_dir, json_log_enabled)

    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    print(f"[Dual Group Training] Processing {len(reward_inputs)} reward inputs")
    print(f"Sample question: {reward_inputs[0].get('question', 'N/A')[:100]}...")

    # Group by original_idx
    idx_to_inputs = defaultdict(list)
    for i, reward_input in enumerate(reward_inputs):
        original_idx = reward_input.get("original_idx", i)
        idx_to_inputs[original_idx].append((i, reward_input))

    print(f"[Dual Group Training] Found {len(idx_to_inputs)} unique samples")

    # Initialize results
    scores = [None] * len(reward_inputs)

    # Process each sample group in parallel
    def process_sample_group(original_idx, inputs_list):
        """Process all rollouts for one sample (Group A, Group B, and single).

        - Group A: grounding task; compute ACC + format + box reward.
        - Group B: math task (for comparison); compute ACC + format + consistency reward.
        - single: standalone math task (e.g. thinklite); compute ACC + format.

        Uses the global cache to store and retrieve the best reference rollout
        (only for Group A/B pairs).
        """
        group_a_inputs = []
        group_b_inputs = []
        single_inputs = []  # Data that only uses the math prompt, e.g. thinklite

        for idx, reward_input in inputs_list:
            group = reward_input.get("group", "A")
            if group == "A":
                group_a_inputs.append((idx, reward_input))
            elif group == "B":
                group_b_inputs.append((idx, reward_input))
            else:  # group == "single"
                single_inputs.append((idx, reward_input))

        results = {}

        # Step 1: Compute Group A scores
        group_a_scores = []
        for idx, reward_input in group_a_inputs:
            score = _compute_group_a_score(reward_input)
            results[idx] = score
            group_a_scores.append((idx, score))

        # Step 2: Find the qualified best Group A rollout in the current batch.
        # Criteria: accuracy == 1 and box_reward >= 0.5
        current_best = None
        if group_a_scores:
            # Ablation: select rollouts with correct answers and box_reward in [0.0, 0.1) as reference
            qualified_scores = [
                (idx, s) for idx, s in group_a_scores
                if s.get("accuracy", 0) == 1.0 and 0.3 <= s.get("box_reward", 0)
            ]

            if qualified_scores:
                # Among the qualified rollouts, pick the one with the highest box_reward
                best_idx, best_score = max(qualified_scores, key=lambda x: x[1]["box_reward"])
                current_best = {
                    "think_content": best_score.get("_think_content"),
                    "accuracy": best_score["accuracy"],
                    "box_reward": best_score["box_reward"],
                    "idx": best_idx,
                }

        # Step 3: Compare with the cache and update
        if current_best is not None and current_best["think_content"] is not None:
            updated = update_reference_cache(
                original_idx,
                current_best["think_content"],
                current_best["accuracy"],
                current_best["box_reward"]
            )
            if updated:
                print(f"[Sample {original_idx}] Cache UPDATED: acc={current_best['accuracy']:.1f}, "
                      f"box_reward={current_best['box_reward']:.3f}")

        # Step 4: Fetch the best reference from the cache (current batch or historical)
        best_reference_think = None
        cached = get_cached_reference(original_idx)

        if cached is not None:
            best_reference_think = cached["think_content"]
            # Determine whether the reference is from the cache or the current batch
            if current_best is not None and current_best["think_content"] == best_reference_think:
                print(f"[Sample {original_idx}] Using CURRENT batch reference: "
                      f"acc={cached['accuracy']:.1f}, box_reward={cached['box_reward']:.3f}")
            else:
                print(f"[Sample {original_idx}] Using CACHED reference: "
                      f"acc={cached['accuracy']:.1f}, box_reward={cached['box_reward']:.3f}, "
                      f"from step {cached['step']}")
        else:
            print(f"[Sample {original_idx}] No reference available (cache empty & no qualified rollout), "
                  f"Group B consistency will be 0")

        # Step 5: Compute Group B scores (using the cached reference_think)
        for idx, reward_input in group_b_inputs:
            score = _compute_group_b_score(reward_input, best_reference_think)
            results[idx] = score

        # Step 6: Compute scores for `single` samples (e.g. thinklite — only ACC + format)
        for idx, reward_input in single_inputs:
            score = _compute_single_score(reward_input)
            results[idx] = score

        return results

    # Process in parallel with a thread pool
    max_workers = min(64, len(idx_to_inputs) + 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_sample_group, original_idx, inputs_list): original_idx 
            for original_idx, inputs_list in idx_to_inputs.items()
        }
        
        for future in as_completed(future_to_idx):
            original_idx = future_to_idx[future]
            try:
                results = future.result()
                for idx, score in results.items():
                    scores[idx] = score
            except Exception as exc:
                print(f'[WARNING] Sample {original_idx} generated an exception: {exc}')
                # Set default scores (all numeric fields)
                for idx, reward_input in idx_to_inputs[original_idx]:
                    is_group_a = 1.0 if reward_input.get("group", "A") == "A" else 0.0
                    scores[idx] = {
                        "overall": 0.0,
                        "format": 0.0,
                        "accuracy": 0.0,
                        "box_reward": 0.0,
                        "box_valid": 0.0,
                        "box_num": 0.0,
                        "consistency": 0.0,
                        "is_group_a": is_group_a,
                    }
    
    # Increment the global step counter (used to track cache update times)
    increment_global_step()

    # Print statistics
    group_a_scores = [s for s in scores if s and s.get("is_group_a", 1.0) == 1.0]
    group_b_scores = [s for s in scores if s and s.get("is_group_a", 1.0) == 0.0]
    single_scores = [s for s in scores if s and s.get("is_group_a", 1.0) == -1.0]  # `single` type

    # Print batch composition
    print(f"[Dual Group Training] Batch composition: "
          f"Group A={len(group_a_scores)}, Group B={len(group_b_scores)}, Single={len(single_scores)}, "
          f"Total={len(scores)}")
    
    if group_a_scores:
        avg_a_overall = sum(s["overall"] for s in group_a_scores) / len(group_a_scores)
        avg_a_format = sum(s.get("format", 0) for s in group_a_scores) / len(group_a_scores)
        avg_a_acc = sum(s.get("accuracy", 0) for s in group_a_scores) / len(group_a_scores)
        avg_a_box = sum(s.get("box_reward", 0) for s in group_a_scores) / len(group_a_scores)
        avg_a_box_valid = sum(s.get("box_valid", 0) for s in group_a_scores) / len(group_a_scores)
        print(f"[Dual Group Training] Group A: count={len(group_a_scores)}, "
              f"overall={avg_a_overall:.3f}, format={avg_a_format:.3f}, acc={avg_a_acc:.3f}, "
              f"box_reward={avg_a_box:.3f}, box_valid={avg_a_box_valid:.3f}")
    
    if group_b_scores:
        avg_b_overall = sum(s["overall"] for s in group_b_scores) / len(group_b_scores)
        avg_b_format = sum(s.get("format", 0) for s in group_b_scores) / len(group_b_scores)
        avg_b_acc = sum(s.get("accuracy", 0) for s in group_b_scores) / len(group_b_scores)
        avg_b_consistency = sum(s.get("consistency", 0) for s in group_b_scores) / len(group_b_scores)
        
        # Track only consistency scores actually produced by the LLM
        # (excluding those set to 0 because no reference was available)
        llm_consistency_scores = [
            s.get("consistency", 0) for s in group_b_scores 
            if s.get("_consistency_from_llm", False)
        ]
        print(f"[Dual Group Training] Group B: count={len(group_b_scores)}, "
              f"overall={avg_b_overall:.3f}, format={avg_b_format:.3f}, acc={avg_b_acc:.3f}, "
              f"consistency={avg_b_consistency:.3f}")
        if llm_consistency_scores:
            avg_llm_consistency = sum(llm_consistency_scores) / len(llm_consistency_scores)
            print(f"[Dual Group Training] LLM Judge consistency: count={len(llm_consistency_scores)}, "
                  f"avg_score={avg_llm_consistency:.3f} (excluding {len(group_b_scores) - len(llm_consistency_scores)} skipped)")
        else:
            print(f"[Dual Group Training] LLM Judge consistency: no LLM calls (all skipped due to no reference)")
    
    # Print stats for `single` (e.g. thinklite)
    if single_scores:
        avg_single_overall = sum(s["overall"] for s in single_scores) / len(single_scores)
        avg_single_format = sum(s.get("format", 0) for s in single_scores) / len(single_scores)
        avg_single_acc = sum(s.get("accuracy", 0) for s in single_scores) / len(single_scores)
        print(f"[Dual Group Training] Single (thinklite): count={len(single_scores)}, "
              f"overall={avg_single_overall:.3f}, format={avg_single_format:.3f}, acc={avg_single_acc:.3f}")
    
    # Print cache statistics
    cache_stats = get_cache_stats()
    print(f"[Reference Cache] size={cache_stats['size']}, "
          f"avg_accuracy={cache_stats['avg_accuracy']:.3f}, "
          f"avg_box_reward={cache_stats['avg_box_reward']:.3f}")
    
    # Strip internal fields (those starting with `_`); keep only numeric fields for metrics
    cleaned_scores = []
    for score in scores:
        if score is None:
            cleaned_scores.append({"overall": 0.0, "format": 0.0, "accuracy": 0.0, 
                                   "box_reward": 0.0, "box_valid": 0.0, "box_num": 0.0,
                                   "consistency": 0.0, "is_group_a": 1.0})
        else:
            cleaned_score = {k: v for k, v in score.items() if not k.startswith("_")}
            cleaned_scores.append(cleaned_score)
    
    # ============================================================================
    # Compute per-group metrics (for wandb logging)
    # ============================================================================
    # Count the size of each group (for logging)
    num_group_a = len([s for s in scores if s and s.get("is_group_a", 1.0) == 1.0])
    num_group_b = len([s for s in scores if s and s.get("is_group_a", 1.0) == 0.0])
    num_single = len([s for s in scores if s and s.get("is_group_a", 1.0) == -1.0])
    total_samples = len(scores)

    # Add prefixed metrics for each group.
    # Note: only add a group's metrics to samples belonging to that group; do not add fields
    # from other groups. This way wandb only averages samples that actually have values.
    for idx, cleaned_score in enumerate(cleaned_scores):
        score_obj = scores[idx] if scores[idx] is not None else {}
        is_group_a = score_obj.get("is_group_a", 1.0)

        # Record group counts on every sample (so wandb can monitor batch composition)
        cleaned_score["batch/num_group_a"] = float(num_group_a)
        cleaned_score["batch/num_group_b"] = float(num_group_b)
        cleaned_score["batch/num_single"] = float(num_single)
        cleaned_score["batch/total"] = float(total_samples)

        if is_group_a == 1.0:
            # Group A metrics (grounding task) - add only group_a fields
            cleaned_score["group_a/overall"] = cleaned_score["overall"]
            cleaned_score["group_a/format"] = cleaned_score["format"]
            cleaned_score["group_a/accuracy"] = cleaned_score["accuracy"]
            cleaned_score["group_a/box_reward"] = cleaned_score["box_reward"]
            cleaned_score["group_a/box_valid"] = cleaned_score["box_valid"]
            cleaned_score["group_a/box_num"] = cleaned_score["box_num"]
            cleaned_score["group_a/consistency"] = 0.0  # Group A doesn't compute consistency
        elif is_group_a == 0.0:
            # Group B metrics (math task with consistency) - add only group_b fields
            cleaned_score["group_b/overall"] = cleaned_score["overall"]
            cleaned_score["group_b/format"] = cleaned_score["format"]
            cleaned_score["group_b/accuracy"] = cleaned_score["accuracy"]
            cleaned_score["group_b/consistency"] = cleaned_score["consistency"]
            cleaned_score["group_b/box_reward"] = 0.0  # Group B doesn't compute box reward
            cleaned_score["group_b/box_valid"] = 0.0
            cleaned_score["group_b/box_num"] = 0.0
        else:  # is_group_a == -1.0 (single)
            # Single metrics (standalone math task like thinklite) - add only single fields
            cleaned_score["single/overall"] = cleaned_score["overall"]
            cleaned_score["single/format"] = cleaned_score["format"]
            cleaned_score["single/accuracy"] = cleaned_score["accuracy"]

    # ============================================================================
    # Save JSON log
    # ============================================================================
    save_step_json_log(reward_inputs, cleaned_scores, _global_step)
    
    print(f"[Dual Group Training] Sample output: {cleaned_scores[0] if cleaned_scores else 'None'}")
    
    return cleaned_scores


if __name__ == "__main__":
    # Smoke test for the dual-group training reward
    print("=== Dual-group training reward test ===")
    
    test_inputs = [
        # Group A rollouts (grounding, with boxes)
        {
            "response": "<think>Looking at the image, I can see a red ball <box>[100,100,200,200]</box> on the left side.</think><answer>red</answer>",
            "ground_truth": "red",
            "question": "What color is the ball on the left?",
            "target_instances": [{"bbox": [100, 100, 200, 200]}],
            "original_idx": 0,
            "group": "A",
            "datasource": "test",
        },
        {
            "response": "<think>The ball <box>[95,105,195,195]</box> appears to be red in color.</think><answer>red</answer>",
            "ground_truth": "red",
            "question": "What color is the ball on the left?",
            "target_instances": [{"bbox": [100, 100, 200, 200]}],
            "original_idx": 0,
            "group": "A",
            "datasource": "test",
        },
        # Group B rollouts (math, no boxes)
        {
            "response": "<think>Looking at the left side of the image, there is a ball that appears to be red.</think><answer>red</answer>",
            "ground_truth": "red",
            "question": "What color is the ball on the left?",
            "target_instances": [{"bbox": [100, 100, 200, 200]}],
            "original_idx": 0,
            "group": "B",
            "datasource": "test",
        },
        {
            "response": "<think>The ball on the left side is blue.</think><answer>blue</answer>",
            "ground_truth": "red",
            "question": "What color is the ball on the left?",
            "target_instances": [{"bbox": [100, 100, 200, 200]}],
            "original_idx": 0,
            "group": "B",
            "datasource": "test",
        },
    ]
    
    scores = ours_compute_score(test_inputs)
    
    for i, (inp, score) in enumerate(zip(test_inputs, scores)):
        group_label = "A" if score.get('is_group_a', 1.0) == 1.0 else "B"
        print(f"\n--- Rollout {i} (Group {group_label}) ---")
        print(f"Response: {inp['response'][:80]}...")
        print(f"Scores: overall={score['overall']:.3f}, format={score['format']:.3f}, "
              f"accuracy={score['accuracy']:.3f}, box_reward={score.get('box_reward', 0):.3f}, "
              f"consistency={score.get('consistency', 0):.3f}")

