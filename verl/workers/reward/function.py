# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                    "question": data.non_tensor_batch["question"][i], #cbzhang
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        decoded_responses = []  # Store decoded strings to avoid re-decoding
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            decoded_responses.append(response_str)
            
            # Build the base reward_input
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
                "question": data.non_tensor_batch["question"][i], #cbzhang
                "target_instances": data.non_tensor_batch["target_instances"][i], # cbzhang;
                "question_type": data.non_tensor_batch["question_type"][i],
                "gt_box_index": data.non_tensor_batch["gt_box_index"][i], #cbzhang
                "num_all_boxes": data.non_tensor_batch["num_all_boxes"][i],
                "datasource": data.non_tensor_batch["datasource"][i],
                "task": data.non_tensor_batch["task"][i],
            }
            
            # Add dual-group-training fields if present
            if "uid" in data.non_tensor_batch:
                reward_input["uid"] = data.non_tensor_batch["uid"][i]
            if "group" in data.non_tensor_batch:
                reward_input["group"] = data.non_tensor_batch["group"][i]
            if "original_idx" in data.non_tensor_batch:
                reward_input["original_idx"] = data.non_tensor_batch["original_idx"][i]
            
            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        
        if len(scores) > 0 and "adagrounding" in scores[0] and scores[0]["adagrounding"] == 1 and \
            "uid" in list(data.non_tensor_batch.keys()): 
            # cbzhang, only for adagrounding: for rollouts that answered correctly without a box, replace box-iou and box-valid with the group mean
            id2score = defaultdict(list)
            bsz = len(scores)
            index = data.non_tensor_batch["uid"]

            # Group rollouts by uid
            for i in range(bsz):
                id2score[index[i]].append((i, scores[i]))

            # Process rollouts within each group
            for uid, rollouts in id2score.items():
                # Split rollouts into box_num > 0 and box_num == 0
                valid_box_rollouts = [(idx, score) for idx, score in rollouts if score["box_num"] > 0]
                zero_box_rollouts = [(idx, score) for idx, score in rollouts if score["box_num"] == 0]

                if valid_box_rollouts and zero_box_rollouts:
                    # Mean of box_iou and box_valid across rollouts that have boxes
                    box_iou_mean = sum(score["box_iou"] for _, score in valid_box_rollouts) / len(valid_box_rollouts)
                    box_valid_mean = sum(score["box_valid"] for _, score in valid_box_rollouts) / len(valid_box_rollouts)

                    # Recompute overall for box_num==0 rollouts
                    for idx, score in zero_box_rollouts:
                        accuracy = score["accuracy"]
                        format_score = score["format"]
                        new_overall = accuracy * 1.5 + format_score + accuracy * box_iou_mean + accuracy * box_valid_mean
                        scores[idx]["overall"] = new_overall
            



        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        
        # Prepare segment-info tensors if available
        segment1_rewards = []
        segment2_rewards = []
        draft_end_positions = []
        
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
            
            # Collect segment info if present
            if "segment1_reward" in score:
                segment1_rewards.append(score["segment1_reward"])
            if "segment2_reward" in score:
                segment2_rewards.append(score["segment2_reward"])
            if "draft_end_pos" in score:
                draft_end_positions.append(score["draft_end_pos"])
        
        # Convert segment info into tensors and add them to the result
        result_dict = dict(reward_metrics)
        if segment1_rewards and segment2_rewards and draft_end_positions:
            # Convert character positions to token positions
            token_draft_end_positions = []
            for i, (char_pos, response_str) in enumerate(zip(draft_end_positions, decoded_responses)):
                if char_pos == -1:
                    # Format error, mark with -1
                    token_draft_end_positions.append(-1)
                else:
                    # Convert character position to token position
                    token_pos = self._char_to_token_position(
                        response_str, 
                        char_pos, 
                        response_ids[i][:int(response_length[i].item())]
                    )
                    token_draft_end_positions.append(token_pos)
            
            result_dict["_segment1_rewards_tensor"] = torch.tensor(segment1_rewards, dtype=torch.float32)
            result_dict["_segment2_rewards_tensor"] = torch.tensor(segment2_rewards, dtype=torch.float32)
            result_dict["_draft_end_positions_tensor"] = torch.tensor(token_draft_end_positions, dtype=torch.long)
            

        return reward_tensor, result_dict
    
    def _char_to_token_position(self, response_str: str, char_pos: int, token_ids: torch.Tensor) -> int:
        """
        Convert a character position to a token position.

        Args:
            response_str: Decoded response string.
            char_pos: Character position.
            token_ids: Token sequence.

        Returns:
            Corresponding token position.
        """
        try:
            # Decode tokens one by one to find the matching character position
            current_char_pos = 0
            for token_idx in range(len(token_ids)):
                # Decode the current token
                current_token = self.tokenizer.decode([token_ids[token_idx]], skip_special_tokens=self.config.skip_special_tokens)

                # Update the character position
                current_char_pos += len(current_token)

                # Once the running char position meets or exceeds the target, return the current token position
                if current_char_pos >= char_pos:
                    return token_idx + 1  # +1 because we want the position right after the draft ends

            # If not found, return the end of the sequence
            return len(token_ids)

        except Exception as e:
            print(f"[WARNING] Error converting char pos to token pos: {e}")
            # Fallback: estimate the token position
            estimated_tokens_per_char = len(token_ids) / max(len(response_str), 1)
            estimated_token_pos = int(char_pos * estimated_tokens_per_char)
            return min(estimated_token_pos, len(token_ids))
