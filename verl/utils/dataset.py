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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
import datasets
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import copy
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from qwen_vl_utils.vision_process import smart_resize
def process_target_boxes(target_instances, processed_images, original_width, original_height):
    current_width, current_height = processed_images.width, processed_images.height
    h_bar, w_bar = smart_resize(current_height, current_width)
    for i in range(len(target_instances)):
        box = target_instances[i]["bbox"]
        scale_width = current_width / original_width
        scale_height = current_height / original_height
        target_instances[i]["bbox"][0], target_instances[i]["bbox"][2] = round(box[0] * scale_width), round(box[2] * scale_width)
        target_instances[i]["bbox"][1], target_instances[i]["bbox"][3] = round(box[1] * scale_height), round(box[3] * scale_height)
    return copy.deepcopy(target_instances)

def process_groundingdino_boxes(gdino_boxes, processed_images, original_width, original_height):
    current_width, current_height = processed_images.width, processed_images.height
    h_bar, w_bar = smart_resize(current_height, current_width)
    for i in range(len(gdino_boxes)):
        box = gdino_boxes[i]
        scale_width = current_width / original_width
        scale_height = current_height / original_height
        gdino_boxes[i][0], gdino_boxes[i][2] = round(box[0] * scale_width), round(box[2] * scale_width)
        gdino_boxes[i][1], gdino_boxes[i][3] = round(box[1] * scale_height), round(box[3] * scale_height)

    return copy.deepcopy(gdino_boxes)

def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int], return_size: Optional[bool] = None
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    original_width, original_height = image.width, image.height
    width, height = original_width, original_height

    h_bar, w_bar = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels) # add here
    image = image.resize((w_bar, h_bar))

    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if return_size is not None:
        return image, original_width, original_height
    return image


def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()
            with open("./examples/format_prompt/adagrounding_thyme_v30_given_proposals_validation.jinja", encoding="utf-8") as f:
                self.format_prompt_validation = f.read()
            with open("./examples/format_prompt/adagrounding_thyme_v31_thinklite_prompt.jinja", encoding="utf-8") as f:
                self.format_prompt_thinklite = f.read()
            with open("./examples/format_prompt/treevgr.jinja", encoding="utf-8") as f:
                self.format_prompt_grounding = f.read()
            with open("./examples/format_prompt/adagrounding_thyme_mathdata.jinja", encoding="utf-8") as f:
                self.format_prompt_mathdata = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
        self.used_datasource = ["ours_arxivqa", "ours_mmk12",
    "ours_thinklite",
    "ours_tqa",
    "ours_virl",
    "ours_wemath_pro",
    "ours_wemath_std", "ours_docqa", "ours_infoqa"]
        # self.dataset = self._filter_task() # Keep only box data
        if len(self.dataset) > 10000:
            self.dataset = self._filter_unused_data()


        # Build the index mapping table for dual-group training.
        # thinklite data is not duplicated (math prompt only).
        # Other data is duplicated (Group A: grounding, Group B: math).
        self._build_index_mapping()

    def _build_index_mapping(self):
        """
        Build the index mapping table for dual-group training.

        thinklite data: not duplicated, math prompt only, group="single".
        Other data: duplicated into Group A (grounding) and Group B (math).

        Mapping structure: [(original_idx, group), ...]
        - group = "A": use the grounding prompt.
        - group = "B": use the math prompt (for dual-group comparison).
        - group = "single": standalone math task (thinklite data).
        """
        self.index_mapping = []

        for original_idx in range(len(self.dataset)):
            example = self.dataset[original_idx]
            datasource = example.get("datasource", "unknown")

            if datasource in self.used_datasource:
                # thinklite data is not duplicated; math prompt only
                self.index_mapping.append((original_idx, "single"))
            else:
                # Other data is duplicated: Group A (grounding) + Group B (math)
                self.index_mapping.append((original_idx, "A"))
                self.index_mapping.append((original_idx, "B"))

        # Statistics
        single_count = sum(1 for _, g in self.index_mapping if g == "single")
        dual_count = sum(1 for _, g in self.index_mapping if g in ["A", "B"])
        print(f"[Dataset] Index mapping built: total={len(self.index_mapping)}, "
              f"single(thinklite)={single_count}, dual(A+B)={dual_count}")

    def _filter_unused_data(self):
        def doc2len(doc):
            if doc["datasource"] == "treevgr" or doc["datasource"] in self.used_datasource:
                return True
            # if len(doc["target_instances"]) == 0 or doc["question_type"] == "math":
            #     return False
            # else:
            #     return True
            # if doc["datasource"] == "treevgr":
            #     return True
            # elif doc["datasource"] == "thyme":
            #     if len(doc["target_instances"]) >= 5:
            #         return True
            #     else:
            #         return False
            else:
                return False
            # if doc["used"] == 1:
            #     if doc["datasource"] == "thinklite":
            #         return True
            #     if len(doc["target_instances"]) == 0:
            #         return False
            #     else:
            #         return True
            # else:
            #     return False
            # if doc["used"] == 1 and doc["datasource"] in ["thinklite"]:
            #     return True 
            # else:
            #     return False

        dataframe = self.dataset.filter(
            lambda doc: doc2len(doc),
            num_proc=16,
            desc=f"Filtering textcot task",
        )
        return dataframe


    def _filter_task(self):
        def doc2len(doc):
            if "question_type" not in list(doc.keys()):
                return True
            if len(doc["target_instances"]) == 0 or doc["question_type"] in ["stem_science", "chart_analysis", "math"]:
                return False
            else:
                return True
        # def doc2len(doc):
        #     if "question_type" not in list(doc.keys()):
        #         return True
        #     if doc["question_type"] == "general_vqa":
        #         return True 
        #     else:
        #         return False

        dataframe = self.dataset.filter(
            lambda doc: doc2len(doc),
            num_proc=16,
            desc=f"Filtering textcot task",
        )
        return dataframe

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            system_prompt = format_prompt.render(content=prompt_str)
            # prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>\n")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
            # return [
                # {"role": "user", "content": content_list}]
            return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
    
    def _build_messages_grounding(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt_grounding.strip())
            # system_prompt = format_prompt.render(content=prompt_str)
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>\n")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
            return [
                {"role": "user", "content": content_list}]
            # return [{"role": "system", "content": system_prompt},
            #     {"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _build_messages_user_prompt(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # system_prompt = format_prompt.render(content=prompt_str)
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>\n")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
            return [
                {"role": "user", "content": content_list}]
            # return [{"role": "system", "content": system_prompt},
            #     {"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _build_messages_validation(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt_validation.strip())
            system_prompt = format_prompt.render(content=prompt_str)
            # prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>\n")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
            # return [
                # {"role": "user", "content": content_list}]
            return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
    
    def _build_messages_math(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str =  "<image>\n" + example["query"]
        
        format_prompt = Template(self.format_prompt_thinklite.strip())
        prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>\n")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
    
    def _build_messages_mathdata(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str =  "<image>\n" + example["query"]
        
        format_prompt = Template(self.format_prompt_mathdata.strip())
        prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>\n")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        # Length of the index mapping table.
        # thinklite data is not duplicated; other data is duplicated.
        return len(self.index_mapping)


    def compute_iou(self, box1, box2):
        """
        Compute the IoU between two boxes.
        Box format: [x1, y1, x2, y2].
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Compute the intersection region
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # No intersection
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        # Intersection area
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Areas of the two boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Union area
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = inter_area / union_area if union_area > 0 else 0.0

        return iou


    def perturb_box_with_iou_constraint(self, box, min_iou=0.75, max_attempts=100):
        """
        Randomly perturb a box so that the perturbed box's IoU with the original > min_iou.

        The IoU constraint bounds the perturbation magnitude.
        For IoU=0.75, perturbations are roughly within ±15% of the box dimensions.
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # To keep IoU > 0.75, restrict the perturbation range.
        # Use a conservative estimate: ±12% on width/height and ±10% on the center.
        max_width_change = 0.15
        max_height_change = 0.15
        max_center_shift = 0.10

        # max_width_change = 0.3  # Perturbation range B
        # max_height_change = 0.3
        # max_center_shift = 0.2

        # max_width_change = 0.5  # Largest perturbation range C
        # max_height_change = 0.5
        # max_center_shift = 0.3

        # max_width_change = 0.4  # D
        # max_height_change = 0.4
        # max_center_shift = 0.3

        # Randomly perturb width and height
        new_width = width * (1 + random.uniform(-max_width_change, max_width_change))
        new_height = height * (1 + random.uniform(-max_height_change, max_height_change))

        # Randomly perturb the center
        new_cx = cx + random.uniform(-max_center_shift * width, max_center_shift * width)
        new_cy = cy + random.uniform(-max_center_shift * height, max_center_shift * height)

        # Compute the new box coordinates
        new_x1 = new_cx - new_width / 2
        new_y1 = new_cy - new_height / 2
        new_x2 = new_cx + new_width / 2
        new_y2 = new_cy + new_height / 2
        
        new_box = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

        return new_box

    # def make_proposals(self, target_instances, width, height, gdino_boxes):
    #     # Save the original gt boxes
    #     original_gt_boxes = []
    #     for i in range(len(target_instances)):
    #         original_gt_boxes.append(target_instances[i]["bbox"])

    #     all_boxes = gdino_boxes
    #     if len(all_boxes) >= 30:
    #         all_boxes = all_boxes[:30]

    #     # Shuffle
    #     random.shuffle(all_boxes)

    #     # Find the index of each original gt box by IoU
    #     original_indices = []
    #     for gt_box in original_gt_boxes:
    #         max_iou = 0.0
    #         best_idx = -1

    #         # Iterate over all_boxes to find the one with the highest IoU against gt_box
    #         for idx, box in enumerate(all_boxes):
    #             iou = self.compute_iou(gt_box, box)
    #             if iou > max_iou:
    #                 max_iou = iou
    #                 best_idx = idx

    #         # Only consider the gt matched if max_iou > 0.75
    #         # if max_iou > 0.75:
    #         if max_iou > 0.5:
    #             original_indices.append(best_idx)
    #         # If no box with IoU > 0.75 is found, we can skip this gt or append -1 to mark it invalid.
    #         # Here we skip it — meaning the gt box has no corresponding match in all_boxes.
        
    #     proposal_str = "\nBounding Boxes:\n"
    #     for i in range(len(all_boxes)):
    #         proposal_str += f"<box{i}> {int(all_boxes[i][0])} {int(all_boxes[i][1])} {int(all_boxes[i][2])} {int(all_boxes[i][3])}\n"        

    #     return proposal_str, original_indices, len(all_boxes)

    def make_proposals(self, target_instances, width, height):
        # Save the original gt boxes
        original_gt_boxes = []
        for i in range(len(target_instances)):
            original_gt_boxes.append(target_instances[i]["bbox"])

        instances = []
        for i in range(len(target_instances)):
            instances.append(target_instances[i]["bbox"])

        # Step 1: randomly drop boxes
        # num_to_remove = int(random.uniform(0, 0.5) * len(instances))
        # instances = random.sample(instances, len(instances) - num_to_remove)
        n = len(instances)

        for i in range(n):
            instances[i] = self.perturb_box_with_iou_constraint(instances[i])

        if n <= 5:
            # If n < 3, generate enough to make total 6 (original + generated = 6)
            num_generate = 15 - n
            # num_generate = 30 - n
        else:
            # Otherwise, generate n new boxes (total 2n)
            num_generate = n
        # num_generate = 50 - n
        
        generated_boxes = []
        for _ in range(num_generate):
            # Generate random box: ensure x1 < x2, y1 < y2, and within bounds
            x1 = random.uniform(0, width * 0.8)  # Leave room for width
            y1 = random.uniform(0, height * 0.8)
            x2 = min(width, x1 + random.uniform(10, width * 0.5))  # Min size 10, max half width
            y2 = min(height, y1 + random.uniform(10, height * 0.5))
            generated_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        # Combine original and generated
        all_boxes = instances[:] + generated_boxes  # Copy to avoid modifying original
        
        # Shuffle
        random.shuffle(all_boxes)
        
        # Find the index of each original gt box by IoU
        original_indices = []
        for gt_box in original_gt_boxes:
            max_iou = 0.0
            best_idx = -1

            # Iterate over all_boxes to find the one with the highest IoU against gt_box
            for idx, box in enumerate(all_boxes):
                iou = self.compute_iou(gt_box, box)
                if iou > max_iou:
                    max_iou = iou
                    best_idx = idx

            # Only consider the gt matched if max_iou > 0.75
            if max_iou > 0.75:
            # if max_iou > 0.5:
                original_indices.append(best_idx)
            # If no box with IoU > 0.75 is found, we can skip this gt or append -1 to mark it invalid.
            # Here we skip it — meaning the gt box has no corresponding match in all_boxes.
        
        proposal_str = "\nBounding Boxes:\n"
        for i in range(len(all_boxes)):
            proposal_str += f"<box{i}> {int(all_boxes[i][0])} {int(all_boxes[i][1])} {int(all_boxes[i][2])} {int(all_boxes[i][3])}\n"        

        return proposal_str, original_indices, len(all_boxes)
    

    def make_proposal_points(self, target_instances, width, height):
        # Save the original gt boxes
        original_gt_boxes = []
        for i in range(len(target_instances)):
            original_gt_boxes.append(target_instances[i]["bbox"])

        instances = []
        for i in range(len(target_instances)):
            instances.append(target_instances[i]["bbox"])

        # Step 1: randomly drop boxes
        # num_to_remove = int(random.uniform(0, 0.5) * len(instances))
        # instances = random.sample(instances, len(instances) - num_to_remove)
        n = len(instances)
        
        # for i in range(n):
        #     instances[i] = self.perturb_box_with_iou_constraint(instances[i])

        if n <= 7:
            num_outside = 15 - n
        else:
            # Otherwise, generate n new boxes (total 2n)
            num_outside = n

        # Generate one random point inside each original box
        inside_points = []
        for box in original_gt_boxes:
            x1, y1, x2, y2 = box
            # Random point inside the box
            px = random.randint(x1, x2)
            py = random.randint(y1, y2)
            inside_points.append([px, py])
        # Helper: check whether a point lies inside any box
        def is_inside_any_box(px, py, boxes):
            for box in boxes:
                x1, y1, x2, y2 = box
                if x1 <= px <= x2 and y1 <= py <= y2:
                    return True
            return False

        # Generate random points outside all boxes
        outside_points = []
        max_attempts = 1000
        attempts = 0
        while len(outside_points) < num_outside and attempts < max_attempts:
            px = random.randint(0, width - 1)
            py = random.randint(0, height - 1)
            if not is_inside_any_box(px, py, original_gt_boxes):
                outside_points.append([px, py])
            attempts += 1

        # If we run out of attempts before generating enough outside points
        # (extreme case where boxes cover most of the image), fall back to border points
        while len(outside_points) < num_outside:
            # Random point on the image border
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                px, py = random.randint(0, width - 1), 0
            elif edge == 'bottom':
                px, py = random.randint(0, width - 1), height - 1
            elif edge == 'left':
                px, py = 0, random.randint(0, height - 1)
            else:
                px, py = width - 1, random.randint(0, height - 1)
            outside_points.append([px, py])
        
        all_points = inside_points + outside_points
        
        # Shuffle
        random.shuffle(all_points)
        original_indices = []
        for kk in range(len(inside_points)):
            for jj in range(len(all_points)):
                if inside_points[kk][0] == all_points[jj][0] and inside_points[kk][1] == all_points[jj][1]:
                    original_indices.append(jj)
        
        proposal_str = "\nPoints:\n"
        for i in range(len(all_points)):
            proposal_str += f"<point{i}> {int(all_points[i][0])} {int(all_points[i][1])}\n"        

        return proposal_str, original_indices, len(all_points)



    def __getitem__(self, index):
        # Look up the original index and group via the index mapping table
        original_idx, group = self.index_mapping[index]

        example: dict = self.dataset[original_idx]
        example = copy.deepcopy(example)  # Avoid mutating the original data

        # Add the dual-group-training identifier fields
        example["original_idx"] = original_idx
        example["group"] = group

        # if example["question_type"] is in ["stem_science", "chart_analysis", "general_vqa", "math"]
        if "question_type" not in list(example.keys()):
            example["question_type"] = "validation"
        if "target_instances" not in list(example.keys()):
            example["target_instances"] = []

        # Pick the prompt based on group
        if group == "A":
            # Group A: use the grounding prompt; require boxes inside <think>
            messages = self._build_messages_grounding(example)
            example["task"] = "grounding"
        elif group == "B":
            # Group B: use the math prompt; no boxes required (for dual-group comparison)
            messages = self._build_messages_math(example)
            example["task"] = "math_nobox"
        else:
            # group == "single": thinklite data, math prompt only
            messages = self._build_messages_mathdata(example)
            example["task"] = "math_single"

        if self.image_key in example:
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                tmp_img, original_width, original_height = process_image(image, self.min_pixels, self.max_pixels, return_size=True)
                processed_images.append(tmp_img)
            
            # cbzhang: should target_instances be resized?
            if "target_instances" in list(example.keys()) and len(example["target_instances"]) > 0:
                example["target_instances"] = process_target_boxes(example["target_instances"], processed_images[0], original_width, original_height)
                # gdino_boxes = process_groundingdino_boxes(example["grounding_dino"], processed_images[0], original_width, original_height)
                if example["task"] == "choice":
                    # object_proposals, gt_index, num_all_boxes = self.make_proposals(example["target_instances"], processed_images[0].width, processed_images[0].height)
                    # object_proposals, gt_index, num_all_boxes = self.make_proposals(example["target_instances"], processed_images[0].width, processed_images[0].height, gdino_boxes)
                    object_proposals, gt_index, num_all_boxes = self.make_proposal_points(example["target_instances"], processed_images[0].width, processed_images[0].height)
#                     messages[0]["content"][1]["text"] += object_proposals + """
# Given the image, a question, and some bounding boxes in the image, the model needs to carefully think about the question and then provide an answer. The bounding boxes are provided in the format <box{index}> x1 y1 x2 y2, where index is the number of the box. The model's thought process should be enclosed in <think> </think>, and the final answer should be enclosed in <answer> </answer>. In the reasoning process, when referring to objects in the image, the model must use the corresponding <box> tag (e.g., "the man <box0> is standing behind the car <box2>").
# """
                    messages[0]["content"][1]["text"] += object_proposals + """\nGiven an image, a question, and some point positions within the image, the model needs to carefully consider the question and then provide an answer. The points are provided in the format <point{index}> x y, where {index} is the number of the point. The model's thought process should be enclosed in <think> </think>, and the final answer should be enclosed in <answer> </answer>. When referring to a specific object or region in the image, the model must use the corresponding <point{index}> to localize it (e.g., \"the man at <point3> is standing behind the car <point10>\").\n**Example**:\nUser: What is the man doing?\nPoints:\n<point0> 100 372\n<point1> 378 920\n<point2> 19 179\n<point3> 90 718\n<point4> 175 65\nAssistant:\n<think>The man <point3> is standing in front of the car <point1>. There is another object <point0> to the right of the man <point3>. Based on the position and context, it appears that the man is standing and looking at the car.</think>\n<answer>The man is standing and looking at the car.</answer>"""
                    example["gt_box_index"] = gt_index
                    example["num_all_boxes"] = num_all_boxes
                else:
                    example["gt_box_index"] = []
                    example["num_all_boxes"] = 0
            else:
                example["gt_box_index"] = []
                example["num_all_boxes"] = 0
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": processed_images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        if "query" in list(example.keys()):
            example["question"] = example["query"]
        else:
            example["question"] = example.pop(self.prompt_key)
        
        if "datasource" not in list(example.keys()):
            example["datasource"] = "unknown"

        return example
