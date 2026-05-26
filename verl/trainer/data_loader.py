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

from typing import Optional, Iterator

import torch
from torch.utils.data import RandomSampler, SequentialSampler, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


class PairedRandomSampler(Sampler[int]):
    """
    Custom sampler for dual-group training.
    Keeps paired indices (0,1), (2,3), (4,5), ... together.

    How it works:
    1. Group indices into paired units: [(0,1), (2,3), (4,5), ...].
    2. Randomly shuffle the paired units.
    3. Flatten into the final index sequence.
    """

    def __init__(self, data_source, generator=None, pair_size: int = 2):
        self.data_source = data_source
        self.generator = generator
        self.pair_size = pair_size
        self._num_samples = len(data_source)

        # Dataset size must be a multiple of pair_size
        assert self._num_samples % pair_size == 0, \
            f"Dataset size ({self._num_samples}) must be divisible by pair_size ({pair_size})"

    def __iter__(self) -> Iterator[int]:
        n = self._num_samples
        num_pairs = n // self.pair_size

        # Generate indices of paired units
        if self.generator is None:
            pair_indices = torch.randperm(num_pairs).tolist()
        else:
            pair_indices = torch.randperm(num_pairs, generator=self.generator).tolist()

        # Flatten into the final index sequence
        for pair_idx in pair_indices:
            start_idx = pair_idx * self.pair_size
            for offset in range(self.pair_size):
                yield start_idx + offset
    
    def __len__(self) -> int:
        return self._num_samples


class MixedPairedRandomSampler(Sampler[int]):
    """
    Mixed sampler for dual-group training combined with `single` data.

    Behavior:
    1. Sample from paired Group A/B at the given ratio (e.g. 70%).
    2. Sample from `single` for the remaining ratio (e.g. 30%).
    3. Always keep paired A/B together.

    How it works:
    1. Split index_mapping into paired indices (A/B) and single indices.
    2. Allocate by ratio for each epoch.
    3. Paired samples are guaranteed to appear together.
    """

    def __init__(self, data_source, index_mapping, generator=None,
                 paired_ratio: float = 0.7, batch_size: int = 512):
        self.data_source = data_source
        self.index_mapping = index_mapping
        self.generator = generator
        self.paired_ratio = paired_ratio
        self.batch_size = batch_size

        # Separate paired and single indices
        self.paired_indices = []  # [(idx_A, idx_B), ...]
        self.single_indices = []  # [idx, ...]

        # Group by original_idx to find paired A/B
        from collections import defaultdict
        idx_groups = defaultdict(dict)
        for i, (original_idx, group) in enumerate(index_mapping):
            idx_groups[original_idx][group] = i

        for original_idx, groups in idx_groups.items():
            if "A" in groups and "B" in groups:
                self.paired_indices.append((groups["A"], groups["B"]))
            if "single" in groups:
                self.single_indices.append(groups["single"])

        # Number of paired and single samples per batch.
        # paired takes up paired_ratio, with each pair contributing 2 samples.
        # single takes up 1 - paired_ratio.
        self.num_paired_per_batch = int(batch_size * paired_ratio) // 2  # Divide by 2 since each pair has 2 samples
        self.num_single_per_batch = batch_size - self.num_paired_per_batch * 2

        # Total sample count (ensure full batches can be formed).
        # Use all paired samples per epoch, then top up with single samples by ratio.
        self.num_pairs = len(self.paired_indices)
        self.num_singles = len(self.single_indices)

        # Compute the number of full batches that can be formed
        if self.num_paired_per_batch > 0 and self.num_pairs > 0:
            # Compute batches based on the paired sample count
            self.num_batches = self.num_pairs // self.num_paired_per_batch
        else:
            self.num_batches = 0
        
        self._num_samples = self.num_batches * batch_size
        
        print(f"[MixedPairedRandomSampler] paired_indices={len(self.paired_indices)}, "
              f"single_indices={len(self.single_indices)}")
        print(f"[MixedPairedRandomSampler] per_batch: paired={self.num_paired_per_batch}*2, "
              f"single={self.num_single_per_batch}")
        print(f"[MixedPairedRandomSampler] num_batches={self.num_batches}, "
              f"total_samples={self._num_samples}")
    
    def __iter__(self) -> Iterator[int]:
        # Shuffle paired indices
        if self.generator is None:
            paired_perm = torch.randperm(len(self.paired_indices)).tolist()
            single_perm = torch.randperm(len(self.single_indices)).tolist()
        else:
            paired_perm = torch.randperm(len(self.paired_indices), generator=self.generator).tolist()
            single_perm = torch.randperm(len(self.single_indices), generator=self.generator).tolist()

        # Build each batch
        paired_ptr = 0
        single_ptr = 0

        for batch_idx in range(self.num_batches):
            batch_indices = []

            # Add paired samples
            for _ in range(self.num_paired_per_batch):
                if paired_ptr < len(paired_perm):
                    pair_idx = paired_perm[paired_ptr]
                    idx_a, idx_b = self.paired_indices[pair_idx]
                    batch_indices.append(idx_a)
                    batch_indices.append(idx_b)
                    paired_ptr += 1

            # Add `single` samples
            for _ in range(self.num_single_per_batch):
                if single_ptr >= len(single_perm):
                    # If `single` is exhausted, reshuffle and start over
                    if self.generator is None:
                        single_perm = torch.randperm(len(self.single_indices)).tolist()
                    else:
                        single_perm = torch.randperm(len(self.single_indices), generator=self.generator).tolist()
                    single_ptr = 0

                if len(self.single_indices) > 0:
                    single_idx = single_perm[single_ptr]
                    batch_indices.append(self.single_indices[single_idx])
                    single_ptr += 1

            # Yield all indices in this batch
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        return self._num_samples


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    # use sampler for better ckpt resume
    # Detect dual-group training mode by checking for the index_mapping attribute
    is_dual_group_training = hasattr(train_dataset, 'index_mapping')

    # Check for pure dual-group pairing (all samples are A/B pairs, no `single`)
    has_single_samples = False
    if is_dual_group_training:
        has_single_samples = any(g == "single" for _, g in train_dataset.index_mapping)
    
    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        
        if is_dual_group_training and not has_single_samples:
            # Pure dual-group training (no `single` samples): use the paired sampler
            # so Group A and Group B land in the same batch
            print("[DataLoader] Using PairedRandomSampler for pure dual-group training")
            sampler = PairedRandomSampler(
                data_source=train_dataset,
                generator=train_dataloader_generator,
                pair_size=2  # Each pair contains a Group A and a Group B sample
            )
        elif is_dual_group_training and has_single_samples:
            # Mixed data: use the mixed sampler to keep A/B pairs together
            # while mixing in `single` samples by ratio.
            # Read paired_ratio from config; defaults to 0.7 (70% paired, 30% single)
            paired_ratio = getattr(config, 'paired_ratio', 0.7)
            print(f"[DataLoader] Mixed dataset (dual + single), using MixedPairedRandomSampler")
            print(f"[DataLoader] paired_ratio={paired_ratio} ({paired_ratio*100:.0f}% paired A/B, "
                  f"{(1-paired_ratio)*100:.0f}% single)")
            sampler = MixedPairedRandomSampler(
                data_source=train_dataset,
                index_mapping=train_dataset.index_mapping,
                generator=train_dataloader_generator,
                paired_ratio=paired_ratio,
                batch_size=train_batch_size
            )
        else:
            # Not dual-group training: use a plain random sampler
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    # Dual-group training imposes a constraint on batch_size
    if is_dual_group_training:
        if not has_single_samples:
            # Pure dual-group training: batch_size must be even
            assert train_batch_size % 2 == 0, \
                f"[Dual Group Training] batch_size ({train_batch_size}) must be even to keep paired samples together"
        print(f"[DataLoader] Dual-group training enabled, batch_size={train_batch_size}")

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        image_dir=config.image_dir,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
