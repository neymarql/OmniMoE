"""Dataset definitions for Omni-Stack MoE multimodal training."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
import torchvision.transforms as T
from transformers import AutoTokenizer


@dataclass
class ConversationSample:
    """Single sample of text tokens and optional image tensor."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: Optional[torch.Tensor]
    image_mask: torch.Tensor
    pad_token_id: int


class ConversationDataset(Dataset):
    """Dataset reading a JSON manifest with multimodal conversations."""

    def __init__(
        self,
        json_path: str,
        image_folder: str,
        tokenizer: AutoTokenizer,
        max_text_length: int,
        image_size: int = 384,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        cache_prompts: bool = True,
    ) -> None:
        if not os.path.isfile(json_path):
            print(f"[Dataset][ERROR] Manifest not found: {json_path}")
            raise FileNotFoundError(f"Dataset manifest not found: {json_path}")
        self.json_path = json_path
        self.image_folder = image_folder.rstrip("/")
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.cache_prompts = cache_prompts
        self._prompt_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.image_size = image_size

        mean = mean or [0.5, 0.5, 0.5]
        std = std or [0.5, 0.5, 0.5]
        self.image_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=mean, std=std),
            ]
        )

        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            print(f"[Dataset][ERROR] Expected list in {json_path}, got {type(data)}")
            raise ValueError(f"Expected list in {json_path}, got {type(data)}")
        self.records = data
        print(f"[ConversationDataset] Loaded {len(self.records):,} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_turn(self, content: str) -> List[int]:
        encoded = self.tokenizer(
            content,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        return encoded["input_ids"]

    def _build_text_pair(self, conversations: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_tokens: List[int] = []
        label_tokens: List[int] = []
        for turn in conversations:
            speaker = turn.get("from")
            content = turn.get("value", "")
            turn_text = content if content.endswith("\n") else f"{content}\n"
            tokens = self._tokenize_turn(turn_text)
            if speaker == "human":
                input_tokens.extend(tokens)
                label_tokens.extend([-100] * len(tokens))
            elif speaker == "gpt":
                input_tokens.extend(tokens)
                label_tokens.extend(tokens)
            else:
                raise ValueError(f"Unexpected speaker role: {speaker}")
        if len(input_tokens) > self.max_text_length:
            input_tokens = input_tokens[: self.max_text_length]
            label_tokens = label_tokens[: self.max_text_length]
        attention_mask = [1] * len(input_tokens)
        if len(attention_mask) == 0:
            input_tokens = [self.tokenizer.eos_token_id]
            label_tokens = [-100]
            attention_mask = [1]
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        labels = torch.tensor(label_tokens, dtype=torch.long)
        attention = torch.tensor(attention_mask, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }

    def _load_image(self, sample: Dict[str, Any]) -> torch.Tensor:
        image_list: List[str] = sample.get("image", []) or []
        blank = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        if not image_list:
            return blank
        image_rel_path = image_list[0]
        full_path = os.path.join(self.image_folder, image_rel_path)
        if not os.path.isfile(full_path):
            print(f"[Dataset][WARN] Missing image {full_path}, substituting blank tensor")
            return blank
        with Image.open(full_path) as img:
            image = img.convert("RGB")
        return self.image_transform(image)

    def count_missing_images(self) -> int:
        """Scan the manifest and count missing image files on disk.

        This is O(N) over the manifest and purely informational; call it when
        building the dataloader to report dataset hygiene.
        """
        missing = 0
        for rec in self.records:
            image_list: List[str] = rec.get("image", []) or []
            if not image_list:
                continue
            image_rel_path = image_list[0]
            full_path = os.path.join(self.image_folder, image_rel_path)
            if not os.path.isfile(full_path):
                missing += 1
        return missing

    def __getitem__(self, index: int) -> ConversationSample:
        record = self.records[index]
        conversations = record.get("conversations", [])
        cache_key = record.get("id", str(index))
        if self.cache_prompts and cache_key in self._prompt_cache:
            vectors = self._prompt_cache[cache_key]
        else:
            vectors = self._build_text_pair(conversations)
            if self.cache_prompts:
                self._prompt_cache[cache_key] = vectors
        pixel_values = self._load_image(record)
        has_image = bool(record.get("image")) and torch.count_nonzero(pixel_values).item() > 0
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        return ConversationSample(
            input_ids=vectors["input_ids"],
            attention_mask=vectors["attention_mask"],
            labels=vectors["labels"],
            pixel_values=pixel_values,
            image_mask=torch.tensor(1 if has_image else 0, dtype=torch.long),
            pad_token_id=int(pad_token_id),
        )


class WeightedMultiSourceSampler(Sampler[int]):
    """Sampler that draws from grouped datasets according to sample_rate.

    Works with a top-level ConcatDataset of dataset groups. Each group is a
    ConcatDataset of one or more ConversationDataset built from the same
    data source. The sampler precomputes an epoch plan so we can log exact
    per-group sampling counts.
    """

    def __init__(self, group_lengths: List[int], sample_rates: List[float], seed: int = 42, epoch_length: int | None = None) -> None:
        if len(group_lengths) != len(sample_rates) or len(group_lengths) == 0:
            raise ValueError("group_lengths and sample_rates must be same non-zero length")
        self.group_lengths = group_lengths
        self.sample_rates = sample_rates
        self.seed = int(seed)
        self.epoch_length = int(epoch_length) if epoch_length is not None else int(sum(group_lengths))
        # Compute normalized weights and planned counts
        total_rate = float(sum(sample_rates))
        if total_rate <= 0:
            raise ValueError("Sum of sample_rates must be positive")
        weights = [r / total_rate for r in sample_rates]
        counts = [int(round(w * self.epoch_length)) for w in weights]
        delta = self.epoch_length - sum(counts)
        if delta != 0:
            # Adjust last bucket to match exact epoch_length
            counts[-1] += delta
        self.planned_counts = counts
        # Precompute group base offsets for top-level ConcatDataset
        base = 0
        self.group_offsets: List[int] = []
        for L in group_lengths:
            self.group_offsets.append(base)
            base += L
        # Build index plan deterministically
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices: List[int] = []
        for gid, (L, c) in enumerate(zip(group_lengths, counts)):
            if L <= 0 or c <= 0:
                continue
            # Sample with replacement to allow over/under-sampling
            local = torch.randint(low=0, high=L, size=(c,), generator=g)
            global_idx = local + self.group_offsets[gid]
            indices.extend(global_idx.tolist())
        # Shuffle final plan
        perm = torch.randperm(len(indices), generator=g).tolist()
        self.plan: List[int] = [indices[i] for i in perm]

    def __len__(self) -> int:
        return len(self.plan)

    def __iter__(self):
        return iter(self.plan)


def load_datasets_for_stage(stage_config: Dict[str, Any], tokenizer: AutoTokenizer, max_text_length: int,
                            seed: int = 42, scan_missing: bool = True):
    """Build a grouped ConcatDataset and a WeightedMultiSourceSampler from config.

    Returns (concat_dataset, sampler, info_dict) where info_dict contains
    lengths, planned_counts and missing image statistics per group.
    """
    group_datasets: List[ConcatDataset] = []
    group_lengths: List[int] = []
    sample_rates: List[float] = []
    group_info: List[Dict[str, Any]] = []

    for si, source in enumerate(stage_config.get("data_sources", [])):
        data_paths: Iterable[str] = source.get("data_path", [])
        image_folder: str = source.get("image_folder", "")
        rate: float = float(source.get("sample_rate", 1.0))
        cds: List[ConversationDataset] = []
        missing = 0
        total = 0
        for manifest in data_paths:
            ds = ConversationDataset(
                json_path=manifest,
                image_folder=image_folder,
                tokenizer=tokenizer,
                max_text_length=max_text_length,
            )
            cds.append(ds)
            total += len(ds)
            if scan_missing:
                missing += ds.count_missing_images()
        if not cds:
            continue
        group_ds = ConcatDataset(cds)
        group_datasets.append(group_ds)
        group_lengths.append(len(group_ds))
        sample_rates.append(rate)
        group_info.append({
            "group_index": si,
            "manifests": list(data_paths),
            "image_folder": image_folder,
            "length": len(group_ds),
            "missing_images": missing,
            "sample_rate": rate,
        })

    if not group_datasets:
        raise ValueError("No datasets constructed for stage")

    top_level = ConcatDataset(group_datasets)
    sampler = WeightedMultiSourceSampler(group_lengths=group_lengths, sample_rates=sample_rates, seed=seed)

    # Log sampling plan summary
    planned = sampler.planned_counts
    total_planned = sum(planned)
    summary = []
    for i, info in enumerate(group_info):
        pct = (planned[i] / max(1, total_planned)) * 100.0
        summary.append({
            **info,
            "planned_draws": planned[i],
            "planned_percent": pct,
        })
    stage_info = {
        "groups": summary,
        "total_planned": total_planned,
        "epoch_length": len(sampler),
    }
    print("[Dataset] Stage composition summary:")
    for item in summary:
        print("  - group={group_index} len={length} rate={sample_rate} planned={planned_draws} ({planned_percent:.2f}%) missing_images={missing_images} folder={image_folder}".format(**item))
    return top_level, sampler, stage_info
