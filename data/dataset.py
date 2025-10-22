"""Dataset definitions for Omni-Stack MoE multimodal training."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
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


class MultiSourceDataset(Dataset):
    """Mixture dataset drawing from multiple ConversationDataset instances."""

    def __init__(self, datasets: List[ConversationDataset], sample_rates: List[float]) -> None:
        if len(datasets) != len(sample_rates):
            raise ValueError("datasets and sample_rates must be same length")
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        self.datasets = datasets
        total = float(sum(sample_rates))
        if total <= 0:
            raise ValueError("Sum of sample_rates must be positive")
        probs = [rate / total for rate in sample_rates]
        cumulative: List[float] = []
        running = 0.0
        for prob in probs:
            running += prob
            cumulative.append(running)
        cumulative[-1] = 1.0
        self.cumulative = cumulative
        self.length = max(int(sum(len(ds) * prob for ds, prob in zip(datasets, probs))), 1)

    def __len__(self) -> int:
        return self.length

    def _choose_dataset(self) -> ConversationDataset:
        r = random.random()
        for idx, boundary in enumerate(self.cumulative):
            if r <= boundary:
                return self.datasets[idx]
        return self.datasets[-1]

    def __getitem__(self, index: int) -> ConversationSample:
        dataset = self._choose_dataset()
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]


def load_datasets_for_stage(stage_config: Dict[str, Any], tokenizer: AutoTokenizer, max_text_length: int) -> MultiSourceDataset:
    """Utility to build a MultiSourceDataset from the stage configuration."""

    datasets: List[ConversationDataset] = []
    sample_rates: List[float] = []
    for source in stage_config.get("data_sources", []):
        data_paths: Iterable[str] = source.get("data_path", [])
        image_folder: str = source.get("image_folder", "")
        sample_rate: float = float(source.get("sample_rate", 1.0))
        for manifest in data_paths:
            dataset = ConversationDataset(
                json_path=manifest,
                image_folder=image_folder,
                tokenizer=tokenizer,
                max_text_length=max_text_length,
            )
            datasets.append(dataset)
            sample_rates.append(sample_rate)
    return MultiSourceDataset(datasets=datasets, sample_rates=sample_rates)
