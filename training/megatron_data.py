"""Megatron-Core compatible dataset adapter for OmniMoE JSON manifests (text-only).

Converts the multimodal JSON (with conversations) into text sequences for LLM
pretraining/fine-tuning. Removes <image> tags and concatenates alternating
human/gpt turns, masking labels for human turns by default.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class JsonTextOnlyDataset(Dataset):
    def __init__(self, json_path: str, tokenizer_name: str, max_length: int = 4096) -> None:
        with open(json_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            print(f"[MegatronData][ERROR] Expected list in {json_path}, got {type(data)}")
            raise ValueError("Invalid JSON format for dataset")
        self.records: List[Dict[str, Any]] = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.max_length = max_length
        print(f"[MegatronData] Loaded {len(self.records):,} samples from {json_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        conv = rec.get("conversations", [])
        text_tokens: List[int] = []
        label_tokens: List[int] = []
        for turn in conv:
            role = turn.get("from")
            val = (turn.get("value", "").replace("<image>", "")).strip()
            if not val.endswith("\n"):
                val = val + "\n"
            ids = self.tokenizer(val, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            if role == "human":
                text_tokens.extend(ids)
                label_tokens.extend([-100] * len(ids))
            elif role == "gpt":
                text_tokens.extend(ids)
                label_tokens.extend(ids)
        text_tokens = text_tokens[: self.max_length]
        label_tokens = label_tokens[: self.max_length]
        attn = [1] * len(text_tokens)
        if not text_tokens:
            text_tokens = [self.tokenizer.eos_token_id]
            label_tokens = [-100]
            attn = [1]
        return {
            "input_ids": torch.tensor(text_tokens, dtype=torch.long),
            "labels": torch.tensor(label_tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def collate_text_only(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids = []
    labels = []
    attn = []
    for x in batch:
        pad = max_len - x["input_ids"].size(0)
        if pad > 0:
            input_ids.append(torch.cat([x["input_ids"], torch.zeros(pad, dtype=torch.long)], dim=0))
            labels.append(torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)], dim=0))
            attn.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)], dim=0))
        else:
            input_ids.append(x["input_ids"]) ; labels.append(x["labels"]) ; attn.append(x["attention_mask"])
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attn, dim=0),
    }

