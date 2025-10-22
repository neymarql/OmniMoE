"""Collation utilities for Omni-Stack MoE batches."""
from __future__ import annotations

from typing import Dict, List

import torch

from .dataset import ConversationSample


def multimodal_collate_fn(batch: List[ConversationSample]) -> Dict[str, torch.Tensor]:
    """Pad sequences and stack image tensors for a minibatch."""

    if not batch:
        raise ValueError("Batch is empty")

    pad_token_id = batch[0].pad_token_id
    max_seq_len = max(sample.input_ids.size(0) for sample in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for sample in batch:
        seq_len = sample.input_ids.size(0)
        pad_len = max_seq_len - seq_len
        if pad_len > 0:
            input_pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            mask_pad = torch.zeros((pad_len,), dtype=torch.long)
            label_pad = torch.full((pad_len,), -100, dtype=torch.long)
            input_ids.append(torch.cat([sample.input_ids, input_pad], dim=0))
            attention_masks.append(torch.cat([sample.attention_mask, mask_pad], dim=0))
            labels.append(torch.cat([sample.labels, label_pad], dim=0))
        else:
            input_ids.append(sample.input_ids)
            attention_masks.append(sample.attention_mask)
            labels.append(sample.labels)

    batch_dict: Dict[str, torch.Tensor] = {
        "input_ids": torch.stack(input_ids, dim=0),
        "attention_mask": torch.stack(attention_masks, dim=0),
        "labels": torch.stack(labels, dim=0),
    }

    image_tensors = []
    image_mask = []
    for sample in batch:
        image_tensors.append(sample.pixel_values)
        image_mask.append(sample.image_mask)
    pixel_values = torch.stack(image_tensors, dim=0)
    batch_dict["pixel_values"] = pixel_values
    batch_dict["image_mask"] = torch.stack(image_mask, dim=0)

    return batch_dict
