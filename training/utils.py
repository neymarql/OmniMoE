"""Utility helpers for Omni-Stack training scripts."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_tokenizer_padding(tokenizer) -> None:  # noqa: D401
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def maybe_load_checkpoint(model_engine, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    if os.path.isdir(checkpoint_path):
        model_engine.load_checkpoint(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location="cpu")
        model_engine.module.load_state_dict(state, strict=False)


def save_hf_checkpoint(model_engine, tokenizer, output_dir: str) -> None:
    """Save a HuggingFace-compatible checkpoint from a DeepSpeed engine.

    This exports the model weights and config via PreTrainedModel.save_pretrained
    and writes tokenizer files so serving frameworks (vLLM/OpenCompass) can load.
    """
    ensure_dir(output_dir)
    # Only rank 0 writes to disk
    try:
        import deepspeed
        is_rank0 = (deepspeed.comm.get_rank() == 0)
    except Exception:
        is_rank0 = True
    if not is_rank0:
        return
    model = model_engine.module
    # Some PreTrainedModel subclasses need to be on CPU to save comfortably
    model_to_save = model
    model_to_save.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


__all__ += ["save_hf_checkpoint"]


__all__ = [
    "load_json",
    "load_yaml",
    "set_seed",
    "ensure_dir",
    "ensure_tokenizer_padding",
    "maybe_load_checkpoint",
]
