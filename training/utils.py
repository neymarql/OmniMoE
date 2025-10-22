"""Utility helpers for Omni-Stack training scripts."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
import time


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
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


_ALL_TO_ALL_MONITOR = {
    "enabled": False,
    "orig_single": None,
    "orig_all": None,
    "total_ms": 0.0,
    "call_count": 0,
}


def _wrap_all_to_all(fn):
    def wrapped(*args, **kwargs):
        start_event = end_event = None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            _ALL_TO_ALL_MONITOR["total_ms"] += float(start_event.elapsed_time(end_event))
        else:
            _ALL_TO_ALL_MONITOR["total_ms"] += (time.time() - start_time) * 1000.0
        _ALL_TO_ALL_MONITOR["call_count"] += 1
        return result

    return wrapped


def enable_all_to_all_monitor() -> None:
    if _ALL_TO_ALL_MONITOR["enabled"]:
        return
    if not dist.is_available():
        return
    try:
        _ALL_TO_ALL_MONITOR["orig_single"] = dist.all_to_all_single
        _ALL_TO_ALL_MONITOR["orig_all"] = dist.all_to_all
        dist.all_to_all_single = _wrap_all_to_all(dist.all_to_all_single)
        dist.all_to_all = _wrap_all_to_all(dist.all_to_all)
        _ALL_TO_ALL_MONITOR["enabled"] = True
    except Exception as err:
        print(f"[Utils][WARN] Failed to enable all-to-all monitor: {err}")


def get_all_to_all_stats(reset: bool = True) -> Tuple[float, int]:
    total_ms = _ALL_TO_ALL_MONITOR["total_ms"]
    calls = _ALL_TO_ALL_MONITOR["call_count"]
    if reset:
        _ALL_TO_ALL_MONITOR["total_ms"] = 0.0
        _ALL_TO_ALL_MONITOR["call_count"] = 0
    return total_ms, calls


__all__ = [
    "load_json",
    "load_yaml",
    "set_seed",
    "ensure_dir",
    "ensure_tokenizer_padding",
    "maybe_load_checkpoint",
    "save_hf_checkpoint",
    "enable_all_to_all_monitor",
    "get_all_to_all_stats",
]
