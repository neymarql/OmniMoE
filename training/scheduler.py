"""Learning-rate scheduling utilities supporting multi-stage curricula."""
from __future__ import annotations

from typing import Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer: Optimizer, hyperparams: Dict[str, float], total_steps: int) -> LambdaLR:
    warmup = int(hyperparams.get("warmup_steps", 0))
    min_lr_scale = hyperparams.get("min_lr_scale", 0.1)

    def lr_lambda(step: int) -> float:
        if step < warmup and warmup > 0:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_scheduler"]
