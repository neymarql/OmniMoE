"""Optimizer registry and parameter grouping for Omni-Stack MoE.

Provides separation of parameter groups to allow different learning rates
for MoE experts, projector modules, and dense backbone parameters.
Optionally supports advanced optimizers if installed.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch


def _split_param_groups(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    groups: Dict[str, List[torch.nn.Parameter]] = {"projector": [], "moe": [], "dense": []}
    for name, module in model.named_modules():
        if module.__class__.__name__ == "ProjectorBridge" or name.startswith("projector"):
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    groups["projector"].append(p)
        if module.__class__.__name__ == "MoEFeedForward":
            for p in module.parameters(recurse=False):
                if p.requires_grad:
                    groups["moe"].append(p)
    # Dense: anything still requires_grad and not already claimed
    claimed = set([id(p) for ps in groups.values() for p in ps])
    for p in model.parameters():
        if p.requires_grad and id(p) not in claimed:
            groups["dense"].append(p)
    return groups


def build_optimizer(
    model: torch.nn.Module,
    base_lr: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    moe_lr_mult: float = 1.0,
    projector_lr_mult: float = 1.0,
    optimizer_name: str = "adamw",
) -> torch.optim.Optimizer:
    groups = _split_param_groups(model)
    param_groups: List[Dict[str, Any]] = []
    if groups["dense"]:
        param_groups.append({"params": groups["dense"], "lr": base_lr})
    if groups["moe"]:
        param_groups.append({"params": groups["moe"], "lr": base_lr * moe_lr_mult})
    if groups["projector"]:
        param_groups.append({"params": groups["projector"], "lr": base_lr * projector_lr_mult})

    name = optimizer_name.lower()
    if name == "adamw" or name == "adamw_torch":
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "muon":
        import muon  # type: ignore
        return muon.Muon(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # Default fallback
    return torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, eps=eps, weight_decay=weight_decay)


__all__ = ["build_optimizer"]
