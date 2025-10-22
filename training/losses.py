"""Loss utilities for Omni-Stack MoE."""
from __future__ import annotations

import torch
import torch.nn as nn


class LanguageModelingLoss(nn.Module):
    """Wrapper around cross entropy ignoring masked positions."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # noqa: D401
        vocab_size = logits.size(-1)
        return self.criterion(logits.view(-1, vocab_size), labels.view(-1))


def compute_alignment_loss(projector_aux: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0:
        return torch.tensor(0.0, device=projector_aux.device)
    return projector_aux * weight


__all__ = ["LanguageModelingLoss", "compute_alignment_loss"]
