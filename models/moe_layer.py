"""Reusable MoE building blocks for Omni-Stack models."""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

try:
    from deepspeed.moe.layer import MoE as DeepSpeedMoE
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
except ModuleNotFoundError:  # pragma: no cover - fallback path
    DeepSpeedMoE = None  # type: ignore
    split_params_into_different_moe_groups_for_optimizer = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class ExpertFFN(nn.Module):
    """Feed-forward expert consisting of two linear layers and activation."""

    def __init__(self, hidden_size: int, ffn_dim: int, activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        activation = activation or nn.SiLU()
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.act = activation
        self.fc2 = nn.Linear(ffn_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.fc2(self.act(self.fc1(x)))


class MoEFeedForward(nn.Module):
    """Wrapper that converts a dense FFN into a sparse MoE equivalent."""

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        num_experts: int,
        ep_size: int,
        router_top_k: int = 1,
        noisy_gate_policy: str = "Jitter",
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        drop_tokens: bool = False,
        use_tutel: bool = True,
    ) -> None:
        super().__init__()
        if DeepSpeedMoE is None:
            raise RuntimeError(
                "DeepSpeed MoE extension not found. Install deepspeed>=0.12 with --enable-moe."  # noqa: E501
            )
        expert = ExpertFFN(hidden_size, ffn_dim)
        self.moe = DeepSpeedMoE(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=num_experts,
            ep_size=ep_size,
            k=router_top_k,
            noisy_gate_policy=noisy_gate_policy,
            capacity_factor=capacity_factor,
            min_capacity=min_capacity,
            drop_tokens=drop_tokens,
            use_tutel=use_tutel,
            enable_expert_tensor_parallelism=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            flat_states = hidden_states.reshape(-1, original_shape[-1])
            output = self.moe(flat_states)
            return output.reshape(original_shape)
        return self.moe(hidden_states)


__all__ = ["MoEFeedForward", "ExpertFFN", "split_params_into_different_moe_groups_for_optimizer"]
