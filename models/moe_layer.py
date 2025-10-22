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
        router_top_k: int = 2,
        noisy_gate_policy: str = "Jitter",
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        drop_tokens: bool = False,
        use_tutel: bool = True,
        scope: str | None = None,
        use_shared_expert: bool = True,
        shared_expert_scale: float = 0.1,
        use_megablocks_dropless: bool = False,
    ) -> None:
        super().__init__()
        self._backend = "deepspeed"
        expert = ExpertFFN(hidden_size, ffn_dim)
        self.moe = None
        if use_megablocks_dropless:
            try:
                # Attempt to import MegaBlocks dropless MoE. The concrete API may
                # differ across versions; adapt as needed when installed.
                from megablocks.torch.moe import DroplessMoE as MegaBlocksMoE  # type: ignore

                self._backend = "megablocks"
                self.moe = MegaBlocksMoE(
                    hidden_size=hidden_size,
                    expert=expert,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    capacity_factor=capacity_factor,
                    min_capacity=min_capacity,
                    noisy_gate_policy=noisy_gate_policy,
                )
            except Exception as e:
                print("[MoE][ERROR] MegaBlocks dropless routing requested but unavailable/incompatible:", e)
                raise RuntimeError(
                    "Requested MegaBlocks dropless routing but megablocks is not available or incompatible."
                ) from e
        if self.moe is None:
            if DeepSpeedMoE is None:
                print("[MoE][ERROR] DeepSpeed MoE extension not found. Please install deepspeed with --enable-moe.")
                raise RuntimeError(
                    "DeepSpeed MoE extension not found. Install deepspeed>=0.12 with --enable-moe."
                )
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
        # Aux bookkeeping
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_expert_counts: Optional[torch.Tensor] = None
        self._ema_expert_counts: Optional[torch.Tensor] = None
        self._ema_decay: float = 0.9
        self._moe_scope: str | None = scope
        self.use_shared_expert = use_shared_expert
        self.shared_expert_scale = shared_expert_scale
        self.shared_expert = ExpertFFN(hidden_size, ffn_dim) if use_shared_expert else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            flat_states = hidden_states.reshape(-1, original_shape[-1])
            outputs = self.moe(flat_states)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                output, l_aux = outputs[0], outputs[1]
                self._last_aux_loss = l_aux
                if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                    self._last_expert_counts = outputs[2].detach()
                    with torch.no_grad():
                        if self._ema_expert_counts is None:
                            self._ema_expert_counts = self._last_expert_counts.clone().float()
                        else:
                            self._ema_expert_counts.mul_(self._ema_decay).add_(
                                self._last_expert_counts.float(), alpha=(1.0 - self._ema_decay)
                            )
                # Drop ratio (if provided by backend)
                if isinstance(outputs, tuple) and len(outputs) >= 4 and isinstance(outputs[3], torch.Tensor):
                    try:
                        drops = outputs[3].detach().float()
                        total = drops.sum().item()
                        self._last_drop_ratio = float(total)
                    except Exception:
                        self._last_drop_ratio = None
            else:
                output = outputs
            if self.shared_expert is not None:
                shared_out = self.shared_expert(flat_states)
                output = output + self.shared_expert_scale * shared_out
            return output.reshape(original_shape)
        outputs = self.moe(hidden_states)
        if isinstance(outputs, tuple) and len(outputs) >= 2:
            out, l_aux = outputs[0], outputs[1]
            self._last_aux_loss = l_aux
            if len(outputs) >= 3 and isinstance(outputs[2], torch.Tensor):
                self._last_expert_counts = outputs[2].detach()
                with torch.no_grad():
                    if self._ema_expert_counts is None:
                        self._ema_expert_counts = self._last_expert_counts.clone().float()
                    else:
                        self._ema_expert_counts.mul_(self._ema_decay).add_(
                            self._last_expert_counts.float(), alpha=(1.0 - self._ema_decay)
                        )
            if isinstance(outputs, tuple) and len(outputs) >= 4 and isinstance(outputs[3], torch.Tensor):
                try:
                    drops = outputs[3].detach().float()
                    total = drops.sum().item()
                    self._last_drop_ratio = float(total)
                except Exception:
                    self._last_drop_ratio = None
            if self.shared_expert is not None:
                shared_out = self.shared_expert(hidden_states)
                out = out + self.shared_expert_scale * shared_out
            return out
        output = outputs
        if self.shared_expert is not None:
            output = output + self.shared_expert_scale * self.shared_expert(hidden_states)
        return output


__all__ = ["MoEFeedForward", "ExpertFFN", "split_params_into_different_moe_groups_for_optimizer"]
