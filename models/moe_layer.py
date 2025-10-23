"""Reusable MoE building blocks for Omni-Stack models."""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.distributed as dist

from deepspeed.moe.layer import MoE as DeepSpeedMoE
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from .ep_dispatch import EpAllToAllDispatcher

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

    EP_GROUP_CACHE: Dict[Tuple[int, ...], dist.ProcessGroup] = {}

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
        use_expert_choice_router: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = max(1, ep_size)
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.router_top_k = router_top_k
        self.noisy_gate_policy = noisy_gate_policy
        self._router_noise_std = 0.01 if noisy_gate_policy.lower() == "jitter" else 0.0
        self._router_temperature: float = 1.0
        self._moe_scope = scope
        self.use_shared_expert = use_shared_expert
        self.shared_expert_scale = shared_expert_scale
        self.shared_expert = ExpertFFN(hidden_size, ffn_dim) if use_shared_expert else None
        self.use_expert_choice = use_expert_choice_router
        self.use_megablocks = use_megablocks_dropless
        # Backend selection establishes how EP is handled:
        #  - "deepspeed": EP groups are managed internally by DeepSpeedMoE
        #  - "expert_choice" / "megablocks": Python-side EP group is created
        self._backend = "expert_choice" if use_expert_choice_router else "deepspeed"
        self.num_local_experts = 0
        self._global_expert_offset = 0
        self._ep_group: Optional[dist.ProcessGroup] = None

        self.router: Optional[nn.Linear] = None
        self.local_experts: Optional[nn.ModuleList] = None
        expert = ExpertFFN(hidden_size, ffn_dim)
        self.moe = None
        if self.use_expert_choice:
            self._init_expert_choice(ffn_dim)
        else:
            self._init_standard_moe(
                expert=expert,
                use_tutel=use_tutel,
                drop_tokens=drop_tokens,
                use_megablocks_dropless=use_megablocks_dropless,
            )

        # EP group policy: only build Python-side EP group for expert_choice/megablocks.
        if self._backend in {"expert_choice", "megablocks"}:
            self._ensure_ep_group()
            if self.use_expert_choice and self._ep_group is not None:
                self._ec_dispatcher = EpAllToAllDispatcher(self._ep_group, self.ep_size, self.num_local_experts, self._ep_rank)
        else:
            # DeepSpeed backend manages EP internally; keep local counts simple.
            self._ep_group = None
            self.num_local_experts = self.num_experts
            self._global_expert_offset = 0

        # Aux bookkeeping
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_expert_counts: Optional[torch.Tensor] = None
        self._ema_expert_counts: Optional[torch.Tensor] = None
        self._ema_decay: float = 0.9
        self._last_drop_ratio: Optional[float] = None
        self._moe_timing_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._last_moe_forward_ms: float = 0.0

    def _ensure_ep_group(self) -> None:
        if not dist.is_available() or not dist.is_initialized() or self.ep_size <= 1:
            self._ep_group = None
            self._global_expert_offset = 0
            self.num_local_experts = self.num_experts
            return
        world = dist.get_world_size()
        rank = dist.get_rank()
        group_idx = rank // self.ep_size
        start = group_idx * self.ep_size
        ranks = tuple(range(start, min(start + self.ep_size, world)))
        if ranks not in MoEFeedForward.EP_GROUP_CACHE:
            MoEFeedForward.EP_GROUP_CACHE[ranks] = dist.new_group(list(ranks))
        self._ep_group = MoEFeedForward.EP_GROUP_CACHE[ranks]
        self.num_local_experts = self.num_experts // self.ep_size
        self._global_expert_offset = group_idx * self.num_local_experts
        self._ep_rank = rank - start

    def _init_standard_moe(self, expert: nn.Module, use_tutel: bool, drop_tokens: bool, use_megablocks_dropless: bool) -> None:
        if use_megablocks_dropless:
            from megablocks.torch.moe import DroplessMoE as MegaBlocksMoE  # type: ignore

            self._backend = "megablocks"
            self.moe = MegaBlocksMoE(
                hidden_size=self.hidden_size,
                expert=expert,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                capacity_factor=self.capacity_factor,
                min_capacity=self.min_capacity,
                noisy_gate_policy=self.noisy_gate_policy,
            )
        if self.moe is None:
            self.moe = DeepSpeedMoE(
                hidden_size=self.hidden_size,
                expert=expert,
                num_experts=self.num_experts,
                ep_size=self.ep_size,
                k=self.router_top_k,
                noisy_gate_policy=self.noisy_gate_policy,
                capacity_factor=self.capacity_factor,
                min_capacity=self.min_capacity,
                drop_tokens=drop_tokens,
                use_tutel=use_tutel,
                enable_expert_tensor_parallelism=False,
            )
        self._backend = "deepspeed" if not use_megablocks_dropless else "megablocks"

    def _init_expert_choice(self, ffn_dim: int) -> None:
        if self.num_experts % self.ep_size != 0:
            raise ValueError("num_experts must be divisible by ep_size for expert-choice routing")
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_experts = nn.ModuleList(ExpertFFN(self.hidden_size, ffn_dim) for _ in range(self.num_local_experts))
        self._backend = "expert_choice"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_expert_choice:
            return self._forward_expert_choice(hidden_states)
        return self._forward_standard(hidden_states)

    def _forward_standard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        timing_enabled = torch.cuda.is_available() and hidden_states.is_cuda
        if timing_enabled and self._moe_timing_events is None:
            self._moe_timing_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        self._last_drop_ratio = None
        if hidden_states.dim() == 3:
            flat_states = hidden_states.reshape(-1, original_shape[-1])
            if timing_enabled and self._moe_timing_events is not None:
                start_evt, end_evt = self._moe_timing_events
                start_evt.record()
            outputs = self.moe(flat_states)
            if timing_enabled and self._moe_timing_events is not None:
                end_evt.record()
                torch.cuda.synchronize()
                self._last_moe_forward_ms = float(start_evt.elapsed_time(end_evt))
            else:
                self._last_moe_forward_ms = 0.0
            output = self._process_backend_outputs(outputs, flat_states)
            return output.reshape(original_shape)
        if timing_enabled and self._moe_timing_events is not None:
            start_evt, end_evt = self._moe_timing_events
            start_evt.record()
        outputs = self.moe(hidden_states)
        if timing_enabled and self._moe_timing_events is not None:
            end_evt.record()
            torch.cuda.synchronize()
            self._last_moe_forward_ms = float(start_evt.elapsed_time(end_evt))
        else:
            self._last_moe_forward_ms = 0.0
        return self._process_backend_outputs(outputs, hidden_states)

    def _process_backend_outputs(self, outputs, input_tensor):
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
            if len(outputs) >= 4 and isinstance(outputs[3], torch.Tensor):
                self._last_drop_ratio = float(outputs[3].detach().float().sum().item())
            if self.shared_expert is not None:
                shared_out = self.shared_expert(input_tensor).to(out.dtype)
                out = out + self.shared_expert_scale * shared_out
            return out
        output = outputs
        if self.shared_expert is not None:
            shared_out = self.shared_expert(input_tensor).to(output.dtype)
            output = output + self.shared_expert_scale * shared_out
        return output

    def _forward_expert_choice(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            tokens = hidden_states.reshape(-1, original_shape[-1])
        else:
            tokens = hidden_states
        dtype = tokens.dtype
        scores = self.router(tokens.float())
        if self._router_noise_std > 0 and self.training:
            scores = scores + self._router_noise_std * torch.randn_like(scores)
        temp = max(self._router_temperature, 1e-6)
        N, E = scores.shape
        capacity = max(self.min_capacity, math.ceil(self.capacity_factor * N / E))
        capacity = min(capacity, N)
        counts_local = torch.zeros(self.num_experts, device=tokens.device)
        importance_local = torch.zeros(self.num_experts, device=tokens.device)
        if getattr(self, "_ec_dispatcher", None) is not None:
            # EP token-level routing for EC
            agg, counts, importance = self._ec_dispatcher.route_expert_choice(
                tokens.to(dtype), scores, capacity, self.num_local_experts, temperature=temp,
                local_expert_call=lambda lid, x: self.local_experts[lid](x)
            )
            output = agg.to(dtype)
            counts_local = counts
            importance_local = importance
        else:
            top_scores, top_indices = torch.topk((scores / temp).transpose(0, 1), k=capacity, dim=-1)
            selected = tokens.index_select(0, top_indices.reshape(-1)).view(E, capacity, -1)
            weights = torch.softmax(top_scores, dim=-1).unsqueeze(-1).to(dtype)
            output = torch.zeros_like(tokens)
            for local_idx in range(self.num_local_experts):
                global_idx = self._global_expert_offset + local_idx
                expert_tokens = selected[global_idx].to(dtype)
                gate = weights[global_idx]
                expert_out = self.local_experts[local_idx](expert_tokens)
                weighted = expert_out * gate
                idx = top_indices[global_idx]
                output.index_add_(0, idx, weighted)
                counts_local[global_idx] += float(capacity)
                importance_local[global_idx] += float(top_scores[global_idx].sum().item())
        total_counts = counts_local.sum().clamp(min=1.0)
        counts_prob = counts_local / total_counts
        total_importance = importance_local.sum().clamp(min=1.0)
        importance_prob = importance_local / total_importance
        target = 1.0 / self.num_experts
        aux_loss = torch.sum((counts_prob - target) ** 2 + (importance_prob - target) ** 2)
        self._last_aux_loss = aux_loss
        self._last_expert_counts = counts_local.detach()
        self._ema_expert_counts = counts_local.detach().float()
        self._last_drop_ratio = torch.tensor(0.0, device=tokens.device)
        output = output.to(dtype)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(tokens.to(dtype))
            output = output + self.shared_expert_scale * shared_out
        self._last_moe_forward_ms = 0.0
        if hidden_states.dim() == 3:
            return output.reshape(original_shape)
        return output

    # Runtime router controls (apply to expert-choice/megablocks backends)
    def set_router_temperature(self, temperature: float) -> None:
        self._router_temperature = float(temperature)

    def set_router_jitter(self, std: float) -> None:
        self._router_noise_std = float(std)


__all__ = ["MoEFeedForward", "ExpertFFN", "split_params_into_different_moe_groups_for_optimizer"]
