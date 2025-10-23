"""Expert-Parallel all-to-all dispatcher reusable across modules.

Implements a two-phase token routing for expert-parallel groups:
 1) counts exchange + three all_to_all_single (tokens, meta, weights)
 2) local expert compute + send-back all_to_all_single to original owners
Finally, aggregates results into the original token order via index_add.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist


class EpAllToAllDispatcher:
    def __init__(self, ep_group: dist.ProcessGroup, ep_size: int, num_local_experts: int, ep_rank: int) -> None:
        if ep_group is None or ep_size <= 1:
            raise ValueError("EpAllToAllDispatcher requires a valid EP group and ep_size>1")
        self.group = ep_group
        self.ep_size = int(ep_size)
        self.num_local_experts = int(num_local_experts)
        self.ep_rank = int(ep_rank)
        # Tiny profiler state
        self.profile_enabled: bool = False
        self.last_ms_tokens: float = 0.0
        self.last_ms_meta: float = 0.0
        self.last_ms_weights: float = 0.0
        self.last_ms_local: float = 0.0

    def enable_profiling(self, enabled: bool = True) -> None:
        self.profile_enabled = bool(enabled)

    def _dispatch_and_combine(
        self,
        expanded_tokens: torch.Tensor,  # [M, H]
        owners_flat: torch.Tensor,      # [M] dest rank in EP group
        local_idx_flat: torch.Tensor,   # [M] local expert id on dest rank
        weights_flat: torch.Tensor,     # [M]
        token_ids_flat: torch.Tensor,   # [M] original token indices in [0, N)
        token_space_len: int,
        local_expert_call,
    ) -> torch.Tensor:
        device = expanded_tokens.device
        dtype = expanded_tokens.dtype
        H = expanded_tokens.size(-1)
        # Order by owners for contiguous splits
        order = torch.argsort(owners_flat)
        owners_sorted = owners_flat[order]
        local_idx_sorted = local_idx_flat[order]
        weights_sorted = weights_flat[order]
        token_ids_sorted = token_ids_flat[order]
        tokens_sorted = expanded_tokens[order]

        # Build split sizes and exchange recv sizes
        send_counts = torch.bincount(owners_sorted, minlength=self.ep_size)
        send_counts_list = send_counts.tolist()
        counts_tensor = send_counts.to(torch.long)
        counts_gather = [torch.zeros_like(counts_tensor) for _ in range(self.ep_size)]
        dist.all_gather(counts_gather, counts_tensor, group=self.group)
        counts_matrix = torch.stack(counts_gather, dim=0)
        recv_counts = counts_matrix[:, self.ep_rank]
        recv_counts_list = recv_counts.tolist()

        # First A2A: tokens
        if self.profile_enabled and torch.cuda.is_available():
            evt_s_tok = torch.cuda.Event(enable_timing=True); evt_e_tok = torch.cuda.Event(enable_timing=True)
            evt_s_tok.record()
        recv_total = int(sum(recv_counts_list))
        recv_tokens = torch.empty((recv_total, H), device=device, dtype=dtype)
        dist.all_to_all_single(recv_tokens, tokens_sorted, recv_counts_list, send_counts_list, group=self.group)
        if self.profile_enabled and torch.cuda.is_available():
            evt_e_tok.record(); torch.cuda.synchronize();
            self.last_ms_tokens = float(evt_s_tok.elapsed_time(evt_e_tok))

        # Second A2A: meta (local_idx, origin, token_ids)
        if self.profile_enabled and torch.cuda.is_available():
            evt_s_meta = torch.cuda.Event(enable_timing=True); evt_e_meta = torch.cuda.Event(enable_timing=True)
            evt_s_meta.record()
        origin_sorted = torch.full_like(local_idx_sorted, fill_value=self.ep_rank)
        send_meta = torch.stack([local_idx_sorted, origin_sorted, token_ids_sorted], dim=1).to(torch.long)
        recv_meta = torch.empty((recv_total, 3), device=device, dtype=torch.long)
        dist.all_to_all_single(recv_meta, send_meta, recv_counts_list, send_counts_list, group=self.group)
        if self.profile_enabled and torch.cuda.is_available():
            evt_e_meta.record(); torch.cuda.synchronize();
            self.last_ms_meta = float(evt_s_meta.elapsed_time(evt_e_meta))

        # Third A2A: weights
        if self.profile_enabled and torch.cuda.is_available():
            evt_s_w = torch.cuda.Event(enable_timing=True); evt_e_w = torch.cuda.Event(enable_timing=True)
            evt_s_w.record()
        send_weights = weights_sorted.unsqueeze(1).to(dtype)
        recv_weights = torch.empty((recv_total, 1), device=device, dtype=dtype)
        dist.all_to_all_single(recv_weights, send_weights, recv_counts_list, send_counts_list, group=self.group)
        if self.profile_enabled and torch.cuda.is_available():
            evt_e_w.record(); torch.cuda.synchronize();
            self.last_ms_weights = float(evt_s_w.elapsed_time(evt_e_w))

        # Local expert compute grouped by local expert id
        local_idx_recv = recv_meta[:, 0]
        origin_recv = recv_meta[:, 1]
        token_ids_recv = recv_meta[:, 2]
        weights_recv = recv_weights[:, 0]

        outputs: List[torch.Tensor] = []
        origins_out: List[torch.Tensor] = []
        token_ids_out: List[torch.Tensor] = []
        if self.profile_enabled and torch.cuda.is_available():
            evt_s_local = torch.cuda.Event(enable_timing=True); evt_e_local = torch.cuda.Event(enable_timing=True)
            evt_s_local.record()
        if recv_tokens.size(0) > 0:
            for local_id in range(self.num_local_experts):
                mask = local_idx_recv == local_id
                if mask.any():
                    tokens_local = recv_tokens[mask]
                    weights_local = weights_recv[mask]
                    out_local = local_expert_call(local_id, tokens_local)
                    out_local = out_local * weights_local.unsqueeze(-1)
                    outputs.append(out_local)
                    origins_out.append(origin_recv[mask])
                    token_ids_out.append(token_ids_recv[mask])
        if self.profile_enabled and torch.cuda.is_available():
            evt_e_local.record(); torch.cuda.synchronize();
            self.last_ms_local = float(evt_s_local.elapsed_time(evt_e_local))
        if outputs:
            outputs_cat = torch.cat(outputs, dim=0)
            origins_cat = torch.cat(origins_out, dim=0)
            token_ids_cat = torch.cat(token_ids_out, dim=0)
        else:
            outputs_cat = torch.empty((0, H), device=device, dtype=dtype)
            origins_cat = torch.empty((0,), device=device, dtype=torch.long)
            token_ids_cat = torch.empty((0,), device=device, dtype=torch.long)

        # Send back to original EP rank owners
        if outputs_cat.size(0) > 0:
            send_back_counts = torch.bincount(origins_cat, minlength=self.ep_size)
        else:
            send_back_counts = torch.zeros(self.ep_size, device=device, dtype=torch.long)
        send_back_counts_list = send_back_counts.tolist()
        counts_back_tensor = send_back_counts.clone()
        gather_back = [torch.zeros_like(counts_back_tensor) for _ in range(self.ep_size)]
        dist.all_gather(gather_back, counts_back_tensor, group=self.group)
        recv_back_counts = torch.stack(gather_back, dim=0)[:, self.ep_rank]
        recv_back_counts_list = recv_back_counts.tolist()
        recv_back_total = int(sum(recv_back_counts_list))

        recv_outputs = torch.empty((recv_back_total, H), device=device, dtype=dtype)
        dist.all_to_all_single(recv_outputs, outputs_cat, recv_back_counts_list, send_back_counts_list, group=self.group)

        recv_token_ids = torch.empty((recv_back_total,), device=device, dtype=torch.long)
        dist.all_to_all_single(recv_token_ids, token_ids_cat.to(torch.long), recv_back_counts_list, send_back_counts_list, group=self.group)

        # Aggregate back to original token order
        agg = torch.zeros((token_space_len, H), device=device, dtype=dtype)
        if recv_back_total > 0:
            agg.index_add_(0, recv_token_ids, recv_outputs)
        return agg

    def route_per_token_topk(
        self,
        tokens: torch.Tensor,      # [B, T, H]
        top_indices: torch.Tensor, # [B, T, K] global expert ids
        top_weights: torch.Tensor, # [B, T, K]
        num_local_experts: int,
        local_expert_call,
    ) -> torch.Tensor:
        B, T, H = tokens.shape
        K = top_indices.size(-1)
        tokens_flat = tokens.reshape(-1, H)
        # Build send plan
        owners = top_indices.reshape(-1, K) // num_local_experts
        local_idx = top_indices.reshape(-1, K) % num_local_experts
        owners_flat = owners.reshape(-1)
        local_idx_flat = local_idx.reshape(-1)
        weights_flat = top_weights.reshape(-1, K).reshape(-1)
        token_ids = torch.arange(B * T, device=tokens.device).unsqueeze(1).expand(-1, K).reshape(-1)
        expanded_tokens = tokens_flat.unsqueeze(1).expand(-1, K, -1).reshape(-1, H)

        agg = self._dispatch_and_combine(
            expanded_tokens=expanded_tokens,
            owners_flat=owners_flat,
            local_idx_flat=local_idx_flat,
            weights_flat=weights_flat,
            token_ids_flat=token_ids,
            token_space_len=B * T,
            local_expert_call=local_expert_call,
        )
        return agg.reshape(B, T, H)

    def route_expert_choice(
        self,
        tokens: torch.Tensor,               # [N, H]
        scores: torch.Tensor,               # [N, E] (pre-softmax)
        capacity: int,                      # tokens per expert
        num_local_experts: int,
        temperature: float = 1.0,
        local_expert_call=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """EC routing: each expert selects `capacity` tokens, then dispatch.

        Returns aggregated output [N, H], expert_counts [E], importance [E].
        """
        dtype = tokens.dtype
        N, E = scores.shape
        temp = max(temperature, 1e-6)
        # Top-k per expert across tokens (E x capacity)
        top_scores, top_indices = torch.topk((scores / temp).transpose(0, 1), k=capacity, dim=-1)
        selected = tokens.index_select(0, top_indices.reshape(-1)).view(E * capacity, -1)
        weights = torch.softmax(top_scores, dim=-1).reshape(-1).to(dtype)

        owners_flat = (torch.arange(E, device=tokens.device).unsqueeze(1).expand(E, capacity).reshape(-1) // num_local_experts)
        local_idx_flat = (torch.arange(E, device=tokens.device).unsqueeze(1).expand(E, capacity).reshape(-1) % num_local_experts)
        token_ids = top_indices.reshape(-1)
        agg = self._dispatch_and_combine(
            expanded_tokens=selected.to(dtype),
            owners_flat=owners_flat,
            local_idx_flat=local_idx_flat,
            weights_flat=weights,
            token_ids_flat=token_ids,
            token_space_len=N,
            local_expert_call=local_expert_call,
        )
        # Compute simple counts/importances for aux logging
        counts = torch.full((E,), float(capacity), device=tokens.device)
        importance = top_scores.sum(dim=1)
        return agg, counts, importance

    # Caller must provide local_expert_call callbacks to route_per_token_topk/route_expert_choice.
