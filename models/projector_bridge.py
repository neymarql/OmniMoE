"""Cross-modal projector bridging SigLIP embeddings and Qwen tokens."""
from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .moe_layer import ExpertFFN


class CrossModalMoELayer(nn.Module):
    """Cross-attention layer with expert routing over image and text streams."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_dim: int,
        num_experts: int,
        ep_group: Optional[dist.ProcessGroup] = None,
        ep_size: int = 1,
        ep_rank: int = 0,
        global_offset: int = 0,
        local_experts: int = 0,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_group = ep_group
        self.ep_size = max(1, ep_size)
        self.ep_rank = ep_rank
        self.global_offset = global_offset
        self.num_local_experts = num_experts if ep_group is None else local_experts
        self.top_k = num_experts if ep_group is None else max(1, min(top_k, num_experts))
        self.capacity_factor = capacity_factor

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.image_gate = nn.Linear(hidden_size * 2, num_experts)
        self.text_gate = nn.Linear(hidden_size * 2, num_experts)
        self.register_buffer("_router_temperature", torch.tensor(1.0), persistent=False)
        self.register_buffer("_router_jitter_std", torch.tensor(0.0), persistent=False)
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self._last_gate_entropy: Optional[torch.Tensor] = None
        self.local_experts = nn.ModuleList(ExpertFFN(hidden_size, intermediate_dim) for _ in range(self.num_local_experts))

    def forward(
        self,
        query_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_context: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = query_tokens.device
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:self_attn")
        except Exception as err:
            print(f"[NVTX][WARN] {err}")
        q_norm = self.norm_q(query_tokens)
        attn_output, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        query_tokens = query_tokens + self.dropout(attn_output)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as err:
            print(f"[NVTX][WARN] {err}")

        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:cross_attn")
        except Exception as err:
            print(f"[NVTX][WARN] {err}")
        cross_norm = self.norm_cross(query_tokens)
        key_value = image_tokens if image_mask is None else image_tokens * image_mask.unsqueeze(-1)
        cross_output, _ = self.cross_attn(cross_norm, key_value, key_value, need_weights=False)
        query_tokens = query_tokens + self.dropout(cross_output)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as err:
            print(f"[NVTX][WARN] {err}")

        image_context = image_tokens.mean(dim=1, keepdim=True)
        text_context = text_context.mean(dim=1, keepdim=True)

        img_gate_in = torch.cat([image_tokens, text_context.expand_as(image_tokens)], dim=-1)
        text_gate_in = torch.cat([query_tokens, image_context.expand_as(query_tokens)], dim=-1)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:gating")
        except Exception as err:
            print(f"[NVTX][WARN] {err}")
        image_logits = self.image_gate(img_gate_in).float()
        text_logits = self.text_gate(text_gate_in).float()
        if float(self._router_jitter_std.item()) > 0 and self.training:
            image_logits = image_logits + self._router_jitter_std * torch.randn_like(image_logits)
            text_logits = text_logits + self._router_jitter_std * torch.randn_like(text_logits)
        temp = torch.clamp(self._router_temperature, min=1e-3)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as err:
            print(f"[NVTX][WARN] {err}")

        if self.ep_group is None:
            image_probs = F.softmax(image_logits / temp, dim=-1)
            text_probs = F.softmax(text_logits / temp, dim=-1)
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push("proj:text_experts")
            except Exception as err:
                print(f"[NVTX][WARN] {err}")
            q_ffn = self.norm_ffn(query_tokens)
            query_tokens, image_tokens = self._dense_moe_forward(q_ffn, query_tokens, image_tokens, image_probs, text_probs)
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
            except Exception as err:
                print(f"[NVTX][WARN] {err}")
            image_for_aux = image_probs
            text_for_aux = text_probs
        else:
            text_topk_vals, text_topk_idx = torch.topk(text_logits / temp, k=self.top_k, dim=-1)
            text_topk_probs = torch.softmax(text_topk_vals, dim=-1)
            image_topk_vals, image_topk_idx = torch.topk(image_logits / temp, k=self.top_k, dim=-1)
            image_topk_probs = torch.softmax(image_topk_vals, dim=-1)
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push("proj:text_experts")
            except Exception as err:
                print(f"[NVTX][WARN] {err}")
            query_tokens = query_tokens + self.dropout(
                self._ep_mlp(self.norm_ffn(query_tokens), text_topk_idx, text_topk_probs)
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
            except Exception as err:
                print(f"[NVTX][WARN] {err}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push("proj:image_experts")
            except Exception as err:
                print(f"[NVTX][WARN] {err}")
            image_tokens = image_tokens + self.dropout(
                self._ep_mlp(image_tokens, image_topk_idx, image_topk_probs)
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
            except Exception as err:
                print(f"[NVTX][WARN] {err}")
            image_for_aux = torch.zeros(*image_topk_idx.shape[:2], self.num_experts, device=device, dtype=image_topk_probs.dtype)
            text_for_aux = torch.zeros(*text_topk_idx.shape[:2], self.num_experts, device=device, dtype=text_topk_probs.dtype)
            image_for_aux.scatter_(-1, image_topk_idx, image_topk_probs)
            text_for_aux.scatter_(-1, text_topk_idx, text_topk_probs)

        with torch.no_grad():
            ent_img = -(image_for_aux.clamp_min(1e-8) * (image_for_aux.clamp_min(1e-8)).log()).sum(-1).mean()
            ent_txt = -(text_for_aux.clamp_min(1e-8) * (text_for_aux.clamp_min(1e-8)).log()).sum(-1).mean()
            self._last_gate_entropy = 0.5 * (ent_img + ent_txt)
        aux_loss = self._load_balance_loss(image_for_aux, text_for_aux)
        self._last_aux_loss = aux_loss.detach()
        return query_tokens, image_tokens

    def _dense_moe_forward(
        self,
        ffn_in: torch.Tensor,
        query_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        image_probs: torch.Tensor,
        text_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_outputs = []
        for expert in self.local_experts:
            expert_outputs.append(expert(ffn_in))
        stacked = torch.stack(expert_outputs, dim=-2)
        mixture = (stacked * text_probs.unsqueeze(-1)).sum(dim=-2)
        query_tokens = query_tokens + self.dropout(mixture)

        image_expert_outputs = []
        for expert in self.local_experts:
            image_expert_outputs.append(expert(image_tokens))
        img_stacked = torch.stack(image_expert_outputs, dim=-2)
        image_tokens = image_tokens + self.dropout((img_stacked * image_probs.unsqueeze(-1)).sum(dim=-2))
        return query_tokens, image_tokens

    def _ep_mlp(
        self,
        tokens: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_group is None or self.ep_size <= 1:
            raise RuntimeError("EP MLP requested without an expert-parallel group")
        device = tokens.device
        dtype = tokens.dtype
        B, T, H = tokens.shape
        K = top_indices.size(-1)
        tokens_flat = tokens.reshape(-1, H)
        expanded_tokens = tokens_flat.unsqueeze(1).expand(-1, K, -1).reshape(-1, H)
        owners = top_indices.reshape(-1, K) // self.num_local_experts
        local_idx = top_indices.reshape(-1, K) % self.num_local_experts
        owners_flat = owners.reshape(-1)
        local_idx_flat = local_idx.reshape(-1)
        weights_flat = top_weights.reshape(-1, K).reshape(-1)
        token_ids = torch.arange(B * T, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
        origin = torch.full_like(owners_flat, fill_value=self.ep_rank)

        order = torch.argsort(owners_flat)
        owners_sorted = owners_flat[order]
        local_idx_sorted = local_idx_flat[order]
        weights_sorted = weights_flat[order]
        token_ids_sorted = token_ids[order]
        origin_sorted = origin[order]
        tokens_sorted = expanded_tokens[order]

        send_counts = torch.bincount(owners_sorted, minlength=self.ep_size)
        send_counts_list = send_counts.tolist()
        counts_tensor = send_counts.to(torch.long)
        counts_gather = [torch.zeros_like(counts_tensor) for _ in range(self.ep_size)]
        dist.all_gather(counts_gather, counts_tensor, group=self.ep_group)
        counts_matrix = torch.stack(counts_gather, dim=0)
        recv_counts = counts_matrix[:, self.ep_rank]
        recv_counts_list = recv_counts.tolist()

        send_tokens = tokens_sorted
        recv_total = int(sum(recv_counts_list))
        recv_tokens = torch.empty((recv_total, H), device=device, dtype=dtype)
        dist.all_to_all_single(recv_tokens, send_tokens, recv_counts_list, send_counts_list, group=self.ep_group)

        send_meta = torch.stack([local_idx_sorted, origin_sorted, token_ids_sorted], dim=1).to(torch.long)
        recv_meta = torch.empty((recv_total, 3), device=device, dtype=torch.long)
        dist.all_to_all_single(recv_meta, send_meta, recv_counts_list, send_counts_list, group=self.ep_group)

        send_weights = weights_sorted.unsqueeze(1).to(dtype)
        recv_weights = torch.empty((recv_total, 1), device=device, dtype=dtype)
        dist.all_to_all_single(recv_weights, send_weights, recv_counts_list, send_counts_list, group=self.ep_group)

        local_idx_recv = recv_meta[:, 0]
        origin_recv = recv_meta[:, 1]
        token_ids_recv = recv_meta[:, 2]
        weights_recv = recv_weights[:, 0]

        outputs: List[torch.Tensor] = []
        origins_out: List[torch.Tensor] = []
        token_ids_out: List[torch.Tensor] = []
        if recv_tokens.size(0) > 0:
            for local_id in range(self.num_local_experts):
                mask = local_idx_recv == local_id
                if mask.any():
                    tokens_local = recv_tokens[mask]
                    weights_local = weights_recv[mask]
                    out_local = self.local_experts[local_id](tokens_local)
                    out_local = out_local * weights_local.unsqueeze(-1)
                    outputs.append(out_local)
                    origins_out.append(origin_recv[mask])
                    token_ids_out.append(token_ids_recv[mask])
        if outputs:
            outputs_cat = torch.cat(outputs, dim=0)
            origins_cat = torch.cat(origins_out, dim=0)
            token_ids_cat = torch.cat(token_ids_out, dim=0)
        else:
            outputs_cat = torch.empty((0, H), device=device, dtype=dtype)
            origins_cat = torch.empty((0,), device=device, dtype=torch.long)
            token_ids_cat = torch.empty((0,), device=device, dtype=torch.long)

        if outputs_cat.size(0) > 0:
            send_back_counts = torch.bincount(origins_cat, minlength=self.ep_size)
        else:
            send_back_counts = torch.zeros(self.ep_size, device=device, dtype=torch.long)
        send_back_counts_list = send_back_counts.tolist()
        counts_back_tensor = send_back_counts.clone()
        gather_back = [torch.zeros_like(counts_back_tensor) for _ in range(self.ep_size)]
        dist.all_gather(gather_back, counts_back_tensor, group=self.ep_group)
        recv_back_counts = torch.stack(gather_back, dim=0)[:, self.ep_rank]
        recv_back_counts_list = recv_back_counts.tolist()
        recv_back_total = int(sum(recv_back_counts_list))

        recv_outputs = torch.empty((recv_back_total, H), device=device, dtype=dtype)
        dist.all_to_all_single(recv_outputs, outputs_cat, recv_back_counts_list, send_back_counts_list, group=self.ep_group)

        recv_token_ids = torch.empty((recv_back_total,), device=device, dtype=torch.long)
        dist.all_to_all_single(recv_token_ids, token_ids_cat.to(torch.long), recv_back_counts_list, send_back_counts_list, group=self.ep_group)

        agg = torch.zeros((tokens_flat.size(0), H), device=device, dtype=dtype)
        if recv_back_total > 0:
            agg.index_add_(0, recv_token_ids, recv_outputs)
        return agg.reshape(B, T, H)

    @staticmethod
    def _load_balance_loss(image_probs: torch.Tensor, text_probs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([image_probs, text_probs], dim=1)
        avg = combined.mean(dim=(0, 1))
        target = torch.full_like(avg, 1.0 / avg.numel())
        return torch.sum((avg - target) ** 2)


class ProjectorBridge(nn.Module):
    """Projects image embeddings into a sequence of visual prompt tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_queries: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        ec_routing: bool = False,
        capacity_factor: float = 1.25,
        ep_size: int = 1,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.ep_size = max(1, ep_size)
        self.ep_group: Optional[dist.ProcessGroup] = None
        self.ep_rank = 0
        self.global_offset = 0
        self.local_num_experts = num_experts
        if self.ep_size > 1 and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
            group_idx = rank // self.ep_size
            start = group_idx * self.ep_size
            ranks = list(range(start, min(start + self.ep_size, world)))
            self.ep_group = dist.new_group(ranks)
            self.ep_rank = rank - start
            self.local_num_experts = num_experts // self.ep_size
            self.global_offset = group_idx * self.local_num_experts
        self.layers = nn.ModuleList(
            CrossModalMoELayer(
                hidden_size,
                num_heads,
                hidden_size * 4,
                num_experts,
                ep_group=self.ep_group,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_offset=self.global_offset,
                local_experts=self.local_num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
            )
            for _ in range(num_layers)
        )
        self.active_layers = num_layers
        self._temperature = 1.0
        self._jitter_std = 0.0
        self._ec_routing = ec_routing

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = image_embeddings.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for li, layer in enumerate(self.layers):
            if li >= self.active_layers:
                break
            layer._router_temperature = torch.tensor(self._temperature, device=queries.device)
            layer._router_jitter_std = torch.tensor(self._jitter_std, device=queries.device)
            queries, image_embeddings = layer(queries, image_embeddings, text_embeddings, image_mask)
        return queries

    def aux_loss(self) -> torch.Tensor:
        losses = []
        for layer in self.layers:
            aux = getattr(layer, "_last_aux_loss", None)
            if aux is not None:
                losses.append(aux)
        if not losses:
            return torch.tensor(0.0, device=self.query_tokens.device)
        return torch.stack(losses).mean()

    def set_active_layers(self, n: int) -> None:
        self.active_layers = max(1, min(n, len(self.layers)))

    def set_router_temperature(self, temperature: float) -> None:
        self._temperature = float(temperature)

    def set_router_jitter(self, std: float) -> None:
        self._jitter_std = float(std)

    def set_ec_routing(self, enabled: bool) -> None:
        self._ec_routing = bool(enabled)
        for layer in self.layers:
            if enabled:
                layer.top_k = max(1, min(layer.num_experts, layer.top_k))


__all__ = ["ProjectorBridge"]
