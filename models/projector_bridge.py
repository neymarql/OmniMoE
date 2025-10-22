"""Cross-modal projector bridging SigLIP embeddings and Qwen tokens."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_layer import ExpertFFN


class CrossModalMoELayer(nn.Module):
    """Cross-attention layer with expert routing over image and text streams."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_dim: int,
        num_experts: int,
        ec_routing: bool = False,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.image_gate = nn.Linear(hidden_size * 2, num_experts)
        self.text_gate = nn.Linear(hidden_size * 2, num_experts)
        # Router controls
        self.register_buffer("_router_temperature", torch.tensor(1.0), persistent=False)
        self.register_buffer("_router_jitter_std", torch.tensor(0.0), persistent=False)
        self.experts = nn.ModuleList(ExpertFFN(hidden_size, intermediate_dim) for _ in range(num_experts))
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self._last_gate_entropy: Optional[torch.Tensor] = None
        self.ec_routing = ec_routing
        self.capacity_factor = capacity_factor

    def forward(
        self,
        query_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_context: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:self_attn")
        except Exception as e:
            print(f"[NVTX][WARN] {e}")
        q_norm = self.norm_q(query_tokens)
        attn_output, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        query_tokens = query_tokens + self.dropout(attn_output)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as e:
            print(f"[NVTX][WARN] {e}")

        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:cross_attn")
        except Exception as e:
            print(f"[NVTX][WARN] {e}")
        cross_norm = self.norm_cross(query_tokens)
        key_value = image_tokens if image_mask is None else image_tokens * image_mask.unsqueeze(-1)
        cross_output, _ = self.cross_attn(cross_norm, key_value, key_value, need_weights=False)
        query_tokens = query_tokens + self.dropout(cross_output)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as e:
            print(f"[NVTX][WARN] {e}")

        image_context = image_tokens.mean(dim=1, keepdim=True)
        text_context = text_context.mean(dim=1, keepdim=True)

        img_gate_in = torch.cat([image_tokens, text_context.expand_as(image_tokens)], dim=-1)
        text_gate_in = torch.cat([query_tokens, image_context.expand_as(query_tokens)], dim=-1)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:gating")
        except Exception as e:
            print(f"[NVTX][WARN] {e}")
        image_logits = self.image_gate(img_gate_in).float()
        text_logits = self.text_gate(text_gate_in).float()
        if float(self._router_jitter_std.item()) > 0:
            image_logits = image_logits + self._router_jitter_std * torch.randn_like(image_logits)
            text_logits = text_logits + self._router_jitter_std * torch.randn_like(text_logits)
        temp = torch.clamp(self._router_temperature, min=1e-3)
        image_probs = F.softmax(image_logits / temp, dim=-1)
        text_probs = F.softmax(text_logits / temp, dim=-1)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as e:
            print(f"[NVTX][WARN] {e}")

        ffn_in = self.norm_ffn(query_tokens)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:text_experts")
        except Exception as e:
            print(f"[NVTX][WARN] {e}")
        if not self.ec_routing:
            # Standard soft mixture
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(ffn_in))
            stacked = torch.stack(expert_outputs, dim=-2)
            text_probs = text_probs.unsqueeze(-1)
            mixture = (stacked * text_probs).sum(dim=-2)
            query_tokens = query_tokens + self.dropout(mixture)
        else:
            # Expert-Choice style: experts select top tokens up to capacity, then combine
            B, T, H = ffn_in.shape
            E = text_probs.shape[-1]
            cap = max(1, int((self.capacity_factor * T + E - 1) // E))
            chosen = torch.zeros((B, T, E), dtype=torch.bool, device=ffn_in.device)
            # For each expert, select tokens with highest score
            topk_vals, topk_idx = torch.topk(text_probs.transpose(1, 2), k=min(cap, T), dim=-1)
            # Build mask
            for b in range(B):
                for e in range(E):
                    idxs = topk_idx[b, e]
                    chosen[b, idxs, e] = True
            # Compute expert outputs and accumulate where chosen
            acc = torch.zeros_like(ffn_in)
            counts = torch.zeros((B, T, 1), dtype=ffn_in.dtype, device=ffn_in.device)
            for e, expert in enumerate(self.experts):
                out_e = expert(ffn_in)
                mask_e = chosen[:, :, e].unsqueeze(-1)
                acc = acc + out_e * mask_e
                counts = counts + mask_e.to(counts.dtype)
            counts = counts.clamp_min(1.0)
            query_tokens = query_tokens + self.dropout(acc / counts)
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as e:
            print(f"[NVTX][WARN] {e}")

        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push("proj:image_experts")
        except Exception as e:
            print(f"[NVTX][WARN] {e}")
        if not self.ec_routing:
            image_expert_outputs = []
            for expert in self.experts:
                image_expert_outputs.append(expert(image_tokens))
            img_stacked = torch.stack(image_expert_outputs, dim=-2)
            image_probs = image_probs.unsqueeze(-1)
            image_tokens = image_tokens + self.dropout((img_stacked * image_probs).sum(dim=-2))
        else:
            # EC for image stream
            B, V, H = image_tokens.shape
            E = image_probs.shape[-1]
            cap_img = max(1, int((self.capacity_factor * V + E - 1) // E))
            chosen_img = torch.zeros((B, V, E), dtype=torch.bool, device=image_tokens.device)
            topk_vals_i, topk_idx_i = torch.topk(image_probs.transpose(1, 2), k=min(cap_img, V), dim=-1)
            for b in range(B):
                for e in range(E):
                    idxs = topk_idx_i[b, e]
                    chosen_img[b, idxs, e] = True
            acc_i = torch.zeros_like(image_tokens)
            counts_i = torch.zeros((B, V, 1), dtype=image_tokens.dtype, device=image_tokens.device)
            for e, expert in enumerate(self.experts):
                out_e = expert(image_tokens)
                mask_e = chosen_img[:, :, e].unsqueeze(-1)
                acc_i = acc_i + out_e * mask_e
                counts_i = counts_i + mask_e.to(counts_i.dtype)
            image_tokens = image_tokens + self.dropout(acc_i / counts_i.clamp_min(1.0))
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
        except Exception as e:
            print(f"[NVTX][WARN] {e}")

        # Router metrics
        with torch.no_grad():
            # mean entropy of gates across tokens
            ent_img = -(image_probs.clamp_min(1e-8) * (image_probs.clamp_min(1e-8)).log()).sum(-1).mean()
            ent_txt = -(text_probs.clamp_min(1e-8) * (text_probs.clamp_min(1e-8)).log()).sum(-1).mean()
            self._last_gate_entropy = 0.5 * (ent_img + ent_txt)
        aux_loss = self._load_balance_loss(image_probs.squeeze(-1), text_probs.squeeze(-1))
        self._last_aux_loss = aux_loss.detach()
        return query_tokens, image_tokens

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.layers = nn.ModuleList(
            CrossModalMoELayer(
                hidden_size, num_heads, hidden_size * 4, num_experts,
                ec_routing=ec_routing, capacity_factor=capacity_factor
            ) for _ in range(num_layers)
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
            queries, image_embeddings = layer(queries, image_embeddings, text_embeddings, image_mask)
        return queries

    def aux_loss(self) -> torch.Tensor:
        losses = []
        for layer in self.layers:
            aux = getattr(layer, "_last_aux_loss", None)
            if aux is not None:
                losses.append(aux)
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).mean()

    def set_active_layers(self, n: int) -> None:
        self.active_layers = max(1, min(n, len(self.layers)))

    def set_router_temperature(self, temperature: float) -> None:
        self._temperature = float(temperature)
        for layer in self.layers:
            layer._router_temperature = torch.tensor(self._temperature, device=layer._router_temperature.device)

    def set_router_jitter(self, std: float) -> None:
        self._jitter_std = float(std)
        for layer in self.layers:
            layer._router_jitter_std = torch.tensor(self._jitter_std, device=layer._router_jitter_std.device)

    def set_ec_routing(self, enabled: bool) -> None:
        self._ec_routing = bool(enabled)
        for layer in self.layers:
            layer.ec_routing = self._ec_routing


__all__ = ["ProjectorBridge"]
