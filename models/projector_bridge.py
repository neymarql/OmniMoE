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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.image_gate = nn.Linear(hidden_size * 2, num_experts)
        self.text_gate = nn.Linear(hidden_size * 2, num_experts)
        self.experts = nn.ModuleList(ExpertFFN(hidden_size, intermediate_dim) for _ in range(num_experts))
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_context: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = self.norm_q(query_tokens)
        attn_output, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        query_tokens = query_tokens + self.dropout(attn_output)

        cross_norm = self.norm_cross(query_tokens)
        key_value = image_tokens if image_mask is None else image_tokens * image_mask.unsqueeze(-1)
        cross_output, _ = self.cross_attn(cross_norm, key_value, key_value, need_weights=False)
        query_tokens = query_tokens + self.dropout(cross_output)

        image_context = image_tokens.mean(dim=1, keepdim=True)
        text_context = text_context.mean(dim=1, keepdim=True)

        img_gate_in = torch.cat([image_tokens, text_context.expand_as(image_tokens)], dim=-1)
        text_gate_in = torch.cat([query_tokens, image_context.expand_as(query_tokens)], dim=-1)
        image_logits = self.image_gate(img_gate_in)
        text_logits = self.text_gate(text_gate_in)
        image_probs = F.softmax(image_logits, dim=-1)
        text_probs = F.softmax(text_logits, dim=-1)

        ffn_in = self.norm_ffn(query_tokens)
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(ffn_in))
        stacked = torch.stack(expert_outputs, dim=-2)
        text_probs = text_probs.unsqueeze(-1)
        mixture = (stacked * text_probs).sum(dim=-2)
        query_tokens = query_tokens + self.dropout(mixture)

        image_expert_outputs = []
        for expert in self.experts:
            image_expert_outputs.append(expert(image_tokens))
        img_stacked = torch.stack(image_expert_outputs, dim=-2)
        image_probs = image_probs.unsqueeze(-1)
        image_tokens = image_tokens + self.dropout((img_stacked * image_probs).sum(dim=-2))

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.layers = nn.ModuleList(
            CrossModalMoELayer(hidden_size, num_heads, hidden_size * 4, num_experts) for _ in range(num_layers)
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = image_embeddings.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
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


__all__ = ["ProjectorBridge"]
