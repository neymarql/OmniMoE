"""Qwen LLM wrapper that injects DeepSpeed MoE feed-forward experts."""
from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .moe_layer import MoEFeedForward


class QwenMoELLM(nn.Module):
    """Loads Qwen and replaces specified FFN blocks with MoE layers."""

    def __init__(
        self,
        model_name_or_path: str,
        moe_layers: Iterable[int],
        num_experts: int,
        router_top_k: int,
        ep_size: int,
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        drop_tokens: bool = False,
        noisy_gate_policy: str = "Jitter",
        use_shared_expert: bool = True,
        shared_expert_scale: float = 0.1,
        attn_implementation: str = "flash_attention_2",
        use_megablocks_dropless: bool = False,
        use_expert_choice_router: bool = False,
    ) -> None:
        super().__init__()
        self.attn_implementation = attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                attn_implementation=attn_implementation,
            )
        except Exception as err:
            if attn_implementation == "flash_attention_3":
                print(f"[QwenMoELLM][WARN] flash_attention_3 unavailable ({err}); falling back to flash_attention_2")
                self.attn_implementation = "flash_attention_2"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    attn_implementation="flash_attention_2",
                )
            else:
                print(f"[QwenMoELLM][ERROR] Failed to load model with attention={attn_implementation}: {err}")
                raise
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        hidden_size = self.model.config.hidden_size
        ffn_dim = self.model.config.intermediate_size
        layers = self._locate_transformer_layers()
        total_layers = len(layers)
        target_layers: List[int] = sorted({idx for idx in moe_layers if 0 <= idx < total_layers})
        for idx in target_layers:
            block = layers[idx]
            ffn_module = self._extract_ffn(block)
            moe_ffn = MoEFeedForward(
                hidden_size=hidden_size,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                ep_size=ep_size,
                router_top_k=router_top_k,
                capacity_factor=capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=drop_tokens,
                noisy_gate_policy=noisy_gate_policy,
                scope="text",
                use_shared_expert=use_shared_expert,
                shared_expert_scale=shared_expert_scale,
                use_megablocks_dropless=use_megablocks_dropless,
                use_expert_choice_router=use_expert_choice_router,
            )
            self._inject_ffn(block, moe_ffn)
            del ffn_module

    def _locate_transformer_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers  # type: ignore[return-value]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h  # type: ignore[return-value]
        raise AttributeError("Unsupported Qwen model structure")

    @staticmethod
    def _extract_ffn(block: nn.Module) -> nn.Module:
        if hasattr(block, "mlp"):
            return block.mlp
        if hasattr(block, "ffn"):
            return block.ffn
        raise AttributeError("Unable to locate FFN module in Qwen block")

    @staticmethod
    def _inject_ffn(block: nn.Module, moe_ffn: nn.Module) -> None:
        if hasattr(block, "mlp"):
            block.mlp = moe_ffn  # type: ignore[attr-defined]
        elif hasattr(block, "ffn"):
            block.ffn = moe_ffn  # type: ignore[attr-defined]
        else:
            raise AttributeError("Unable to inject MoE FFN into Qwen block")

    def forward(self, *args, **kwargs):  # noqa: D401
        return self.model(*args, **kwargs)


__all__ = ["QwenMoELLM"]
