"""SigLIP vision encoder with MoE experts injected into the final layers."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SiglipVisionModel

from .moe_layer import MoEFeedForward


class SiglipVisionWithMoE(nn.Module):
    """Wraps the SigLIP vision backbone with sparse MoE FFN layers."""

    def __init__(
        self,
        model_name_or_path: str,
        num_experts: int,
        back_k_layers: int,
        router_top_k: int,
        ep_size: int,
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        drop_tokens: bool = False,
        noisy_gate_policy: str = "Jitter",
        use_mid_tokens: bool = True,
        mid_layer_index: int = -4,
        use_shared_expert: bool = True,
        shared_expert_scale: float = 0.1,
        use_megablocks_dropless: bool = False,
    ) -> None:
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(model_name_or_path)
        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.hidden_size = self.model.config.hidden_size
        self.intermediate_size = self.model.config.intermediate_size
        self.num_layers = self.model.config.num_hidden_layers
        self.use_mid_tokens = use_mid_tokens
        self.mid_layer_index = mid_layer_index

        if back_k_layers <= 0 or back_k_layers > self.num_layers:
            raise ValueError("back_k_layers must be between 1 and num_hidden_layers")

        target_layers: List[int] = list(range(self.num_layers - back_k_layers, self.num_layers))
        encoder_layers = self._get_encoder_layers()
        for idx in target_layers:
            layer = encoder_layers[idx]
            ffn = self._extract_mlp(layer)
            moe_ffn = MoEFeedForward(
                hidden_size=self.hidden_size,
                ffn_dim=self.intermediate_size,
                num_experts=num_experts,
                ep_size=ep_size,
                router_top_k=router_top_k,
                capacity_factor=capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=drop_tokens,
                noisy_gate_policy=noisy_gate_policy,
                scope="vision",
                use_shared_expert=use_shared_expert,
                shared_expert_scale=shared_expert_scale,
                use_megablocks_dropless=use_megablocks_dropless,
            )
            self._inject_mlp(layer, moe_ffn)
            del ffn

    def _get_encoder_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            encoder = self.model.vision_model.encoder
        elif hasattr(self.model, "encoder"):
            encoder = self.model.encoder
        else:
            raise AttributeError("Unable to locate encoder layers in SigLIP model")
        if hasattr(encoder, "layers"):
            return encoder.layers  # type: ignore[return-value]
        if hasattr(encoder, "layer"):
            return encoder.layer  # type: ignore[return-value]
        if hasattr(encoder, "blocks"):
            return encoder.blocks  # type: ignore[return-value]
        raise AttributeError("Unsupported encoder structure for SigLIP model")

    @staticmethod
    def _extract_mlp(block: nn.Module) -> nn.Module:
        if hasattr(block, "mlp"):
            return block.mlp
        if hasattr(block, "ffn"):
            return block.ffn
        if hasattr(block, "layer") and hasattr(block.layer, "mlp"):
            return block.layer.mlp  # type: ignore[return-value]
        raise AttributeError("Unable to locate FFN/MLP submodule in SigLIP block")

    @staticmethod
    def _inject_mlp(block: nn.Module, moe_ffn: nn.Module) -> None:
        if hasattr(block, "mlp"):
            block.mlp = moe_ffn  # type: ignore[attr-defined]
            return
        if hasattr(block, "ffn"):
            block.ffn = moe_ffn  # type: ignore[attr-defined]
            return
        if hasattr(block, "layer") and hasattr(block.layer, "mlp"):
            block.layer.mlp = moe_ffn  # type: ignore[attr-defined]
            return
        raise AttributeError("Unable to inject MoE FFN into SigLIP block")

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=self.use_mid_tokens)
        last_hidden = outputs.last_hidden_state
        result = {"last_hidden": last_hidden}
        if self.use_mid_tokens and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            idx = self.mid_layer_index if self.mid_layer_index < 0 else self.mid_layer_index
            mid = outputs.hidden_states[idx]
            result["mid_hidden"] = mid
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            result["cls"] = outputs.pooler_output
        else:
            # fallback: mean pool
            result["cls"] = last_hidden.mean(dim=1)
        return result


__all__ = ["SiglipVisionWithMoE"]
