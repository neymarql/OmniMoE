"""Omni-Stack MoE multimodal model integrating SigLIP and Qwen."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from .llm_qwen import QwenMoELLM
from .moe_layer import split_params_into_different_moe_groups_for_optimizer
from .projector_bridge import ProjectorBridge
from .vision_siglip import SiglipVisionWithMoE


class OmniMoEConfig(PretrainedConfig):
    model_type = "omni_stack_moe"

    def __init__(
        self,
        llm_model_name: str,
        vision_model_name: str,
        tokenizer_name: Optional[str] = None,
        max_text_length: int = 4096,
        projector: Optional[Dict[str, Any]] = None,
        text_moe: Optional[Dict[str, Any]] = None,
        vision_moe: Optional[Dict[str, Any]] = None,
        router: Optional[Dict[str, Any]] = None,
        alignment: Optional[Dict[str, Any]] = None,
        precision: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm_model_name = llm_model_name
        self.vision_model_name = vision_model_name
        self.tokenizer_name = tokenizer_name or llm_model_name
        self.max_text_length = max_text_length
        self.projector_cfg = projector or {}
        self.text_moe_cfg = text_moe or {}
        self.vision_moe_cfg = vision_moe or {}
        self.router_cfg = router or {}
        self.alignment_cfg = alignment or {}
        self.precision_cfg = precision or {}


class OmniMoEModel(PreTrainedModel):
    config_class = OmniMoEConfig

    def __init__(self, config: OmniMoEConfig) -> None:
        super().__init__(config)
        text_moe = config.text_moe_cfg
        router_common = config.router_cfg
        vision_moe = config.vision_moe_cfg
        projector_cfg = config.projector_cfg

        self.vision = SiglipVisionWithMoE(
            model_name_or_path=config.vision_model_name,
            num_experts=vision_moe.get("num_experts", 8),
            back_k_layers=vision_moe.get("back_k_layers", 2),
            router_top_k=vision_moe.get("router_top_k", 1),
            ep_size=vision_moe.get("ep_size", 8),
            capacity_factor=router_common.get("capacity_factor", 1.25),
            min_capacity=router_common.get("min_capacity", 4),
            drop_tokens=router_common.get("token_drop", False),
            noisy_gate_policy=router_common.get("noisy_gate_policy", "Jitter"),
        )
        self.vision_hidden = self.vision.hidden_size

        self.projector = ProjectorBridge(
            hidden_size=projector_cfg.get("hidden_size", self.vision_hidden),
            num_queries=projector_cfg.get("num_queries", 32),
            num_layers=projector_cfg.get("num_layers", 2),
            num_heads=projector_cfg.get("num_heads", 8),
            num_experts=projector_cfg.get("num_experts", 8),
        )

        self.qwen = QwenMoELLM(
            model_name_or_path=config.llm_model_name,
            moe_layers=text_moe.get("moe_layers", []),
            num_experts=text_moe.get("num_experts", 16),
            router_top_k=text_moe.get("router_top_k", 1),
            ep_size=text_moe.get("ep_size", 8),
            capacity_factor=router_common.get("capacity_factor", 1.25),
            min_capacity=router_common.get("min_capacity", 4),
            drop_tokens=router_common.get("token_drop", False),
            noisy_gate_policy=router_common.get("noisy_gate_policy", "Jitter"),
        )
        self.llm = self.qwen.model
        self.tokenizer = self.qwen.tokenizer
        image_token = config.alignment_cfg.get("image_token", "<image>")
        if image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([image_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.aux_loss_coef = router_common.get("aux_loss_coef", 0.01)
        self.num_projected_tokens = projector_cfg.get("num_queries", 32)

    def get_input_embeddings(self) -> nn.Module:  # noqa: D401
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:  # noqa: D401
        self.llm.set_input_embeddings(value)

    def _strip_image_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        filtered_ids: list[torch.Tensor] = []
        filtered_attention: list[torch.Tensor] = []
        filtered_labels: list[torch.Tensor] = [] if labels is not None else []
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        for idx in range(input_ids.size(0)):
            mask = input_ids[idx] != self.image_token_id
            filtered_ids.append(input_ids[idx][mask])
            if attention_mask is not None:
                filtered_attention.append(attention_mask[idx][mask])
            else:
                filtered_attention.append(torch.ones_like(input_ids[idx][mask]))
            if labels is not None:
                filtered_labels.append(labels[idx][mask])
        max_len = max(t.size(0) for t in filtered_ids)
        padded_ids = input_ids.new_full((input_ids.size(0), max_len), pad_token_id)
        padded_attention = input_ids.new_zeros((input_ids.size(0), max_len))
        padded_labels = None
        if labels is not None:
            padded_labels = labels.new_full((labels.size(0), max_len), -100)
        for idx, tokens in enumerate(filtered_ids):
            length = tokens.size(0)
            padded_ids[idx, :length] = tokens
            padded_attention[idx, :length] = filtered_attention[idx][:length]
            if padded_labels is not None:
                padded_labels[idx, :length] = filtered_labels[idx][:length]
        return padded_ids, padded_attention, padded_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, ...], ...]] = None,
        **kwargs: Any,
    ) -> Any:
        if past_key_values is not None:
            return self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                **kwargs,
            )

        device = input_ids.device
        if image_mask is None and pixel_values is not None:
            image_mask = torch.ones((input_ids.size(0),), dtype=torch.long, device=device)
        if image_mask is None:
            image_mask = torch.zeros((input_ids.size(0),), dtype=torch.long, device=device)

        text_ids, text_attention, filtered_labels = self._strip_image_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        text_embeds = self.get_input_embeddings()(text_ids)

        if pixel_values is not None:
            vision_outputs = self.vision(pixel_values)
        else:
            vision_outputs = torch.zeros(
                (input_ids.size(0), 1, self.vision_hidden), device=device, dtype=text_embeds.dtype
            )
        image_mask_tokens = image_mask.unsqueeze(1).float()
        projected_tokens = self.projector(
            image_embeddings=vision_outputs,
            text_embeddings=text_embeds,
            image_mask=image_mask_tokens,
        )
        batch_size = input_ids.size(0)
        projected_tokens = projected_tokens.to(text_embeds.dtype)
        text_embeds = text_embeds.to(projected_tokens.device)

        combined_embeds = torch.cat([projected_tokens, text_embeds], dim=1)
        proj_mask = image_mask.unsqueeze(1).expand(-1, self.num_projected_tokens)
        filtered_mask = text_attention
        combined_attention = torch.cat([proj_mask.to(filtered_mask.dtype), filtered_mask], dim=1)

        augmented_labels = None
        if filtered_labels is not None:
            pad_prefix = torch.full(
                (filtered_labels.size(0), self.num_projected_tokens),
                -100,
                dtype=filtered_labels.dtype,
                device=filtered_labels.device,
            )
            augmented_labels = torch.cat([pad_prefix, filtered_labels], dim=1)

        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            labels=augmented_labels,
            **kwargs,
        )

        if augmented_labels is not None:
            aux_loss = self.projector.aux_loss().to(outputs.logits.device)
            outputs.loss = outputs.loss + self.aux_loss_coef * aux_loss
        return outputs

    def configure_optimizer_param_groups(self) -> Any:
        if split_params_into_different_moe_groups_for_optimizer is None:
            return self.parameters()
        return split_params_into_different_moe_groups_for_optimizer({"params": self.parameters()})


__all__ = ["OmniMoEConfig", "OmniMoEModel"]
