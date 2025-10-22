"""High-throughput inference utilities with optional KV-cache reuse."""
from __future__ import annotations

from typing import List

import torch

from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel


class OmniInferenceEngine:
    """Wraps Omni-Stack MoE for batched inference and streaming decode."""

    def __init__(self, model_dir: str, config_path: str, device: str = "cuda") -> None:
        config = OmniMoEConfig.from_json_file(config_path)
        self.model = OmniMoEModel.from_pretrained(model_dir, config=config)
        self.model.eval().to(device)
        self.device = device

    def preprocess(self, prompts: List[str], images) -> dict:
        tokenized = self.model.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        pixel_values = self.model.vision.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        image_mask = torch.ones((len(prompts),), dtype=torch.long, device=self.device)
        return {**tokenized, "pixel_values": pixel_values, "image_mask": image_mask}

    @torch.no_grad()
    def generate(self, prompts: List[str], images, max_new_tokens: int = 128) -> List[str]:
        inputs = self.preprocess(prompts, images)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs["pixel_values"],
            image_mask=inputs["image_mask"],
            max_new_tokens=max_new_tokens,
        )
        responses = []
        for idx in range(outputs.size(0)):
            answer_ids = outputs[idx, inputs["input_ids"].size(1) :]
            responses.append(self.model.tokenizer.decode(answer_ids, skip_special_tokens=True).strip())
        return responses


__all__ = ["OmniInferenceEngine"]
