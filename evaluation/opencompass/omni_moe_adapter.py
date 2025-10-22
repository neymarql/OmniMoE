"""OpenCompass model adapter for OmniMoE.

This module exposes a minimal adapter to allow OpenCompass to load the
OmniMoE HuggingFace-style checkpoint and run generate() on multimodal
prompts that include an <image> tag.

Note: This is a lightweight integration stub; depending on the installed
OpenCompass version, the interface names may slightly differ. Adjust the
class name/signature to the exact API in your environment.
"""
from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image
import torch

from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel


class OmniMoECompassModel:
    def __init__(self, model_dir: str, config_path: str, device: str = "cuda") -> None:
        config = OmniMoEConfig.from_json_file(config_path)
        self.model = OmniMoEModel.from_pretrained(model_dir, config=config)
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.model.vision.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt
        tokenized = self.model.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
            pixel_values=pixel_values,
            image_mask=torch.ones((1,), dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
        )
        ans = self.model.tokenizer.decode(out[0, tokenized["input_ids"].size(1) :], skip_special_tokens=True)
        return ans.strip()

