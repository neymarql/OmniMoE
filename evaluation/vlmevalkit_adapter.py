"""VLMEvalKit adapter for OmniMoE.

Implements a simple predict(image, question) function so VLMEvalKit can
evaluate OmniMoE on VQAv2/COCO/ScienceQA.
"""
from __future__ import annotations

from typing import Any

from PIL import Image
import torch

from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel


class OmniMoEVLMEval:
    def __init__(self, model_dir: str, config_path: str, device: str = "cuda") -> None:
        cfg = OmniMoEConfig.from_json_file(config_path)
        self.model = OmniMoEModel.from_pretrained(model_dir, config=cfg)
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, image_path: str, question: str, max_new_tokens: int = 32) -> str:
        img = Image.open(image_path).convert("RGB")
        pixel = self.model.vision.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        prompt = question if question.strip().startswith("<image>") else f"<image>\n{question}"
        tokenized = self.model.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized.get("attention_mask"),
            pixel_values=pixel,
            image_mask=torch.ones((1,), dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
        )
        return self.model.tokenizer.decode(out[0, tokenized["input_ids"].size(1):], skip_special_tokens=True).strip()

