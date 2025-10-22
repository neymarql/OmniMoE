"""Evaluate Omni-Stack MoE on VQA-style benchmarks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from PIL import Image

from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel
from OmniMoE.evaluation.metrics import exact_match


def evaluate_vqa(model_dir: str, config_path: str, dataset_path: str) -> float:
    config = OmniMoEConfig.from_json_file(config_path)
    model = OmniMoEModel.from_pretrained(model_dir, config=config)
    model.cuda().eval()

    with open(dataset_path, "r", encoding="utf-8") as fp:
        samples = json.load(fp)

    correct = 0
    for entry in samples:
        image_path = Path(entry["image_path"]).expanduser()
        question = entry["question"]
        answers: List[str] = entry["answers"]
        image = Image.open(image_path).convert("RGB")
        pixel_values = model.vision.processor(images=image, return_tensors="pt")["pixel_values"].cuda()
        prompt = f"<image>\n{question}"
        tokenized = model.tokenizer(prompt, return_tensors="pt").to(model.llm.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask"),
                pixel_values=pixel_values,
                image_mask=torch.ones((1,), dtype=torch.long, device=model.llm.device),
                max_new_tokens=32,
            )
        output = model.tokenizer.decode(generated[0, tokenized["input_ids"].size(1) :], skip_special_tokens=True)
        correct += exact_match(output, answers)
    return correct / max(len(samples), 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate VQA accuracy")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    accuracy = evaluate_vqa(args.model_dir, args.config, args.dataset)
    print(f"VQA accuracy: {accuracy * 100:.2f}%")
