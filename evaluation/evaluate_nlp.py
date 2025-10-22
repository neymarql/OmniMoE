"""Evaluate text-only tasks by disabling image inputs."""
from __future__ import annotations

import json
from typing import List

import torch

from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel
from OmniMoE.evaluation.metrics import exact_match


def evaluate_text_qa(model_dir: str, config_path: str, dataset_path: str) -> float:
    config = OmniMoEConfig.from_json_file(config_path)
    model = OmniMoEModel.from_pretrained(model_dir, config=config)
    model.cuda().eval()

    with open(dataset_path, "r", encoding="utf-8") as fp:
        samples = json.load(fp)

    correct = 0
    for entry in samples:
        prompt = entry["prompt"]
        answers: List[str] = entry["answers"]
        tokenized = model.tokenizer(prompt, return_tensors="pt").to(model.llm.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask"),
                max_new_tokens=64,
            )
        response = model.tokenizer.decode(generated[0, tokenized["input_ids"].size(1) :], skip_special_tokens=True)
        correct += exact_match(response, answers)
    return correct / max(len(samples), 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate text QA accuracy")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    accuracy = evaluate_text_qa(args.model_dir, args.config, args.dataset)
    print(f"Text QA accuracy: {accuracy * 100:.2f}%")
