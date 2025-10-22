"""Offline generation script for Omni-Stack MoE."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses with Omni-Stack MoE")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_model(model_path: str, config_path: str) -> OmniMoEModel:
    config = OmniMoEConfig.from_json_file(config_path)
    model = OmniMoEModel.from_pretrained(model_path, config=config)
    model.eval()
    model.cuda()
    return model


def main() -> None:
    args = parse_args()
    model = load_model(args.model_path, args.config_path)

    image = Image.open(Path(args.image)).convert("RGB")
    pixel_values = model.vision.processor(images=image, return_tensors="pt")["pixel_values"].cuda()
    prompt = f"<image>\n{args.question.strip()}"
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.llm.device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            image_mask=torch.ones((1,), dtype=torch.long, device=model.llm.device),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
        )
    output_ids = generated[0, inputs["input_ids"].size(1) :]
    answer = model.tokenizer.decode(output_ids, skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
