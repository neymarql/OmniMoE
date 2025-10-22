"""Entry point for Omni-Stack MoE training."""
from __future__ import annotations

import argparse

import deepspeed

from OmniMoE.models.omni_model import OmniMoEConfig
from OmniMoE.training.trainer import CurriculumTrainer
from OmniMoE.training.utils import ensure_dir, load_json, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Omni-Stack MoE")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--hyperparams", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deepspeed.init_distributed()

    model_cfg = load_json(args.model_config)
    dataset_cfg = load_json(args.dataset_config)
    ds_cfg = load_json(args.deepspeed_config)
    hyperparams = load_yaml(args.hyperparams)

    config = OmniMoEConfig(**model_cfg)
    ensure_dir(args.output_dir)

    trainer = CurriculumTrainer(
        config=config,
        dataset_cfg=dataset_cfg,
        hyperparams=hyperparams,
        deepspeed_cfg=ds_cfg,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    trainer.train()


if __name__ == "__main__":
    main()
