"""High-level training loop for Omni-Stack MoE."""
from __future__ import annotations

import math
from typing import Dict, List

import deepspeed
import torch
from torch.utils.data import DataLoader

from OmniMoE.data.collate import multimodal_collate_fn
from OmniMoE.data.dataset import load_datasets_for_stage
from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel
from OmniMoE.training.scheduler import build_scheduler
from OmniMoE.training.utils import ensure_dir, maybe_load_checkpoint, set_seed


class CurriculumTrainer:
    """Runs staged training according to provided curriculum metadata."""

    def __init__(
        self,
        config: OmniMoEConfig,
        dataset_cfg: Dict,
        hyperparams: Dict,
        deepspeed_cfg: Dict,
        output_dir: str,
        resume_from: str = "",
    ) -> None:
        self.config = config
        self.dataset_cfg = dataset_cfg
        self.hyperparams = hyperparams
        self.deepspeed_cfg = deepspeed_cfg
        self.output_dir = output_dir
        ensure_dir(output_dir)
        set_seed(hyperparams.get("global", {}).get("seed", 42))

        self.model = OmniMoEModel(config)
        self.tokenizer = self.model.tokenizer
        trainable_params = self.model.configure_optimizer_param_groups()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=hyperparams.get("stage1", {}).get("learning_rate", 2e-5),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=hyperparams.get("global", {}).get("weight_decay", 0.1),
        )
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config=self.deepspeed_cfg,
        )
        maybe_load_checkpoint(self.model_engine, resume_from)
        self.curriculum_order: List[str] = list(hyperparams.get("curriculum", {}).get("order", []))
        if not self.curriculum_order:
            self.curriculum_order = sorted(dataset_cfg.keys())

    def _build_dataloader(self, stage: str) -> DataLoader:
        stage_cfg = self.dataset_cfg[stage]
        dataset = load_datasets_for_stage(
            stage_cfg,
            tokenizer=self.tokenizer,
            max_text_length=self.config.max_text_length,
        )
        per_gpu_batch = self.hyperparams.get("global", {}).get("train_batch_size_per_gpu", 2)
        return DataLoader(
            dataset,
            batch_size=per_gpu_batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=multimodal_collate_fn,
        )

    def train(self) -> None:
        global_step = 0
        for stage in self.curriculum_order:
            dataloader = self._build_dataloader(stage)
            stage_params = self.hyperparams.get(stage, {})
            epochs = stage_params.get("epochs", 1)
            learning_rate = stage_params.get("learning_rate", 1e-5)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            warmup_steps = stage_params.get("warmup_steps", 0)
            total_steps = stage_params.get("max_steps", len(dataloader) * epochs)
            scheduler = build_scheduler(self.optimizer, {"warmup_steps": warmup_steps}, total_steps)
            stage_dir = f"{self.output_dir}/{stage}"
            ensure_dir(stage_dir)

            for epoch in range(epochs):
                for batch_idx, batch in enumerate(dataloader):
                    batch = {k: v.to(self.model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = self.model_engine(**batch)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    scheduler.step()
                    global_step += 1
                    if global_step % self.hyperparams.get("global", {}).get("logging_steps", 50) == 0:
                        if deepspeed.comm.get_rank() == 0:
                            print(f"[Stage {stage}] step {global_step} loss={loss.item():.4f}")
                    if global_step % self.hyperparams.get("global", {}).get("save_steps", 1000) == 0:
                        if deepspeed.comm.get_rank() == 0:
                            ckpt_path = f"{stage_dir}/step-{global_step}"
                            ensure_dir(ckpt_path)
                            self.model_engine.save_checkpoint(ckpt_path)
                    if global_step >= total_steps:
                        break
                if global_step >= total_steps:
                    break
            if deepspeed.comm.get_rank() == 0:
                final_path = f"{stage_dir}/final"
                ensure_dir(final_path)
                self.model_engine.save_checkpoint(final_path)

        if deepspeed.comm.get_rank() == 0:
            final_dir = f"{self.output_dir}/final"
            ensure_dir(final_dir)
            self.model_engine.save_checkpoint(final_dir)


__all__ = ["CurriculumTrainer"]
