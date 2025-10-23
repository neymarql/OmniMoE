"""High-level training loop for Omni-Stack MoE."""
from __future__ import annotations

import math
import os
from typing import Dict, List

import deepspeed
import torch
from torch.utils.data import DataLoader

from OmniMoE.data.collate import multimodal_collate_fn
from OmniMoE.data.dataset import load_datasets_for_stage
from OmniMoE.models.omni_model import OmniMoEConfig, OmniMoEModel
from OmniMoE.training.scheduler import build_scheduler
from OmniMoE.training.optim import build_optimizer
from torch.utils.tensorboard import SummaryWriter
from OmniMoE.training.utils import (
    ensure_dir,
    maybe_load_checkpoint,
    set_seed,
    save_hf_checkpoint,
    enable_all_to_all_monitor,
    get_all_to_all_stats,
)
import subprocess


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

        print("[Trainer] Initializing OmniMoE model...")
        self.model = OmniMoEModel(config)
        self.tokenizer = self.model.tokenizer
        trainable_params = self.model.configure_optimizer_param_groups()
        base_lr = hyperparams.get("stage1", {}).get("learning_rate", 2e-5)
        g = hyperparams.get("global", {})
        optimizer = build_optimizer(
            self.model,
            base_lr=base_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=g.get("weight_decay", 0.1),
            moe_lr_mult=g.get("moe_lr_mult", 1.0),
            projector_lr_mult=g.get("projector_lr_mult", 1.0),
            optimizer_name=g.get("optimizer", "adamw"),
        )
        print("[Trainer] Building DeepSpeed engine...")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config=self.deepspeed_cfg,
        )
        try:
            maybe_load_checkpoint(self.model_engine, resume_from)
        except Exception as e:
            print(f"[Trainer][ERROR] Failed to load checkpoint '{resume_from}': {e}")
            raise
        if deepspeed.comm.get_rank() == 0:
            self._log_param_summary()
        self.curriculum_order: List[str] = list(hyperparams.get("curriculum", {}).get("order", []))
        if not self.curriculum_order:
            self.curriculum_order = sorted(dataset_cfg.keys())
        # TensorBoard writer (rank 0 only)
        try:
            log_dir = os.path.join(self.output_dir, "runs")
            if deepspeed.comm.get_rank() == 0:
                self.tb_writer = SummaryWriter(log_dir=log_dir)
            else:
                self.tb_writer = None
        except Exception as e:
            print(f"[Trainer][WARN] TensorBoard SummaryWriter unavailable: {e}")
            self.tb_writer = None
        if torch.cuda.is_available():
            self._step_start_event = torch.cuda.Event(enable_timing=True)
            self._step_end_event = torch.cuda.Event(enable_timing=True)
        else:
            self._step_start_event = None
            self._step_end_event = None
        self._last_step_ms = 0.0
        enable_all_to_all_monitor()

    def _log_param_summary(self) -> None:
        total_params = 0
        trainable_params = 0
        per_prefix: Dict[str, Dict[str, float]] = {}
        preview = []
        for name, param in self.model.named_parameters():
            count = param.numel()
            total_params += count
            if param.requires_grad:
                trainable_params += count
            prefix = name.split(".")[0]
            stats = per_prefix.setdefault(prefix, {"total": 0, "trainable": 0})
            stats["total"] += count
            if param.requires_grad:
                stats["trainable"] += count
            if len(preview) < 25:
                preview.append((name, count, param.requires_grad))

        def fmt(num: float) -> str:
            return f"{num/1e6:.3f}M"

        print(f"[Params] Total parameters: {total_params} ({fmt(total_params)})")
        print(f"[Params] Trainable parameters: {trainable_params} ({fmt(trainable_params)})")
        for prefix, stats in per_prefix.items():
            print(
                f"  - {prefix:<15} total={stats['total']} ({fmt(stats['total'])}), trainable={stats['trainable']} ({fmt(stats['trainable'])})"
            )
        print("[Params] Preview of first 25 parameter tensors:")
        for name, count, req_grad in preview:
            print(f"    {name}: numel={count}, requires_grad={req_grad}")

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

    def _apply_freeze(self, stage: str) -> None:
        stage_cfg = self.hyperparams.get(stage, {})
        freeze = stage_cfg.get("freeze", {})

        for p in self.model.parameters():
            p.requires_grad = True

        def freeze_all(module):
            for p in module.parameters():
                p.requires_grad = False

        def unfreeze_all(module):
            for p in module.parameters():
                p.requires_grad = True

        # LLM
        llm_rules = freeze.get("llm", [])
        if "all" in llm_rules:
            freeze_all(self.model.llm)
        else:
            layers = None
            if hasattr(self.model.llm, "model") and hasattr(self.model.llm.model, "layers"):
                layers = self.model.llm.model.layers
            elif hasattr(self.model.llm, "transformer") and hasattr(self.model.llm.transformer, "h"):
                layers = self.model.llm.transformer.h
            if layers is not None:
                for rule in llm_rules:
                    if rule.startswith("bottom_"):
                        n = int(rule.split("_")[1])
                        for li in range(min(n, len(layers))):
                            freeze_all(layers[li])
                    elif rule.startswith("top_"):
                        n = int(rule.split("_")[1])
                        for li in range(len(layers)-n, len(layers)):
                            if li >= 0:
                                freeze_all(layers[li])

        # Vision
        vis_rules = freeze.get("vision", [])
        if "all" in vis_rules:
            freeze_all(self.model.vision)
        for rule in vis_rules:
            if rule == "all_but_moe":
                for name, module in self.model.vision.named_modules():
                    if module.__class__.__name__ == "MoEFeedForward":
                        unfreeze_all(module)
                    else:
                        for p in module.parameters():
                            p.requires_grad = False
            elif rule == "first_half":
                enc = self.model.vision._get_encoder_layers()
                cutoff = len(enc)//2
                for i, blk in enumerate(enc):
                    if i < cutoff:
                        freeze_all(blk)

        # Projector
        proj_rules = freeze.get("projector", [])
        if "all" in proj_rules:
            freeze_all(self.model.projector)

        # Shared expert controls (global across MoE layers)
        shared_action = freeze.get("shared_expert", None)
        if shared_action is not None:
            for m in self.model.modules():
                if hasattr(m, "shared_expert") and isinstance(getattr(m, "shared_expert"), torch.nn.Module):
                    if shared_action == "freeze":
                        freeze_all(m.shared_expert)
                    elif shared_action == "unfreeze":
                        unfreeze_all(m.shared_expert)

    def _rebuild_engine(self, lr: float) -> None:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if hasattr(self, "model_engine"):
            del self.model_engine
        if hasattr(self, "optimizer"):
            del self.optimizer
        torch.cuda.empty_cache()
        g = self.hyperparams.get("global", {})
        optimizer = build_optimizer(
            self.model,
            base_lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=g.get("weight_decay", 0.1),
            moe_lr_mult=g.get("moe_lr_mult", 1.0),
            projector_lr_mult=g.get("projector_lr_mult", 1.0),
            optimizer_name=g.get("optimizer", "adamw"),
        )
        print("[Trainer] Rebuilding DeepSpeed engine for new stage/lr...")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config=self.deepspeed_cfg,
        )

    def _maybe_eval(self, stage: str, step: int, stage_dir: str) -> None:
        eval_every = int(self.hyperparams.get("curriculum", {}).get("eval_interval_steps", 0) or 0)
        if eval_every <= 0:
            return
        if step % eval_every != 0:
            return
        # Export a lightweight HF checkpoint and run OpenCompass adapter
        hf_dir = f"{stage_dir}/hf-step-{step}"
        save_hf_checkpoint(self.model_engine, self.tokenizer, hf_dir)
        if deepspeed.comm.get_rank() == 0:
            try:
                subprocess.run([
                    "bash",
                    "evaluation/run_opencompass.sh",
                    hf_dir,
                    "OmniMoE/configs/omni_moe_config.json",
                ], check=False)
            except FileNotFoundError:
                print("[Eval] OpenCompass runner not found; skip.")

    def train(self) -> None:
        global_step = 0
        for stage in self.curriculum_order:
            print(f"[Trainer] Preparing dataloader for stage '{stage}'...")
            dataloader = self._build_dataloader(stage)
            stage_params = self.hyperparams.get(stage, {})
            epochs = stage_params.get("epochs", 1)
            learning_rate = stage_params.get("learning_rate", 1e-5)
            # apply stage freezing and projector depth, then rebuild engine
            self._apply_freeze(stage)
            proj_layers = stage_params.get("projector_layers", None)
            if proj_layers is not None and hasattr(self.model, "set_projector_active_layers"):
                self.model.set_projector_active_layers(int(proj_layers))
            self._rebuild_engine(learning_rate)
            warmup_steps = stage_params.get("warmup_steps", 0)
            total_steps = stage_params.get("max_steps", len(dataloader) * epochs)
            scheduler = build_scheduler(self.optimizer, {"warmup_steps": warmup_steps}, total_steps)
            stage_dir = f"{self.output_dir}/{stage}"
            ensure_dir(stage_dir)

            # Router schedules (projector-only; DS MoE gates live inside kernels)
            temp_sched = stage_params.get("router_temperature", {})
            temp_start = float(temp_sched.get("start", 1.0))
            temp_end = float(temp_sched.get("end", 1.0))
            temp_steps = int(temp_sched.get("steps", total_steps)) if total_steps else 0
            jitter = float(stage_params.get("router_jitter_std", 0.0))
            if hasattr(self.model, "set_projector_router_jitter"):
                self.model.set_projector_router_jitter(jitter)

            for epoch in range(epochs):
                for batch_idx, batch in enumerate(dataloader):
                    batch = {k: v.to(self.model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    # Apply temperature schedule
                    if temp_steps > 0 and hasattr(self.model, "set_projector_router_temperature"):
                        t = min(global_step, temp_steps)
                        cur_temp = temp_start + (temp_end - temp_start) * (t / max(1, temp_steps))
                        self.model.set_projector_router_temperature(cur_temp)
                    if self._step_start_event is not None:
                        self._step_start_event.record()
                    try:
                        outputs = self.model_engine(**batch)
                    except Exception as e:
                        print(f"[Trainer][ERROR] Forward/backward failure at step {global_step}: {e}")
                        raise
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    scheduler.step()
                    if self._step_end_event is not None:
                        self._step_end_event.record()
                    global_step += 1
                    if global_step % self.hyperparams.get("global", {}).get("logging_steps", 50) == 0:
                        if deepspeed.comm.get_rank() == 0:
                            print(f"[Stage {stage}] step {global_step} loss={loss.item():.6f}")
                            # Log recent MoE aux losses summary
                            if self._step_start_event is not None and self._step_end_event is not None:
                                torch.cuda.synchronize()
                                self._last_step_ms = float(self._step_start_event.elapsed_time(self._step_end_event))
                            step_ms = self._last_step_ms
                            aux_text = aux_vis = 0.0
                            moe_ms = 0.0
                            drop_text = drop_vis = 0.0
                            for m in self.model.modules():
                                if hasattr(m, "_last_aux_loss") and getattr(m, "_last_aux_loss") is not None:
                                    sc = getattr(m, "_moe_scope", None)
                                    v = float(m._last_aux_loss.detach().cpu())
                                    if sc == "text": aux_text += v
                                    elif sc == "vision": aux_vis += v
                                # Optional: drop stats if exposed
                                if hasattr(m, "_last_drop_ratio") and getattr(m, "_last_drop_ratio") is not None:
                                    sc = getattr(m, "_moe_scope", None)
                                    d = float(getattr(m, "_last_drop_ratio"))
                                    if sc == "text": drop_text += d
                                    elif sc == "vision": drop_vis += d
                                if hasattr(m, "_last_moe_forward_ms") and getattr(m, "_last_moe_forward_ms") is not None:
                                    moe_ms += float(getattr(m, "_last_moe_forward_ms"))
                            aux_proj = float(self.model.projector.aux_loss().detach().cpu())
                            # Estimate projector router entropy
                            ent_vals = []
                            for lyr in self.model.projector.layers:
                                ent = getattr(lyr, "_last_gate_entropy", None)
                                if ent is not None:
                                    ent_vals.append(float(ent.detach().cpu()))
                            ent_avg = sum(ent_vals)/len(ent_vals) if ent_vals else 0.0
                            moe_ratio = (moe_ms / step_ms) if step_ms > 0 else 0.0
                            a2a_ms, a2a_calls = get_all_to_all_stats()
                            a2a_ratio = (a2a_ms / step_ms) if step_ms > 0 else 0.0
                            print(
                                "  aux(text)={:.6f} aux(vision)={:.6f} aux(projector)={:.6f} ent(projector)={:.6f} drop(text)={:.6f} drop(vision)={:.6f} moe_ms={:.3f} step_ms={:.3f} moe_ratio={:.3f} a2a_ms={:.3f} a2a_ratio={:.3f} a2a_calls={:.0f}".format(
                                    aux_text,
                                    aux_vis,
                                    aux_proj,
                                    ent_avg,
                                    drop_text,
                                    drop_vis,
                                    moe_ms,
                                    step_ms,
                                    moe_ratio,
                                    a2a_ms,
                                    a2a_ratio,
                                    a2a_calls,
                                )
                            )
                            # TensorBoard scalars
                            if self.tb_writer is not None:
                                self.tb_writer.add_scalar(f"{stage}/loss", float(loss.detach().cpu()), global_step)
                                self.tb_writer.add_scalar(f"{stage}/aux_text", aux_text, global_step)
                                self.tb_writer.add_scalar(f"{stage}/aux_vision", aux_vis, global_step)
                                self.tb_writer.add_scalar(f"{stage}/aux_projector", aux_proj, global_step)
                                self.tb_writer.add_scalar(f"{stage}/proj_entropy", ent_avg, global_step)
                                self.tb_writer.add_scalar(f"{stage}/drop_text", drop_text, global_step)
                                self.tb_writer.add_scalar(f"{stage}/drop_vision", drop_vis, global_step)
                                self.tb_writer.add_scalar(f"{stage}/moe_ms", moe_ms, global_step)
                                self.tb_writer.add_scalar(f"{stage}/step_ms", step_ms, global_step)
                                self.tb_writer.add_scalar(f"{stage}/moe_ratio", moe_ratio, global_step)
                                self.tb_writer.add_scalar(f"{stage}/a2a_ms", a2a_ms, global_step)
                                self.tb_writer.add_scalar(f"{stage}/a2a_ratio", a2a_ratio, global_step)
                                self.tb_writer.add_scalar(f"{stage}/a2a_calls", a2a_calls, global_step)
                    # Periodic evaluation
                    self._maybe_eval(stage, global_step, stage_dir)
                    if global_step % self.hyperparams.get("global", {}).get("save_steps", 1000) == 0:
                        if deepspeed.comm.get_rank() == 0:
                            ckpt_path = f"{stage_dir}/step-{global_step}"
                            ensure_dir(ckpt_path)
                            self.model_engine.save_checkpoint(ckpt_path)
                            # export HF snapshot too
                            save_hf_checkpoint(self.model_engine, self.tokenizer, f"{ckpt_path}-hf")
                    if global_step >= total_steps:
                        break
                if global_step >= total_steps:
                    break
            if deepspeed.comm.get_rank() == 0:
                final_path = f"{stage_dir}/final"
                ensure_dir(final_path)
                self.model_engine.save_checkpoint(final_path)
                save_hf_checkpoint(self.model_engine, self.tokenizer, f"{final_path}-hf")

        if deepspeed.comm.get_rank() == 0:
            final_dir = f"{self.output_dir}/final"
            ensure_dir(final_dir)
            self.model_engine.save_checkpoint(final_dir)
            save_hf_checkpoint(self.model_engine, self.tokenizer, f"{final_dir}-hf")
        # Close TB writer
        try:
            if getattr(self, "tb_writer", None) is not None:
                self.tb_writer.flush(); self.tb_writer.close()
        except Exception as e:
            print(f"[Trainer][WARN] Failed to close TensorBoard writer: {e}")


__all__ = ["CurriculumTrainer"]
