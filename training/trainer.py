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
        # Configure dispatcher profiling (global flag)
        prof_flag = bool(self.hyperparams.get("global", {}).get("dispatch_profile", False))
        self.model.set_dispatcher_profiling(prof_flag)
        if deepspeed.comm.get_rank() == 0:
            print(f"[Trainer] EP dispatcher profiling: {'enabled' if prof_flag else 'disabled'}")
        maybe_load_checkpoint(self.model_engine, resume_from)
        self._resume_meta = None
        if resume_from:
            meta_path = os.path.join(resume_from, "omni_meta.json")
            if os.path.isfile(meta_path):
                import json
                with open(meta_path, "r", encoding="utf-8") as fp:
                    self._resume_meta = json.load(fp)
        if deepspeed.comm.get_rank() == 0:
            self._log_param_summary()
        self.curriculum_order: List[str] = list(hyperparams.get("curriculum", {}).get("order", []))
        if not self.curriculum_order:
            self.curriculum_order = sorted(dataset_cfg.keys())
        # TensorBoard writer (rank 0 only)
        log_dir = os.path.join(self.output_dir, "runs")
        if deepspeed.comm.get_rank() == 0:
            self.tb_writer = SummaryWriter(log_dir=log_dir)
        else:
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

    def _build_dataloader(self, stage: str):
        stage_cfg = self.dataset_cfg[stage]
        dataset, sampler, stage_info = load_datasets_for_stage(
            stage_cfg,
            tokenizer=self.tokenizer,
            max_text_length=self.config.max_text_length,
            seed=self.hyperparams.get("global", {}).get("seed", 42),
            scan_missing=True,
        )
        per_gpu_batch = self.hyperparams.get("global", {}).get("train_batch_size_per_gpu", 2)
        dl = DataLoader(
            dataset,
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=sampler,
            collate_fn=multimodal_collate_fn,
        )
        return dl, stage_info

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
        # Print transparent summary of optimizer groups for this stage
        total_trainable = sum(int(p.requires_grad) * p.numel() for p in self.model.parameters())
        print(f"[Trainer][Rebuild] Trainable parameters: {total_trainable:,}")
        for gi, group in enumerate(optimizer.param_groups):
            group_params = group.get("params", [])
            n_params = sum(p.numel() for p in group_params)
            lr_val = group.get("lr", lr)
            print(f"  - group{gi}: params={n_params:,} lr={lr_val}")
        print("[Trainer] Rebuilding DeepSpeed engine for new stage/lr...")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config=self.deepspeed_cfg,
        )

    def rebuild_optimizer_on_stage_change(self, stage: str, learning_rate: float, total_steps: int):
        """Freeze/unfreeze per stage, rebuild optimizer param groups, and re-create scheduler.

        Must be called whenever entering a new stage to ensure param group
        membership matches the current freeze mask and LR strategy.
        """
        # 1) Apply stage freeze mask first
        self._apply_freeze(stage)
        # 2) Rebuild engine with fresh optimizer groups (based on requires_grad)
        self._rebuild_engine(learning_rate)
        # 3) Recreate scheduler for this stage
        warmup_steps = self.hyperparams.get(stage, {}).get("warmup_steps", 0)
        scheduler = build_scheduler(self.optimizer, {"warmup_steps": warmup_steps}, total_steps)
        # Re-apply dispatcher profiling flag on stage change
        prof_flag = bool(self.hyperparams.get("global", {}).get("dispatch_profile", False))
        self.model.set_dispatcher_profiling(prof_flag)
        if deepspeed.comm.get_rank() == 0:
            print(f"[Trainer][Stage {stage}] EP dispatcher profiling: {'enabled' if prof_flag else 'disabled'}")
        return scheduler

    # -------------------- Checkpoint Metadata Utilities --------------------
    def _optimizer_fingerprint(self) -> list[dict]:
        fp = []
        for gi, g in enumerate(self.optimizer.param_groups):
            params = g.get("params", [])
            n = sum(p.numel() for p in params)
            lr = float(g.get("lr", 0.0))
            fp.append({"group": gi, "numel": int(n), "lr": lr})
        return fp

    def _router_state(self) -> dict:
        state = {"projector": {}, "llm": {}, "vision": {}}
        # Projector
        proj = getattr(self.model, "projector", None)
        if proj is not None:
            state["projector"]["temperature"] = float(getattr(proj, "_temperature", 1.0))
            state["projector"]["jitter"] = float(getattr(proj, "_jitter_std", 0.0))
        # LLM MoE: gather avg across MoEFeedForward modules
        import types
        temps, jits = [], []
        for m in self.model.modules():
            if getattr(m, "__class__", types.SimpleNamespace()).__name__ == "MoEFeedForward":
                if hasattr(m, "_moe_scope") and getattr(m, "_moe_scope") == "text":
                    temps.append(float(getattr(m, "_router_temperature", 1.0)))
                    jits.append(float(getattr(m, "_router_noise_std", 0.0)))
        if temps:
            state["llm"]["temperature"] = sum(temps) / len(temps)
            state["llm"]["jitter"] = sum(jits) / len(jits) if jits else 0.0
        # Vision MoE
        temps, jits = [], []
        for m in self.model.modules():
            if getattr(m, "__class__", types.SimpleNamespace()).__name__ == "MoEFeedForward":
                if hasattr(m, "_moe_scope") and getattr(m, "_moe_scope") == "vision":
                    temps.append(float(getattr(m, "_router_temperature", 1.0)))
                    jits.append(float(getattr(m, "_router_noise_std", 0.0)))
        if temps:
            state["vision"]["temperature"] = sum(temps) / len(temps)
            state["vision"]["jitter"] = sum(jits) / len(jits) if jits else 0.0
        return state

    def _ep_digest(self) -> dict:
        d = {"world_size": 1, "rank": 0}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            d["world_size"] = torch.distributed.get_world_size()
            d["rank"] = torch.distributed.get_rank()
        # EP sizes from config
        d["ep_text"] = int(self.config.text_moe_cfg.get("ep_size", 1))
        d["ep_vision"] = int(self.config.vision_moe_cfg.get("ep_size", 1))
        d["ep_projector"] = int(self.config.projector_cfg.get("ep_size", 1))
        # Rank group indices (if applicable)
        def grp(ep):
            if ep <= 1 or d["world_size"] <= 1:
                return -1
            return d["rank"] // ep
        d["group_text"] = grp(d["ep_text"])
        d["group_vision"] = grp(d["ep_vision"])
        d["group_projector"] = grp(d["ep_projector"])
        return d

    def _collect_metadata_snapshot(self, stage: str, global_step: int) -> dict:
        return {
            "stage_name": stage,
            "global_step": int(global_step),
            "optimizer": self._optimizer_fingerprint(),
            "router": self._router_state(),
            "ep_digest": self._ep_digest(),
        }

    def _write_meta(self, ckpt_dir: str, meta: dict) -> None:
        import json
        path = os.path.join(ckpt_dir, "omni_meta.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, ensure_ascii=False)

    def _verify_resume_consistency(self, stage: str, global_step: int) -> None:
        if not getattr(self, "_resume_meta", None):
            return
        cur = self._collect_metadata_snapshot(stage, global_step)
        prev = self._resume_meta
        print(f"[Resume] Verifying checkpoint metadata vs current stage '{stage}'...")
        # Stage name
        if prev.get("stage_name") != stage:
            print(f"  - Stage mismatch: ckpt={prev.get('stage_name')} current={stage}")
        # Optimizer groups
        p_opt = prev.get("optimizer", [])
        c_opt = cur.get("optimizer", [])
        if len(p_opt) != len(c_opt):
            print(f"  - Optimizer group count differs: ckpt={len(p_opt)} current={len(c_opt)}")
        for i, (pg, cg) in enumerate(zip(p_opt, c_opt)):
            if int(pg.get("numel", -1)) != int(cg.get("numel", -1)) or float(pg.get("lr", -1.0)) != float(cg.get("lr", -1.0)):
                print(f"  - Group{i} diff: ckpt(numel={pg.get('numel')}, lr={pg.get('lr')}) vs current(numel={cg.get('numel')}, lr={cg.get('lr')})")
        # Router
        for branch in ("projector", "llm", "vision"):
            p_b = prev.get("router", {}).get(branch, {})
            c_b = cur.get("router", {}).get(branch, {})
            if p_b != c_b:
                print(f"  - Router {branch} diff: ckpt={p_b} current={c_b}")
        # EP digest
        if prev.get("ep_digest") != cur.get("ep_digest"):
            print(f"  - EP digest diff: ckpt={prev.get('ep_digest')} current={cur.get('ep_digest')}")

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
            subprocess.run([
                "bash",
                "evaluation/run_opencompass.sh",
                hf_dir,
                "OmniMoE/configs/omni_moe_config.json",
            ], check=False)

    def train(self) -> None:
        global_step = 0
        for stage in self.curriculum_order:
            print(f"[Trainer] Preparing dataloader for stage '{stage}'...")
            dataloader, stage_info = self._build_dataloader(stage)
            # Quick data mix summary at stage start
            if stage_info and "groups" in stage_info:
                print(f"[Trainer][Stage {stage}] Data mix summary (epoch_length={stage_info.get('epoch_length')}, total_planned={stage_info.get('total_planned')}):")
                for item in stage_info["groups"]:
                    print(
                        "  - group={group_index} len={length} rate={sample_rate} planned={planned_draws} ({planned_percent:.2f}%) missing_images={missing_images} folder={image_folder}".format(
                            **item
                        )
                    )
            stage_params = self.hyperparams.get(stage, {})
            epochs = stage_params.get("epochs", 1)
            learning_rate = stage_params.get("learning_rate", 1e-5)
            # apply stage projector depth selection
            proj_layers = stage_params.get("projector_layers", None)
            if proj_layers is not None and hasattr(self.model, "set_projector_active_layers"):
                self.model.set_projector_active_layers(int(proj_layers))
            warmup_steps = stage_params.get("warmup_steps", 0)
            total_steps = stage_params.get("max_steps", len(dataloader) * epochs)
            scheduler = self.rebuild_optimizer_on_stage_change(stage, learning_rate, total_steps)
            stage_dir = f"{self.output_dir}/{stage}"
            ensure_dir(stage_dir)

            # Router schedules (projector-only; DS MoE gates live inside kernels)
            # Router temperature schedules (branch-specific with fallback)
            temp_common = stage_params.get("router_temperature", {})
            temp_proj = stage_params.get("router_temperature_projector", temp_common)
            temp_llm = stage_params.get("router_temperature_llm", temp_common)
            temp_vis = stage_params.get("router_temperature_vision", temp_common)

            def _sched_val(sched: Dict, step: int, default_steps: int) -> float:
                if not isinstance(sched, dict):
                    return float(sched)
                start = float(sched.get("start", 1.0))
                end = float(sched.get("end", start))
                steps = int(sched.get("steps", default_steps)) if default_steps else int(sched.get("steps", 0))
                if steps <= 0:
                    return end
                t = min(step, steps)
                return start + (end - start) * (t / max(1, steps))

            # Jitter schedules or constants (branch-specific with fallback)
            jitter_common = stage_params.get("router_jitter_std", 0.0)
            jitter_proj = stage_params.get("router_jitter_projector", jitter_common)
            jitter_llm = stage_params.get("router_jitter_llm", jitter_common)
            jitter_vis = stage_params.get("router_jitter_vision", jitter_common)

            # Apply initial jitter for each branch
            jp0 = _sched_val(jitter_proj, 0, total_steps)
            jl0 = _sched_val(jitter_llm, 0, total_steps)
            jv0 = _sched_val(jitter_vis, 0, total_steps)
            if hasattr(self.model, "set_projector_router_jitter"):
                self.model.set_projector_router_jitter(jp0)
            if hasattr(self.model, "set_llm_router_jitter"):
                self.model.set_llm_router_jitter(jl0)
            if hasattr(self.model, "set_vision_router_jitter"):
                self.model.set_vision_router_jitter(jv0)

            # Initialize router temps at step 0 for consistency with resume checks
            # (same schedule function used below per-step)
            # temp_common/proj/llm/vis defined above
            cur_tp0 = _sched_val(temp_proj, 0, total_steps)
            cur_tl0 = _sched_val(temp_llm, 0, total_steps)
            cur_tv0 = _sched_val(temp_vis, 0, total_steps)
            if hasattr(self.model, "set_projector_router_temperature"):
                self.model.set_projector_router_temperature(cur_tp0)
            if hasattr(self.model, "set_llm_router_temperature"):
                self.model.set_llm_router_temperature(cur_tl0)
            if hasattr(self.model, "set_vision_router_temperature"):
                self.model.set_vision_router_temperature(cur_tv0)

            # Verify resume consistency now that optimizer and router are set
            self._verify_resume_consistency(stage, global_step)

            for epoch in range(epochs):
                for batch_idx, batch in enumerate(dataloader):
                    batch = {k: v.to(self.model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    # Apply temperature schedule
                    # Update temperatures per-branch
                    cur_tp = _sched_val(temp_proj, global_step, total_steps)
                    cur_tl = _sched_val(temp_llm, global_step, total_steps)
                    cur_tv = _sched_val(temp_vis, global_step, total_steps)
                    if hasattr(self.model, "set_projector_router_temperature"):
                        self.model.set_projector_router_temperature(cur_tp)
                    if hasattr(self.model, "set_llm_router_temperature"):
                        self.model.set_llm_router_temperature(cur_tl)
                    if hasattr(self.model, "set_vision_router_temperature"):
                        self.model.set_vision_router_temperature(cur_tv)
                    # Update jitter per-branch if schedules are dicts
                    if isinstance(jitter_proj, dict) and hasattr(self.model, "set_projector_router_jitter"):
                        self.model.set_projector_router_jitter(_sched_val(jitter_proj, global_step, total_steps))
                    if isinstance(jitter_llm, dict) and hasattr(self.model, "set_llm_router_jitter"):
                        self.model.set_llm_router_jitter(_sched_val(jitter_llm, global_step, total_steps))
                    if isinstance(jitter_vis, dict) and hasattr(self.model, "set_vision_router_jitter"):
                        self.model.set_vision_router_jitter(_sched_val(jitter_vis, global_step, total_steps))
                    if self._step_start_event is not None:
                        self._step_start_event.record()
                    outputs = self.model_engine(**batch)
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
                            # branch router temps/jitter
                            proj_T = getattr(self.model.projector, "_temperature", 1.0)
                            proj_J = getattr(self.model.projector, "_jitter_std", 0.0)
                            T_text = []; J_text = []; T_vis = []; J_vis = []
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
                                # Collect temps/jitter from MoE blocks
                                if getattr(m, "__class__", None) and m.__class__.__name__ == "MoEFeedForward":
                                    sc = getattr(m, "_moe_scope", None)
                                    t = float(getattr(m, "_router_temperature", 1.0))
                                    j = float(getattr(m, "_router_noise_std", 0.0))
                                    if sc == "text":
                                        T_text.append(t); J_text.append(j)
                                    elif sc == "vision":
                                        T_vis.append(t); J_vis.append(j)
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
                            # Aggregate EP dispatcher micro-timings
                            ep_tok = ep_meta = ep_w = ep_local = 0.0
                            for mod in self.model.modules():
                                disp = None
                                if hasattr(mod, "dispatcher"):
                                    disp = getattr(mod, "dispatcher")
                                if hasattr(mod, "_ec_dispatcher") and disp is None:
                                    disp = getattr(mod, "_ec_dispatcher")
                                if disp is not None and getattr(disp, "profile_enabled", False):
                                    ep_tok += float(getattr(disp, "last_ms_tokens", 0.0))
                                    ep_meta += float(getattr(disp, "last_ms_meta", 0.0))
                                    ep_w += float(getattr(disp, "last_ms_weights", 0.0))
                                    ep_local += float(getattr(disp, "last_ms_local", 0.0))
                            # Compute avg temps/jitter
                            Tt = sum(T_text)/len(T_text) if T_text else 0.0
                            Tz = sum(T_vis)/len(T_vis) if T_vis else 0.0
                            Jt = sum(J_text)/len(J_text) if J_text else 0.0
                            Jz = sum(J_vis)/len(J_vis) if J_vis else 0.0
                            print(
                                "  aux(text)={:.6f} aux(vision)={:.6f} aux(projector)={:.6f} ent(projector)={:.6f} drop(text)={:.6f} drop(vision)={:.6f} moe_ms={:.3f} step_ms={:.3f} moe_ratio={:.3f} a2a_ms={:.3f} a2a_ratio={:.3f} a2a_calls={:.0f} EP(tok/ms)={:.3f} EP(meta/ms)={:.3f} EP(w/ms)={:.3f} EP(local/ms)={:.3f} Tproj={:.3f} Jproj={:.3f} Ttext={:.3f} Jtext={:.3f} Tvis={:.3f} Jvis={:.3f}".format(
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
                                    ep_tok, ep_meta, ep_w, ep_local,
                                    float(proj_T), float(proj_J), Tt, Jt, Tz, Jz,
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
                                self.tb_writer.add_scalar(f"{stage}/ep_tok_ms", ep_tok, global_step)
                                self.tb_writer.add_scalar(f"{stage}/ep_meta_ms", ep_meta, global_step)
                                self.tb_writer.add_scalar(f"{stage}/ep_w_ms", ep_w, global_step)
                                self.tb_writer.add_scalar(f"{stage}/ep_local_ms", ep_local, global_step)
                                self.tb_writer.add_scalar(f"{stage}/router_T_projector", float(proj_T), global_step)
                                self.tb_writer.add_scalar(f"{stage}/router_J_projector", float(proj_J), global_step)
                                self.tb_writer.add_scalar(f"{stage}/router_T_text", Tt, global_step)
                                self.tb_writer.add_scalar(f"{stage}/router_J_text", Jt, global_step)
                                self.tb_writer.add_scalar(f"{stage}/router_T_vision", Tz, global_step)
                                self.tb_writer.add_scalar(f"{stage}/router_J_vision", Jz, global_step)
                    # Periodic evaluation
                    self._maybe_eval(stage, global_step, stage_dir)
                    if global_step % self.hyperparams.get("global", {}).get("save_steps", 1000) == 0:
                        if deepspeed.comm.get_rank() == 0:
                            ckpt_path = f"{stage_dir}/step-{global_step}"
                            ensure_dir(ckpt_path)
                            self.model_engine.save_checkpoint(ckpt_path)
                            # export HF snapshot too
                            save_hf_checkpoint(self.model_engine, self.tokenizer, f"{ckpt_path}-hf")
                            # write metadata sidecar
                            meta = self._collect_metadata_snapshot(stage, global_step)
                            self._write_meta(ckpt_path, meta)
                    if global_step >= total_steps:
                        break
                if global_step >= total_steps:
                    break
            if deepspeed.comm.get_rank() == 0:
                final_path = f"{stage_dir}/final"
                ensure_dir(final_path)
                self.model_engine.save_checkpoint(final_path)
                save_hf_checkpoint(self.model_engine, self.tokenizer, f"{final_path}-hf")
                meta = self._collect_metadata_snapshot(stage, global_step)
                self._write_meta(final_path, meta)

        if deepspeed.comm.get_rank() == 0:
            final_dir = f"{self.output_dir}/final"
            ensure_dir(final_dir)
            self.model_engine.save_checkpoint(final_dir)
            save_hf_checkpoint(self.model_engine, self.tokenizer, f"{final_dir}-hf")
            meta = self._collect_metadata_snapshot("final", global_step)
            self._write_meta(final_dir, meta)
        # Close TB writer
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.flush(); self.tb_writer.close()


__all__ = ["CurriculumTrainer"]
