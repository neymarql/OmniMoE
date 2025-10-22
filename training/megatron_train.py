"""Megatron-Core training entry for OmniMoE (LLM branch) with SP + FA-3.

This entry trains a GPT + MoE model using Megatron-Core's kernels. It is
intended for pretraining/fine-tuning the LLM branch with sequence-parallel
and FlashAttention-3 on Hopper. The multimodal path (SigLIP + projector)
remains trained via the main DeepSpeed/HF entry. This script focuses on
robust LLM MoE training at maximum efficiency.

Requirements:
  - megatron-core >= 0.14
  - FlashAttention-3 kernels available (optional)
    * enabled by default; pass `--disable-flash-attn-3` to fall back.

Usage (example):
  torchrun --nproc_per_node 8 --nnodes 2 --node_rank $RANK \
    OmniMoE/training/megatron_train.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 8 \
    --num-experts 16 \
    --moe-router-topk 2 \
    --sequence-parallel \
    --train-iters 10000 \
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --lr 2e-4
"""
from __future__ import annotations

import argparse
import os
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from OmniMoE.training.megatron_data import JsonTextOnlyDataset, collate_text_only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Megatron-Core GPT-MoE Trainer")
    p.add_argument("--tensor-model-parallel-size", type=int, default=2)
    p.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    p.add_argument("--expert-model-parallel-size", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--moe-router-topk", type=int, default=2)
    p.add_argument("--sequence-parallel", action="store_true")
    p.add_argument("--disable-flash-attn-3", action="store_true")
    p.add_argument("--train-iters", type=int, default=1000)
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--global-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save", type=str, default="/mnt/checkpoints/omni_megatron")
    p.add_argument("--load", type=str, default="")
    p.add_argument("--json_path", type=str, default="")
    p.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-8B")
    return p.parse_args()


def main() -> None:
    try:
        import megatron.core as mcore  # type: ignore
        from megatron.core import parallel_state as mps  # type: ignore
        from megatron.core.models.gpt.gpt_model import GPTModel  # type: ignore
        from megatron.core.transformer.moe.moe_layer import MoEConfig  # type: ignore
    except Exception as e:
        print("[Megatron][ERROR] megatron-core not available. Please install megatron-core>=0.14.", e, file=sys.stderr)
        raise

    args = parse_args()
    args.use_flash_attn_3 = not args.disable_flash_attn_3

    # Initialize distributed and model-parallelism
    mcore.initialize.initialize_megatron(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        sequence_parallel=args.sequence_parallel,
    )

    # FlashAttention-3 toggle (implementation may vary by version)
    if args.use_flash_attn_3:
        os.environ["FLASH_ATTENTION_VERSION"] = "3"

    # Configure MoE
    moe_cfg = MoEConfig(
        num_experts=args.num_experts,
        top_k=args.moe_router_topk,
        enable_expert_tensor_parallelism=False,
        use_grouped_gemm=True,
    )

    # Minimal GPT config (adjust as needed)
    hidden_size = 4096
    num_layers = 32
    num_attention_heads = 32

    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        vocab_size=32000,
        max_sequence_length=4096,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        apply_residual_connection_post_layernorm=False,
        bias_dropout_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cpu_initialization=False,
        moe_config=moe_cfg,
    )

    model.cuda()
    model.train()

    # Simple optimizer / loop scaffold
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    # Dataset: text-only adapter for our JSON format (optional)
    writer = None
    try:
        if mps.get_data_parallel_rank() == 0:
            writer = SummaryWriter(log_dir=os.path.join(args.save, "runs"))
    except Exception as e:
        print("[Megatron][WARN] TensorBoard unavailable:", e)

    if args.json_path:
        try:
            ds = JsonTextOnlyDataset(args.json_path, tokenizer_name=args.tokenizer_name, max_length=4096)
            dl = DataLoader(ds, batch_size=args.micro_batch_size, shuffle=True, num_workers=2, collate_fn=collate_text_only)
            data_iter = iter(dl)
            print(f"[Megatron] Using JSON dataset: {args.json_path}")
        except Exception as e:
            print(f"[Megatron][ERROR] Failed to build dataset from {args.json_path}:", e)
            raise
    else:
        print("[Megatron] No json_path provided; falling back to dummy random tokens.")
        data_iter = None
    seq_len = 1024
    global_rank = mps.get_data_parallel_rank()
    for step in range(args.train_iters):
        if data_iter is not None:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            tokens = batch["input_ids"].cuda()
            attn = batch["attention_mask"].cuda()
        else:
            tokens = torch.randint(0, 32000, (args.micro_batch_size, seq_len), device="cuda")
            attn = torch.ones_like(tokens)
        try:
            loss = model(tokens, attention_mask=attn)[0]
        except Exception as e:
            print(f"[Megatron][ERROR] Forward failure at step {step}:", e)
            raise
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 50 == 0 and global_rank == 0:
            val = float(loss.detach().cpu())
            print(f"[Megatron] step {step} loss={val:.6f}")
            try:
                if writer is not None:
                    writer.add_scalar("train/loss", val, step)
            except Exception as e:
                print("[Megatron][WARN] TB log failed:", e)

    if global_rank == 0:
        os.makedirs(args.save, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save, "gpt_moe_megatron.pt"))
        try:
            if writer is not None:
                writer.flush(); writer.close()
        except Exception as e:
            print("[Megatron][WARN] TB close failed:", e)


if __name__ == "__main__":
    main()
