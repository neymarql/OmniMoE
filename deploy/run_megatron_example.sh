#!/usr/bin/env bash
set -euo pipefail

# Example launcher for Megatron-Core GPT-MoE training (LLM branch, text-only).
# Adjust TP/PP/EP and nodes to your cluster. Requires megatron-core>=0.14.

NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29500}

export MASTER_ADDR MASTER_PORT

torchrun --nproc_per_node ${NPROC_PER_NODE} --nnodes ${NNODES} \
  OmniMoE/training/megatron_train.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --num-experts 16 \
  --moe-router-topk 2 \
  --sequence-parallel \
  --use-flash-attn-3 \
  --train-iters 10000 \
  --micro-batch-size 2 \
  --global-batch-size 32 \
  --lr 2e-4 \
  --save /mnt/checkpoints/omni_megatron
