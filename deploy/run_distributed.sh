#!/usr/bin/env bash
set -euo pipefail

NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_ROOT=${CONFIG_ROOT:-$(pwd)/OmniMoE/configs}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/checkpoints/omni_stack_moe}

mkdir -p "${OUTPUT_DIR}"

# Encourage Expert-Parallel grouping per node (8 GPUs)
export DEEPSPEED_MOE_GROUP=${DEEPSPEED_MOE_GROUP:-8}
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2}

deepspeed --num_nodes "${NNODES}" \
  --num_gpus "${GPUS_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  OmniMoE/training/train.py \
  --model_config "${CONFIG_ROOT}/omni_moe_config.json" \
  --dataset_config "${CONFIG_ROOT}/dataset_config.json" \
  --deepspeed_config "${CONFIG_ROOT}/deepspeed_zero2_config.json" \
  --hyperparams "${CONFIG_ROOT}/train_hyperparams.yaml" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
