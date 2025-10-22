#!/usr/bin/env bash
set -euo pipefail

NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -s)}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_ROOT=${CONFIG_ROOT:-$(pwd)/OmniMoE/configs}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/checkpoints/omni_stack_moe}

mkdir -p "${OUTPUT_DIR}"

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
