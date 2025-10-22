#!/usr/bin/env bash
set -euo pipefail

# Simple vLLM EP server launcher for OmniMoE HF checkpoints.
# Example:
#   bash OmniMoE/inference/vllm/launch_expert_parallel.sh \
#     /mnt/checkpoints/omni_stack_moe/final-hf 2 1 8 0.0.0.0:8000

MODEL_DIR=${1:-}
TP=${2:-2}
PP=${3:-1}
EP_ENABLE=${4:-8}
BIND=${5:-0.0.0.0:8000}

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Usage: $0 MODEL_DIR [TP] [PP] [EP_SIZE] [HOST:PORT]" >&2
  exit 1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --host "${BIND%%:*}" --port "${BIND##*:}"
