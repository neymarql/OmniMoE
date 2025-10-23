#!/usr/bin/env bash
#SBATCH --job-name=OmniMoE-5stage
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=ai-gpu
#SBATCH --qos=normal
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
module load cuda/12.4
module load python/3.10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni-moe

# NCCL and EP grouping (keep EP=8 within a node)
export NCCL_SOCKET_IFNAME="^lo,docker0"
export NCCL_DEBUG=WARN
export NCCL_MIN_NCHANNELS=8
export NCCL_NET_GDR_LEVEL=2
export DEEPSPEED_MOE_GROUP=8
export PYTHONPATH=$(pwd):$PYTHONPATH

# Paths
CONFIG_ROOT=${CONFIG_ROOT:-OmniMoE/configs}
MODEL_CFG=${MODEL_CFG:-${CONFIG_ROOT}/omni_moe_config.json}
DATA_CFG=${DATA_CFG:-${CONFIG_ROOT}/dataset_config_5stage.json}
HPARAMS=${HPARAMS:-${CONFIG_ROOT}/train_hyperparams_5stage.yaml}
DS_CFG=${DS_CFG:-${CONFIG_ROOT}/deepspeed_zero2_config.json}
OUT_DIR=${OUT_DIR:-/mnt/checkpoints/omni_stack_moe_5stage}

# Seed override (Python code uses this to override YAML global.seed)
SEED=${SEED:-3407}

srun deepspeed \
  --num_nodes ${SLURM_NNODES} \
  --num_gpus 8 \
  OmniMoE/training/train.py \
  --model_config ${MODEL_CFG} \
  --dataset_config ${DATA_CFG} \
  --deepspeed_config ${DS_CFG} \
  --hyperparams ${HPARAMS} \
  --output_dir ${OUT_DIR} \
  --seed ${SEED}

