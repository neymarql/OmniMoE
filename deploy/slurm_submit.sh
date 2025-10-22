#!/usr/bin/env bash
#SBATCH --job-name=OmniMoE
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=ai-gpu
#SBATCH --qos=normal
#SBATCH --output=logs/%x-%j.out

module purge
module load cuda/12.4
module load python/3.10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omni-moe

export NCCL_SOCKET_IFNAME="^lo,docker0"
export NCCL_DEBUG=WARN
export NCCL_MIN_NCHANNELS=8
export NCCL_NET_GDR_LEVEL=2
export PYTHONPATH=$(pwd):$PYTHONPATH

srun deepspeed \
  --num_nodes ${SLURM_NNODES} \
  --num_gpus 8 \
  OmniMoE/training/train.py \
  --model_config OmniMoE/configs/omni_moe_config.json \
  --dataset_config OmniMoE/configs/dataset_config.json \
  --deepspeed_config OmniMoE/configs/deepspeed_zero2_config.json \
  --hyperparams OmniMoE/configs/train_hyperparams.yaml \
  --output_dir /mnt/checkpoints/omni_stack_moe
