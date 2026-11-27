#!/bin/bash
# H1 timing on one GPU.  CKPT=... sbatch --partition=gpu --constraint=gpu:rtx_5000_ada policy/slurm/timing.sbatch
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=timing
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/timing_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
nvidia-smi --query-gpu=name --format=csv,noheader
python -m policy.timing --ckpt $CKPT --device cuda --n 300
python -m policy.timing --ckpt $CKPT --device cuda --n 300 --bf16
