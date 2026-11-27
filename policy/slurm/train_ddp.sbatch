#!/bin/bash
# Multi-GPU training: torchrun, gradients averaged across ranks (policy/train.py).
#   RUN=x ARGS="..." sbatch --partition=gpu_h100 --gpus=4 --cpus-per-task=24 policy/slurm/train_ddp.sbatch
# --batch in ARGS is per GPU; the global batch is batch x GPUs.
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --time=06:00:00
#SBATCH --job-name=train_ddp
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/train_%x_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
OUT=$PROJ/checkpoints/$RUN
mkdir -p $OUT; echo "$ARGS" > $OUT/args.txt
NG=$(nvidia-smi -L | wc -l)
echo "job $SLURM_JOB_ID run $RUN on $(hostname) $NG GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) args: $ARGS"
torchrun --standalone --nproc_per_node=$NG -m policy.train $ARGS --out $OUT --resume --workers ${WORKERS:-5} 2>&1 | tee $OUT/last_job.log
