#!/bin/bash
# CPU job array: pre-extract frames, one task per array index (0..17).
#   sbatch --array=0-17 policy/slurm/extract_frames.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=extract
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/extract_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
TASKS=($ALL_TASKS)
python -m policy.extract_frames --tasks ${TASKS[$SLURM_ARRAY_TASK_ID]} --cams left right wrist
