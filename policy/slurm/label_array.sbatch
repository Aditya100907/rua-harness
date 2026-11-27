#!/bin/bash
# F1: hindsight command labels for all 18 tasks, one task per array index, with a 20-demo simulator check.
#   sbatch --array=0-17 policy/slurm/label_array.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=label
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/label_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
TASKS=($ALL_TASKS)
python -m policy.label_commands --tasks ${TASKS[$SLURM_ARRAY_TASK_ID]} --verify ${VERIFY:-20} --out $SCR/labels 2>&1 | grep -v WARN
