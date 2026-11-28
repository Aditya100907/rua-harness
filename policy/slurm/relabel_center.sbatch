#!/bin/bash
# Relabel training points with the test-time perfect command (policy/relabel_center.py); one task chunk per index.
#   TASKS="A B C" sbatch --array=0-17 policy/slurm/relabel_center.sbatch      (6 chunks of 100 per task)
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=relabel
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/relabel_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
T=(${TASKS:-$ALL_TASKS}); N=${NCHUNK:-6}
TASK=${T[$((SLURM_ARRAY_TASK_ID / N))]}; C=$((SLURM_ARRAY_TASK_ID % N))
python -m policy.relabel_center --task $TASK --chunk $C --chunk-size 100 2>&1 | grep -v WARN
