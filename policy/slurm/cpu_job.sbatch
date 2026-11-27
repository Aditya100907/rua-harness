#!/bin/bash
# Generic CPU job on the day partition: CMD="python -m ..." sbatch policy/slurm/cpu_job.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=cpu
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/cpu_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
echo "$CMD"; eval "$CMD"
