#!/bin/bash
# The watcher (Part M2): 7 days on the agent partition, 1 CPU, 8 GB.
#SBATCH --partition=agent
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=watcher
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/watcher_%j.out
bash /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/watcher.sh
