#!/bin/bash
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=untrained
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/untrained_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
python -m policy.make_untrained --tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave --out $PROJ/checkpoints/untrained_smoke
