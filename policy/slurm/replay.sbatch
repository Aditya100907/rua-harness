#!/bin/bash
# Replay test on Bouchet (CPU, no rendering needed).  sbatch policy/slurm/replay.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=replay
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/replay_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
python -m policy.replay_test --tasks ${TASKS:-PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave} --n ${N:-10} \
  --out $ROOT/runs/replay/bouchet_${SLURM_JOB_ID}.csv 2>&1 | grep -v WARNING
