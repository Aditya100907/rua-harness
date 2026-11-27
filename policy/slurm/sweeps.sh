#!/bin/bash
# Part N sweeps for one checkpoint: one array, one element per task (policy/slurm/sweep_array.sbatch).
#   bash policy/slurm/sweeps.sh /path/ckpt_0020000.pt [EPISODES]
CKPT=$1; EPS=${2:-50}
REPO=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
TASKS=${TASKS:-"PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave"}
N=$(echo $TASKS | wc -w)
CKPT=$CKPT TASKS="$TASKS" EPISODES=$EPS sbatch --parsable --job-name=sweep_$(basename $(dirname $CKPT)) \
  --array=0-$((N-1)) $REPO/policy/slurm/sweep_array.sbatch
