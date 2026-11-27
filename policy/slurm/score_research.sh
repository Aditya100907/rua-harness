#!/bin/bash
# Score a converted research checkpoint (policy/research_ckpt.py) on the 3 smoke tasks, 50 episodes, CPU nodes.
#   bash policy/slurm/score_research.sh /path/to/converted/ckpt_0000000.pt [EPISODES]
CKPT=$1; EPS=${2:-50}
REPO=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
CKPT=$CKPT TASKS="PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave" EPISODES=$EPS NOTE=research_$(basename $(dirname $CKPT)) \
  sbatch --parsable --job-name=score_research --array=0-2 $REPO/policy/slurm/eval_array.sbatch
