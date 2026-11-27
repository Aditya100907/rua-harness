#!/bin/bash
# Part N sweeps for one checkpoint (run on a Bouchet login node; each line is an eval array over TASKS).
#   bash policy/slurm/sweeps.sh /path/ckpt_0020000.pt [EPISODES]
# N1 point error 0,1,2,5,10 cm; N2 distractor point; N3 drop point / name / source; N4 renamed object.
# Respects the GPU cap by throttling each array to 1 running element (%1); the watcher's reserve covers them.
CKPT=$1; EPS=${2:-50}
REPO=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
RUN=$(basename $(dirname $CKPT)); STEP=$(basename $CKPT .pt | sed 's/ckpt_0*//')
TASKS=${TASKS:-"PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave"}
N=$(echo $TASKS | wc -w)
sub() {  # note eval-args
  CKPT=$CKPT TASKS="$TASKS" EPISODES=$EPS NOTE=${RUN}_${STEP}_$1 EVAL_ARGS="$2" \
    sbatch --parsable --job-name=sweep_${RUN}_$1 --array=0-$((N-1))%1 $REPO/policy/slurm/eval_array.sbatch
}
for e in 0 1 2 5 10; do echo "N1 err${e}cm $(sub err${e}cm "--point-error-cm $e")"; done
echo "N2 distractor $(sub distractor --distractor)"
for d in point name source; do echo "N3 drop_$d $(sub drop_$d "--drop $d")"; done
echo "N4 rename $(sub rename_thing '--rename thing')"
