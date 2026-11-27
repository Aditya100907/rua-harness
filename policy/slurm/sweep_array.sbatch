#!/bin/bash
# Part N sweeps in ONE array (one element per task), each element running every mode in turn.
#   CKPT=... TASKS="..." EPISODES=50 sbatch --array=0-2 policy/slurm/sweep_array.sbatch
# Modes: N1 point error 0,1,2,5,10 cm; N2 distractor; N3 drop point / name / source; N4 rename.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=sweep
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/sweep_%A_%a.out
REPO=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
RUN=$(basename $(dirname $CKPT)); STEP=$(basename $CKPT .pt | sed 's/ckpt_0*//')
declare -a NAMES=(err0cm err1cm err2cm err5cm err10cm distractor drop_point drop_name drop_source rename_thing)
declare -a ARGS=("--point-error-cm 0" "--point-error-cm 1" "--point-error-cm 2" "--point-error-cm 5" "--point-error-cm 10"
                 "--distractor" "--drop point" "--drop name" "--drop source" "--rename thing")
for k in "${!NAMES[@]}"; do
  echo "== mode ${NAMES[$k]}"
  NOTE=${RUN}_${STEP}_${NAMES[$k]} EVAL_ARGS="${ARGS[$k]}" bash $REPO/policy/slurm/eval_array.sbatch
done
