#!/bin/bash
# Part N sweeps in ONE array (one element per task), each element running every mode in turn.
#   CKPT=... TASKS="..." EPISODES=50 sbatch --array=0-2 policy/slurm/sweep_array.sbatch
# Modes: N1 point error 0,1,2,5,10 cm; N2 distractor; N3 drop point / name / source; N4 rename.
#SBATCH --partition=day
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --job-name=sweep
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/sweep_%A_%a.out
REPO=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
RUN=$(basename $(dirname $CKPT)); STEP=$(basename $CKPT .pt | sed 's/ckpt_0*//')
declare -a NAMES=(err0cm err1cm err2cm err5cm err10cm err20cm err40cm distractor distractor10pct distractor30pct drop_point drop_name drop_source rename_thing no_command blank_images name_swap)
declare -a ARGS=("--point-error-cm 0" "--point-error-cm 1" "--point-error-cm 2" "--point-error-cm 5" "--point-error-cm 10"
                 "--point-error-cm 20" "--point-error-cm 40" "--distractor" "--distractor-rate 0.1" "--distractor-rate 0.3"
                 "--drop point" "--drop name" "--drop source" "--rename thing" "--drop all" "--blank-images" "--name-swap")
# sentence checkpoints: only the controls apply (no points, no names to drop)
if python -c "import json,sys;sys.exit(0 if json.load(open('$(dirname $CKPT)/config.json')).get('format','sentence')=='sentence' else 1)"; then
  NAMES=(err0cm no_command blank_images); ARGS=("" "--drop all" "--blank-images")
fi
# resume: modes whose merged results already exist are skipped (so adding modes reruns only the new ones)
SKIP_DONE=${SKIP_DONE:-1}
for k in "${!NAMES[@]}"; do
  T=($TASKS); TK=${T[$SLURM_ARRAY_TASK_ID]}
  if [ "$SKIP_DONE" = 1 ] && ls $REPO/results/*_policy_only_${RUN}_${STEP}_${NAMES[$k]}_${TK}.csv >/dev/null 2>&1; then
    echo "== mode ${NAMES[$k]} already done, skipping"; continue
  fi
  echo "== mode ${NAMES[$k]}"
  NOTE=${RUN}_${STEP}_${NAMES[$k]} EVAL_ARGS="${ARGS[$k]}" bash $REPO/policy/slurm/eval_array.sbatch
done
