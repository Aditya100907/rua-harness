#!/bin/bash
# LIBERO training (stock DEM, sentence format) in envs/dem: same as train.sbatch but runs policy.libero.train.
#   RUN=libero_spatial_dem ARGS="--suites libero_spatial --steps 30000" sbatch policy/slurm/libero_train.sbatch
# The job stops itself before the time limit (--max-hours) and resubmits itself
# with the same RUN and ARGS until train_state.pt reaches --steps.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=libero_train
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/train_%x_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
export LIBERO_DATA=$SCR/datasets/libero
OUT=$PROJ/checkpoints/$RUN
mkdir -p $OUT
echo "$ARGS" > $OUT/args.txt
echo "job $SLURM_JOB_ID run $RUN on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) args: $ARGS"
echo "$SLURM_JOB_ID $(date +%F_%T) $SLURM_JOB_PARTITION" >> $OUT/jobs.txt
LIMIT_H=$(squeue -j $SLURM_JOB_ID -h -o %l | awk -F'[-:]' '{ if (NF==4) print $1*24+$2; else if (NF==3) print $1; else print 1 }')
MAXH=$(python -c "print(max(0.3, $LIMIT_H - 0.6))")
python -m policy.libero.train $ARGS --out $OUT --resume --max-hours $MAXH --workers ${WORKERS:-5} 2>&1 | tee $OUT/last_job.log
RC=${PIPESTATUS[0]}
STEP=$(python -c "import torch;print(torch.load('$OUT/train_state.pt',weights_only=False)['step'])" 2>/dev/null || echo 0)
WANT=$(python -c "import sys,re;m=re.search(r'--steps\s+(\d+)', '''$ARGS''');print(m.group(1) if m else 20000)")
echo "finished at step $STEP of $WANT"
# resubmit only after a clean time-limit stop; a crash would otherwise loop forever
if [ "$RC" -ne 0 ] || ! grep -q "TRAIN STOPPED" $OUT/last_job.log; then
  [ "$STEP" -lt "$WANT" ] && echo "not resubmitting: exit code $RC or no clean stop (see $OUT/last_job.log)" && touch $OUT/CRASHED
elif [ "$STEP" -lt "$WANT" ] && [ -z "${NO_RESUBMIT:-}" ] && [ ! -f $OUT/STOP ]; then
  echo "resubmitting"
  RUN=$RUN ARGS="$ARGS" WORKERS=${WORKERS:-5} PARTS=${PARTS:-} CONSTRAINT=${CONSTRAINT:-} \
    sbatch --job-name=$SLURM_JOB_NAME --partition=${PARTS:-$SLURM_JOB_PARTITION} ${CONSTRAINT:+--constraint=$CONSTRAINT} \
    --time=$(squeue -j $SLURM_JOB_ID -h -o %l) $REPO/policy/slurm/libero_train.sbatch
fi
