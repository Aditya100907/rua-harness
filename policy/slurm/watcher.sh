#!/bin/bash
# The cluster drives itself (Part M). A plain bash loop, no LLM. Submit once:
#   sbatch policy/slurm/watcher.sbatch
# Every 10 minutes it:
#   1. resubmits training runs that died before reaching their --steps (unless <run>/STOP exists)
#   2. submits an evaluation for every checkpoint that has none (marker files in <run>/scored/)
#   3. launches the next line of $ROOT/runs/queue.txt when GPUs are free under the cap
#   4. rebuilds the results tables
#   5. appends one line to $ROOT/logs/watcher.log
# Queue line format (one experiment per line, '#' comments allowed):
#   RUN_NAME | train.py arguments
# Score policy per run (config.json "tasks"): every 5k steps for a run on 3 tasks or fewer;
# for bigger runs every 50k steps and the final one. Episodes: 50, or 100 for a run's final checkpoint.
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
GPU_CAP=${GPU_CAP:-8}
JOB_CAP=${JOB_CAP:-12}
EVAL_RESERVE=${EVAL_RESERVE:-0}   # scoring runs on CPU nodes now (YCRC cancels GPU jobs with 0% use)
CK=$PROJ/checkpoints
Q=$ROOT/runs/queue.txt
QDONE=$ROOT/runs/queue_done.txt
LOG=$ROOT/logs/watcher.log
touch $Q $QDONE

gpus_in_use() {  # running + pending GPU jobs of mine (each of our GPU jobs uses 1 GPU)
  # running GPU jobs, plus pending GPU jobs with each throttled array counted once (not per element)
  echo $(( $(squeue -u $USER -h -r -t R -o "%P" | grep -c "^gpu") + $(squeue -u $USER -h -t PD -o "%P" | grep -c "^gpu") ))
}
jobs_in_use() { squeue -u $USER -h -r -t R -o "%P" | grep -c "^gpu"; }  # GPU jobs only; CPU arrays do not count
job_exists() { squeue -u $USER -h -n "$1" | grep -q .; }

best_partition() {  # earliest predicted start among the partitions we train on
  local best="" bt=99999999999 p t
  for p in ${TRAIN_PARTS:-gpu_b200 gpu_h100 gpu_h200}; do
    t=$(RUN=x ARGS=x sbatch --test-only --partition=$p $REPO/policy/slurm/train.sbatch 2>&1 | grep -o "start at [0-9T:-]*" | awk '{print $3}')
    [ -z "$t" ] && continue
    t=$(date -d "$t" +%s 2>/dev/null) || continue
    if [ "$t" -lt "$bt" ]; then bt=$t; best=$p; fi
  done
  echo ${best:-gpu_h100}
}

launch_train() {  # run args
  bash $REPO/policy/slurm/launch_train.sh "$1" "$2"
}

cycle() {
  local msg="" n_resub=0 n_eval=0 n_queue=0
  # 1. dead training runs
  for d in $CK/*/; do
    run=$(basename $d)
    [ -f $d/config.json ] || continue
    [ -f $d/STOP ] && continue
    [ -f $d/CRASHED ] && continue          # needs a person: see <run>/last_job.log
    grep -q '"untrained": true' $d/config.json && continue
    want=$(python -c "import json;print(json.load(open('$d/config.json'))['steps'])")
    # progress = newest checkpoint's step (loading the 1.3 GB train_state.pt failed on the 8 GB agent node,
    # which made the watcher resubmit finished runs every cycle on Nov 27, 05:00 to 06:30)
    last=$(ls $d/ckpt_*.pt 2>/dev/null | grep -v '\.done\.pt' | sed 's/.*ckpt_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    have=${last:-0}
    [ "$have" -ge "$want" ] && continue
    if [ "$have" -lt "$want" ] && ! job_exists $run; then
      args=$(cat $d/args.txt 2>/dev/null)
      if [ -n "$args" ] && [ $(gpus_in_use) -lt $GPU_CAP ]; then
        j=$(launch_train $run "$args"); n_resub=$((n_resub+1)); msg="$msg resub:$run@$have:$j"
      fi
    fi
  done
  # 2. unscored checkpoints
  for ck in $(ls $CK/*/ckpt_*.pt 2>/dev/null); do
    d=$(dirname $ck); run=$(basename $d)
    step=$(basename $ck .pt | sed 's/ckpt_0*//'); step=${step:-0}
    mkdir -p $d/scored
    [ -f $d/scored/$step ] && { grep -q "objsplit_train" $d/config.json 2>/dev/null && [ ! -f $d/scored/${step}_ho ] || continue; }
    grep -q '"untrained": true' $d/config.json 2>/dev/null && { touch $d/scored/$step; continue; }
    read ntasks want tasks <<< $(python -c "
import json; c=json.load(open('$d/config.json')); print(len(c['tasks']), c['steps'], ' '.join(c['tasks']))")
    final=0; [ "$step" -ge "$want" ] && final=1
    if [ "$ntasks" -gt 3 ] && [ $final -eq 0 ] && [ $((step % 50000)) -ne 0 ]; then touch $d/scored/$step; continue; fi
    if [ "$ntasks" -le 3 ] && [ $final -eq 0 ] && [ $((step % 5000)) -ne 0 ]; then touch $d/scored/$step; continue; fi
    eps=50; [ $final -eq 1 ] && eps=100
    conc=$ntasks                          # CPU jobs: no GPU cap
    if [ ! -f $d/scored/$step ]; then
      ea=""; [ "$ntasks" -le 3 ] && ea="--save-traj"    # trajectories for failure sorting (E5)
      j=$(CKPT=$ck TASKS="$tasks" EPISODES=$eps EVAL_ARGS="$ea" sbatch --parsable --job-name=eval_${run}_$step \
          --array=0-$((ntasks-1))%$conc $REPO/policy/slurm/eval_array.sbatch) && touch $d/scored/$step \
          && n_eval=$((n_eval+1)) && msg="$msg eval:${run}@$step:$j"
    fi
    # object-split runs: also score the held-out objects (validation half; the test half stays locked)
    if grep -q "objsplit_train" $d/config.json && [ ! -f $d/scored/${step}_ho ]; then
      vj=$SCR/splits/objsplit_val.json
      vt=$(python -c "import json;print(' '.join(json.load(open('$vj'))))")
      nv=$(echo $vt | wc -w)
      j2=$(CKPT=$ck TASKS="$vt" EPISODES=$eps NOTE=${run}_${step}_heldoutval EVAL_ARGS="--demo-scenes $vj" \
          sbatch --parsable --job-name=evalho_${run}_$step --array=0-$((nv-1))%1 $REPO/policy/slurm/eval_array.sbatch) \
          && touch $d/scored/${step}_ho && msg="$msg evalheldout:${run}@$step:$j2"
    fi
  done
  # 2b. Part N sweeps on the final checkpoint of every point-format run (once)
  for d in $CK/*/; do
    run=$(basename $d)
    [ -f $d/config.json ] || continue
    fmt=$(python -c "import json;print(json.load(open('$d/config.json')).get('format','sentence'))")
    case $fmt in points_2d|points_3d|points_3d_rgb) ;; *) continue ;; esac
    want=$(python -c "import json;print(json.load(open('$d/config.json'))['steps'])")
    fck=$(printf "%s/ckpt_%07d.pt" $d $want)
    [ -f $fck ] && [ ! -f $d/scored/sweeps ] || continue
    tasks=$(python -c "import json;print(' '.join(json.load(open('$d/config.json'))['tasks']))")
    j=$(TASKS="$tasks" bash $REPO/policy/slurm/sweeps.sh $fck 50) && touch $d/scored/sweeps && msg="$msg sweeps:$run:$j"
  done
  # 3. queue
  while [ $(gpus_in_use) -lt $((GPU_CAP - EVAL_RESERVE)) ] && [ $(jobs_in_use) -lt $JOB_CAP ]; do
    line=$(grep -v '^\s*#' $Q | grep -v '^\s*$' | head -1)
    [ -z "$line" ] && break
    run=$(echo "$line" | cut -d'|' -f1 | xargs); args=$(echo "$line" | cut -d'|' -f2- | xargs)
    # remove the line from the queue first, so a failure cannot launch it twice
    grep -vxF "$line" $Q > $Q.tmp; mv $Q.tmp $Q
    if [ -d $CK/$run ] || job_exists $run; then echo "$(date +%F_%T) SKIP exists $line" >> $QDONE; continue; fi
    mkdir -p $CK/$run; echo "$args" > $CK/$run/args.txt
    j=$(launch_train $run "$args"); echo "$(date +%F_%T) $j $line" >> $QDONE
    n_queue=$((n_queue+1)); msg="$msg queue:$run:$j"
  done
  # 4. tables
  (cd $REPO && python scripts/make_tables.py > /dev/null 2>&1) || msg="$msg tables:FAILED"
  echo "$(date +%F_%T) gpus=$(gpus_in_use)/$GPU_CAP jobs=$(jobs_in_use) resub=$n_resub eval=$n_eval queue=$n_queue qleft=$(grep -cv '^\s*#\|^\s*$' $Q)$msg" >> $LOG
}

if [ "${1:-}" = "once" ]; then cycle; exit 0; fi
while true; do
  cycle
  [ -f $ROOT/runs/WATCHER_STOP ] && { echo "$(date +%F_%T) stop file seen" >> $LOG; exit 0; }
  sleep ${PERIOD:-600}
done
