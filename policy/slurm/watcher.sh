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
Q=${QUEUE_FILE:-$ROOT/runs/queue.txt}
QDONE=$ROOT/runs/queue_done.txt
LOG=${WATCHER_LOG:-$ROOT/logs/watcher.log}
touch $Q $QDONE

gpus_in_use() {  # sum of GPUs requested by my running and pending jobs (a 4-GPU job counts 4)
  squeue -u $USER -h -t R,PD -O "tres-per-job:40" | grep -o "gpu:[0-9]*" | cut -d: -f2 | awk '{s+=$1} END {print s+0}'
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
  if [ -n "${DRYRUN:-}" ]; then echo "dry-$1"; return; fi
  bash $REPO/policy/slurm/launch_train.sh "$1" "$2"
}

cycle() {
  local msg="" n_resub=0 n_eval=0 n_queue=0
  if [ -z "${QUEUE_ONLY:-}" ]; then
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
    [ -f $d/scored/$step ] && { { grep -q "objsplit_train" $d/config.json 2>/dev/null && [ ! -f $d/scored/${step}_ho ]; } || { grep -q "Twin" $d/config.json 2>/dev/null && [ ! -f $d/scored/${step}_twin ]; } || continue; }
    grep -q '"untrained": true' $d/config.json 2>/dev/null && { touch $d/scored/$step; continue; }
    read ntasks want tasks <<< $(python -c "
import json; c=json.load(open('$d/config.json')); print(len([t for t in c['tasks'] if 'Twin' not in t]), c['steps'], ' '.join(c['tasks']))")
    final=0; [ "$step" -ge "$want" ] && final=1
    if [ "$ntasks" -gt 3 ] && [ $final -eq 0 ] && [ $((step % 50000)) -ne 0 ]; then touch $d/scored/$step; continue; fi
    if [ "$ntasks" -le 3 ] && [ $final -eq 0 ] && [ $((step % 5000)) -ne 0 ]; then touch $d/scored/$step; continue; fi
    eps=50; [ $final -eq 1 ] && eps=100
    conc=$ntasks                          # CPU jobs: no GPU cap
    # twin-trained runs: the virtual twin task is scored separately on held-out twin scenes (L3)
    if echo "$tasks" | grep -q "Twin"; then
      if [ ! -f $d/scored/${step}_twin ]; then
        TJ=$ROOT/runs/twin_heldout50.json
        j3=$(CKPT=$ck TASKS="PickPlaceCounterToCabinetTwin" EPISODES=50 NOTE=${run}_${step}_twinL3 \
             EVAL_ARGS="--demo-scenes $TJ" sbatch --parsable --job-name=evaltw_${run}_$step --array=0 \
             $REPO/policy/slurm/eval_array.sbatch) && \
        j4=$(CKPT=$ck TASKS="PickPlaceCounterToCabinetTwin" EPISODES=50 NOTE=${run}_${step}_twinpoint \
             EVAL_ARGS="--demo-scenes $TJ --twin-point" sbatch --parsable --job-name=evaltp_${run}_$step --array=0 \
             $REPO/policy/slurm/eval_array.sbatch) && touch $d/scored/${step}_twin && msg="$msg evaltwin:${run}@$step:$j3,$j4"
      fi
      tasks=$(echo $tasks | tr " " "\n" | grep -v Twin | tr "\n" " "); ntasks=$(echo $tasks | wc -w); conc=$ntasks
    fi
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
  fi   # end of steps 1, 2, 2b (skipped with QUEUE_ONLY=1, for tests)
  # 3. queue, with GPU pools (Adi, Nov 28): A = fast 3-task experiments (4), B = long runs (3),
  #    C = open model and HexaShell (1). Lines: "POOL | RUN | ARGS" or "RUN | ARGS" (pool from length).
  #    A pool may exceed its share only when the other pools have nothing queued.
  while [ $(gpus_in_use) -lt $((GPU_CAP - EVAL_RESERVE)) ] && [ $(jobs_in_use) -lt $JOB_CAP ]; do
    pick=""; pickpool=""
    declare -A used=([A]=0 [B]=0 [C]=0) queued=([A]=0 [B]=0 [C]=0)
    for jn in $(squeue -u $USER -h -t R,PD -o "%j|%P" | grep "|gpu" | cut -d"|" -f1); do
      pl=$(cat $CK/$jn/pool.txt 2>/dev/null || echo C); used[$pl]=$(( ${used[$pl]:-0} + 1 ))
    done
    mapfile -t LINES < <(grep -v '^\s*#' $Q | grep -v '^\s*$')
    for line in "${LINES[@]}"; do
      nf=$(echo "$line" | awk -F'|' '{print NF}')
      if [ "$nf" -ge 3 ]; then pl=$(echo "$line" | cut -d'|' -f1 | xargs)
      else st=$(echo "$line" | grep -o -- '--steps [0-9]*' | awk '{print $2}'); pl=A; [ "${st:-0}" -gt 30000 ] && pl=B; fi
      queued[$pl]=$(( ${queued[$pl]:-0} + 1 ))
    done
    declare -A share=([A]=4 [B]=3 [C]=1)
    for want_borrow in 0 1; do
      for line in "${LINES[@]}"; do
        nf=$(echo "$line" | awk -F'|' '{print NF}')
        if [ "$nf" -ge 3 ]; then pl=$(echo "$line" | cut -d'|' -f1 | xargs)
        else st=$(echo "$line" | grep -o -- '--steps [0-9]*' | awk '{print $2}'); pl=A; [ "${st:-0}" -gt 30000 ] && pl=B; fi
        if [ ${used[$pl]} -lt ${share[$pl]} ]; then pick="$line"; pickpool=$pl; break 2; fi
        if [ $want_borrow = 1 ]; then
          others=0; for o in A B C; do [ $o != $pl ] && [ ${used[$o]} -lt ${share[$o]} ] && [ ${queued[$o]} -gt 0 ] && others=1; done
          [ $others = 0 ] && { pick="$line"; pickpool=$pl; break 2; }
        fi
      done
    done
    [ -z "$pick" ] && break
    nf=$(echo "$pick" | awk -F'|' '{print NF}')
    if [ "$nf" -ge 3 ]; then run=$(echo "$pick" | cut -d'|' -f2 | xargs); args=$(echo "$pick" | cut -d'|' -f3- | xargs)
    else run=$(echo "$pick" | cut -d'|' -f1 | xargs); args=$(echo "$pick" | cut -d'|' -f2- | xargs); fi
    grep -vxF "$pick" $Q > $Q.tmp; mv $Q.tmp $Q
    if [ -d $CK/$run ] || job_exists $run; then echo "$(date +%F_%T) SKIP exists $pick" >> $QDONE; continue; fi
    mkdir -p $CK/$run; echo "$args" > $CK/$run/args.txt; echo $pickpool > $CK/$run/pool.txt
    j=$(launch_train $run "$args"); echo "$(date +%F_%T) $j pool $pickpool $pick" >> $QDONE
    n_queue=$((n_queue+1)); msg="$msg queue:$run:$pickpool:$j"
  done
  # 4. tables
  (cd $REPO && python scripts/make_tables.py > /dev/null 2>&1) || msg="$msg tables:FAILED"
  echo "$(date +%F_%T) gpus=$(gpus_in_use)/$GPU_CAP jobs=$(jobs_in_use) resub=$n_resub eval=$n_eval queue=$n_queue qleft=$(grep -cv '^\s*#\|^\s*$' $Q)$msg" >> $LOG
}

if [ -n "${SOURCE_ONLY:-}" ]; then return 0 2>/dev/null || exit 0; fi   # tests: define functions only
if [ "${1:-}" = "once" ]; then cycle; exit 0; fi
while true; do
  cycle
  [ -f $ROOT/runs/WATCHER_STOP ] && { echo "$(date +%F_%T) stop file seen" >> $LOG; exit 0; }
  sleep ${PERIOD:-600}
done
