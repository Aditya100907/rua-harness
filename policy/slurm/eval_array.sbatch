#!/bin/bash
# Evaluation job array: one task per array index, NPROC parallel simulator processes on one GPU.
#   CKPT=/path/ckpt_0020000.pt TASKS="OpenDrawer TurnOnMicrowave" EPISODES=50 \
#     sbatch --array=0-1 policy/slurm/eval_array.sbatch
# Extra eval.py flags go in EVAL_ARGS (e.g. "--wiring" or "--save-traj").
#SBATCH --partition=day
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=eval
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/eval_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
T=($TASKS)
TASK=${T[$SLURM_ARRAY_TASK_ID]}
NPROC=${NPROC:-12}
# CPU node: software EGL rendering and the policy on CPU, 2 threads per process
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
EPISODES=${EPISODES:-50}
if [ -n "${CKPT:-}" ]; then
  RUNID=$(basename $(dirname $CKPT)); STEP=$(basename $CKPT .pt | sed 's/ckpt_0*//'); STEP=${STEP:-0}
  NOTE=${NOTE:-${RUNID}_${STEP}}
  CK="--ckpt $CKPT"
else
  NOTE=${NOTE:-wiring}; CK="--wiring"
fi
D=$REPO/results/parts/${NOTE}_${TASK}
mkdir -p $D
rm -f $D/shard*.csv          # a requeued or repeated job starts clean (no double-counted rows)
echo "task $TASK note $NOTE ckpt ${CKPT:-none} nproc $NPROC on $(hostname) (CPU)"
for s in $(seq 0 $((NPROC-1))); do
  python -m policy.eval $CK --tasks $TASK --episodes $EPISODES --shard $s --nshards $NPROC \
    --video $([ $s -eq 0 ] && echo 2 || echo 0) --out $D/shard$s.csv --outdir $ROOT/runs/eval_media/${NOTE} ${EVAL_ARGS:-} \
    > $D/shard$s.log 2>&1 &
done
wait
# retry pass: a shard that crashed (for example GPU out of memory) runs again alone
for s in $(seq 0 $((NPROC-1))); do
  want=$(python -c "print(len(range($s, $EPISODES, $NPROC)))")
  have=$(($(wc -l < $D/shard$s.csv 2>/dev/null || echo 1) - 1))
  if grep -q Traceback $D/shard$s.log || [ "$have" -lt "$want" ]; then
    echo "retrying shard $s"; mv $D/shard$s.log $D/shard$s.failed.log; rm -f $D/shard$s.csv
    python -m policy.eval $CK --tasks $TASK --episodes $EPISODES --shard $s --nshards $NPROC --video 0 \
      --out $D/shard$s.csv --outdir $ROOT/runs/eval_media/${NOTE} ${EVAL_ARGS:-} > $D/shard$s.log 2>&1
  fi
done
python - <<PY
import csv, glob
rows = []
for f in sorted(glob.glob("$D/shard*.csv")):
    rows += list(csv.DictReader(open(f)))
rows = list({(r["task"], r["episode"]): r for r in rows}.values())     # dedupe
rows.sort(key=lambda r: int(r["episode"]))
expected = $EPISODES
if len(rows) != expected:
    print(f"EVAL_INCOMPLETE $TASK: {len(rows)} of {expected} episodes")
out = "$REPO/results/$(date +%F)_policy_only_${NOTE}_${TASK}.csv"
if rows:
    w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = sum(int(r["success"]) for r in rows)
print(f"EVAL_DONE $TASK {ok}/{len(rows)} (expected {expected}) -> {out}")
PY
grep -h "Error\|Traceback" $D/shard*.log | head -5
