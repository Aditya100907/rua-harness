#!/bin/bash
# Evaluation job array: one task per array index, NPROC parallel simulator processes on one GPU.
#   CKPT=/path/ckpt_0020000.pt TASKS="OpenDrawer TurnOnMicrowave" EPISODES=50 \
#     sbatch --array=0-1 policy/slurm/eval_array.sbatch
# Extra eval.py flags go in EVAL_ARGS (e.g. "--wiring" or "--save-traj").
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=eval
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/eval_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
T=($TASKS)
TASK=${T[$SLURM_ARRAY_TASK_ID]}
NPROC=${NPROC:-10}
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
echo "task $TASK note $NOTE ckpt ${CKPT:-none} nproc $NPROC on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for s in $(seq 0 $((NPROC-1))); do
  python -m policy.eval $CK --tasks $TASK --episodes $EPISODES --shard $s --nshards $NPROC \
    --video $([ $s -eq 0 ] && echo 2 || echo 0) --out $D/shard$s.csv --outdir $ROOT/runs/eval_media/${NOTE} ${EVAL_ARGS:-} \
    > $D/shard$s.log 2>&1 &
done
wait
python - <<PY
import csv, glob
rows = []
for f in sorted(glob.glob("$D/shard*.csv")):
    rows += list(csv.DictReader(open(f)))
rows.sort(key=lambda r: int(r["episode"]))
out = "$REPO/results/$(date +%F)_policy_only_${NOTE}_${TASK}.csv"
if rows:
    w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = sum(int(r["success"]) for r in rows)
print(f"EVAL_DONE $TASK {ok}/{len(rows)} -> {out}")
PY
grep -h "Error\|Traceback" $D/shard*.log | head -5
