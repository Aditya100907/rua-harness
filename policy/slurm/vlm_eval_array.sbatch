#!/bin/bash
# Open-model commander episodes: one task per array index, NPROC parallel episode processes on one CPU node
# (policy on CPU, simulator with software EGL), against a running vLLM server (policy/slurm/vlm_server.sbatch).
#   CKPT=$PROJ/checkpoints/fs_mse_points_2d/ckpt_0020000.pt TASKS="PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave" \
#     EPISODES=10 sbatch --array=0-2 --dependency=after:<server job> policy/slurm/vlm_eval_array.sbatch
# Waits up to WAIT_MIN minutes for runs/vlm_server.txt. Extra open_model.py flags go in VLM_ARGS.
# Output: results/<date>_agent_plus_policy_vlm_<NOTE>_<task>.csv (NOTE default qwen3vl8b_<run id>_<step>).
#SBATCH --partition=day
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=vlm_eval
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/vlm_eval_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
T=($TASKS)
TASK=${T[$SLURM_ARRAY_TASK_ID]}
EPISODES=${EPISODES:-10}
NPROC=${NPROC:-$EPISODES}
WAIT_MIN=${WAIT_MIN:-60}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=2
RUNID=$(basename $(dirname $CKPT)); STEP=$(basename $CKPT .pt | sed 's/ckpt_0*//'); STEP=${STEP:-0}
NOTE=${NOTE:-qwen3vl8b_${RUNID}_${STEP}}
ADDR=$ROOT/runs/vlm_server.txt
for i in $(seq 1 $((WAIT_MIN * 6))); do [ -f $ADDR ] && break; sleep 10; done
[ -f $ADDR ] || { echo "no server address after $WAIT_MIN min"; exit 1; }
URL=$(sed -n 2p $ADDR | cut -d' ' -f2); MODEL_ID=$(sed -n 4p $ADDR | cut -d' ' -f2)
curl -sf --max-time 10 $URL/v1/models > /dev/null || { echo "server at $URL does not answer"; exit 1; }
D=$REPO/results/parts/vlm_${NOTE}_${TASK}
mkdir -p $D; rm -f $D/shard*.csv
echo "task $TASK note $NOTE ckpt $CKPT episodes $EPISODES nproc $NPROC server $URL ($MODEL_ID) on $(hostname)"
for s in $(seq 0 $((NPROC-1))); do
  python -m policy.commander.open_model --ckpt $CKPT --tasks $TASK --episodes $EPISODES --shard $s --nshards $NPROC \
    --url $URL --model-id "$MODEL_ID" --note $NOTE --out $D/shard$s.csv ${VLM_ARGS:-} > $D/shard$s.log 2>&1 &
done
wait
python - <<PY
import csv, glob
rows = []
for f in sorted(glob.glob("$D/shard*.csv")):
    rows += list(csv.DictReader(open(f)))
rows = list({(r["task"], r["episode"]): r for r in rows}.values())
rows.sort(key=lambda r: int(r["episode"]))
out = "$REPO/results/$(date +%F)_agent_plus_policy_vlm_${NOTE}_${TASK}.csv"
if rows:
    w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = sum(int(r["success"]) for r in rows)
print(f"VLM_DONE $TASK {ok}/{len(rows)} (expected $EPISODES) -> {out}")
PY
grep -h "Error\|Traceback" $D/shard*.log | head -5
