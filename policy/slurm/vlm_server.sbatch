#!/bin/bash
# Open vision-language model served by vLLM (OpenAI-compatible API) for the commander
# (policy/commander/open_model.py, docs/vlm_commander.md).
#
#   sbatch -p gpu_devel --time=01:30:00 policy/slurm/vlm_server.sbatch            # size --time to the batch
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct sbatch ... policy/slurm/vlm_server.sbatch   # the other model
#
# Listens on 0.0.0.0:$PORT, so other Bouchet nodes reach it at http://<node>:$PORT/v1. Once healthy it writes
# "host:port", then url, job, model and end time to runs/vlm_server.txt; the file is removed when the job stops.
# Idle guard (YCRC cancels GPU jobs idle for 1 hour): the job exits if no request arrives within
# START_GRACE_MIN of start-up, or none for IDLE_MIN after the last one. Env: $PROJ/envs/vllm
# (setup/setup_vllm_bouchet.sh). Weights from the HF cache on scratch, offline.
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=vlm
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/vlm_%j.out
set -uo pipefail
PROJ=/nfs/roberts/project/pi_dr877/ak3288
SCR=/nfs/roberts/scratch/pi_dr877/ak3288
E=$PROJ/envs/vllm
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
PORT=${PORT:-18400}
MAXLEN=${MAXLEN:-12288}   # episodes use at most about 3.5k tokens; 32768 did not fit the KV cache on a 32 GB GPU (Nov 28)
START_GRACE_MIN=${START_GRACE_MIN:-40}
IDLE_MIN=${IDLE_MIN:-20}
ADDR=$PROJ/apollo-lab/runs/vlm_server.txt
export HF_HUB_CACHE=$SCR/cache/hf HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$SCR/cache/vllm TRITON_CACHE_DIR=$SCR/cache/triton
export XDG_CACHE_HOME=$SCR/cache TMPDIR=$SCR/cache/tmp PATH=$E/bin:$PATH
mkdir -p "$TMPDIR"
HOST=$(hostname -s)
LOG=$PROJ/apollo-lab/logs/vlm_server_${SLURM_JOB_ID:-local}.log
echo "== $(date '+%F %T') $HOST model $MODEL port $PORT job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$E/bin/vllm" serve "$MODEL" --host 0.0.0.0 --port "$PORT" --served-model-name vlm \
    --max-model-len "$MAXLEN" --limit-mm-per-prompt '{"image": 12}' --gpu-memory-utilization 0.90 \
    --seed 0 > "$LOG" 2>&1 &
SERVER=$!
cleanup() {
    kill $SERVER 2>/dev/null || true
    if [ -f "$ADDR" ] && grep -q "job ${SLURM_JOB_ID:-local}" "$ADDR"; then rm -f "$ADDR"; fi
}
trap cleanup EXIT
for i in $(seq 1 1200); do
    curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -40 "$LOG"; exit 1; }
    sleep 1
done
echo "== server up after ${i} s"
URL="http://$HOST:$PORT"
curl -sf --max-time 5 "$URL/v1/models" && echo " <- models at $URL"
END=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null || echo unknown)
printf '%s:%s\nurl %s\njob %s\nmodel %s\nends %s\n' "$HOST" "$PORT" "$URL" "${SLURM_JOB_ID:-local}" "$MODEL" "$END" > "$ADDR"
echo "== serving at $URL until $END (address in $ADDR)"

# idle guard: count chat requests in the server's access log
start=$(date +%s); last=$start; seen=0; n_prev=0
while kill -0 $SERVER 2>/dev/null; do
    sleep 60
    n=$(grep -c "POST /v1/chat/completions" "$LOG" 2>/dev/null); n=${n:-0}
    now=$(date +%s)
    if [ "$n" -gt "$n_prev" ]; then last=$now; seen=1; n_prev=$n; fi
    if [ $seen -eq 0 ] && [ $((now - start)) -gt $((START_GRACE_MIN * 60)) ]; then
        echo "== $(date '+%T') no request in ${START_GRACE_MIN} min; stopping"; break; fi
    if [ $seen -eq 1 ] && [ $((now - last)) -gt $((IDLE_MIN * 60)) ]; then
        echo "== $(date '+%T') idle ${IDLE_MIN} min after $n requests; stopping"; break; fi
done
echo "== requests served: $(grep -c 'POST /v1/chat/completions' "$LOG")"
