#!/bin/bash
# SAM 3 service on a GPU node: HexaShell's services/sam3_server.py with facebook/sam3's sam3.pt.
#
#   sbatch policy/slurm/sam3_service.sbatch                                  # serve (default), partition gpu, 6 h
#   sbatch -p gpu_devel --time=01:00:00 --export=ALL,MODE=smoke policy/slurm/sam3_service.sbatch   # smoke test only
#
# 1. downloads sam3.pt (3.4 GB, gated; the HF login in ~/.cache/huggingface/token has access) to $SCR/models/sam3
#    once;
# 2. starts the server on 127.0.0.1:$INNER (HexaShell's server listens on localhost only) and a TCP forwarder
#    (policy/tcp_forward.py) on 0.0.0.0:$PORT, so other Bouchet nodes reach it at http://<node>:$PORT;
# 3. runs the smoke test (policy/sam3_smoke.py) through the forwarder;
# 4. MODE=serve: writes "host:port" and the URL to runs/sam3_service.txt and serves until the job's time ends
#    (the file is removed when the job stops). MODE=smoke: stops after the test.
# Envs: $PROJ/envs/sam3 (setup/setup_hexashell_bouchet.sh sam3). Details: docs/sam3_service.md.
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=sam3
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/sam3_%j.out
set -euo pipefail
PROJ=/nfs/roberts/project/pi_dr877/ak3288
SCR=/nfs/roberts/scratch/pi_dr877/ak3288
HX=$PROJ/apollo-lab/HexaShell
RUA=$PROJ/apollo-lab/rua-harness
PY=$PROJ/envs/sam3/bin/python
MODE=${MODE:-serve}
PORT=${PORT:-18300}
INNER=${INNER:-18299}
CKDIR=$SCR/models/sam3
CKPT=$CKDIR/sam3.pt
ADDR=$PROJ/apollo-lab/runs/sam3_service.txt
export HF_HUB_CACHE=$SCR/cache/hf HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export PYTHONPATH=$RUA RUA_FRAMES=$SCR/frames
HOST=$(hostname -s)

echo "== $(date '+%F %T') $HOST mode $MODE port $PORT (inner $INNER) job ${SLURM_JOB_ID:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -f "$CKPT" ]; then
    echo "== downloading facebook/sam3 sam3.pt to $CKDIR"
    "$PY" - "$CKDIR" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download("facebook/sam3", "sam3.pt", local_dir=sys.argv[1]))
EOF
fi
ls -l "$CKPT"

LOG=$PROJ/apollo-lab/logs/sam3_server_${SLURM_JOB_ID:-local}.log
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 MALLOC_ARENA_MAX=4 \
    "$PY" "$HX/services/sam3_server.py" --port "$INNER" --checkpoint "$CKPT" > "$LOG" 2>&1 &
SERVER=$!
"$PY" -m policy.tcp_forward --listen "0.0.0.0:$PORT" --target "127.0.0.1:$INNER" >> "$LOG.fwd" 2>&1 &
FWD=$!
cleanup() {
    kill $SERVER $FWD 2>/dev/null || true
    if [ -f "$ADDR" ] && grep -q "job ${SLURM_JOB_ID:-local}" "$ADDR"; then rm -f "$ADDR"; fi
}
trap cleanup EXIT
for i in $(seq 1 900); do
    if curl -sf --max-time 5 "http://127.0.0.1:$INNER/health" > /dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -30 "$LOG"; exit 1; }
    sleep 1
done
echo "== server up after ${i} s"; tail -4 "$LOG"
URL="http://$HOST:$PORT"
curl -sf --max-time 5 "$URL/health" && echo " <- health through the forwarder at $URL"

"$PY" -m policy.sam3_smoke --url "$URL" --task TurnOnMicrowave --camera left --frame 0 \
    --text microwave "robot gripper" --save "$PROJ/apollo-lab/logs/sam3_smoke_${SLURM_JOB_ID:-local}" || true
"$PY" -m policy.sam3_smoke --url "$URL" --task PickPlaceCounterToCabinet --camera left --frame 0 \
    --text cabinet counter || true

if [ "$MODE" = serve ]; then
    END=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null || echo unknown)
    printf '%s:%s\nurl %s\njob %s\nends %s\n' "$HOST" "$PORT" "$URL" "${SLURM_JOB_ID:-local}" "$END" > "$ADDR"
    echo "== serving at $URL until $END (address in $ADDR)"
    wait $SERVER
fi
