#!/bin/bash
# SAM 3 service on a GPU node: HexaShell's services/sam3_server.py with facebook/sam3's sam3.pt.
#
#   sbatch policy/slurm/sam3_service.sbatch                          # smoke test, then exit (gpu_devel)
#   MODE=serve sbatch -p gpu --time=1-00:00:00 policy/slurm/sam3_service.sbatch   # keep serving
#
# 1. downloads sam3.pt (3.4 GB, gated: needs approved access for the HF account in ~/.cache/huggingface/token)
#    to $SCR/models/sam3 once;
# 2. starts the server on 127.0.0.1:$PORT (it listens on localhost only, so clients run on this node);
# 3. MODE=smoke: segments one RoboCasa frame from our frame store (policy/sam3_smoke.py) and stops;
#    MODE=serve: writes the node and port to logs/sam3_service.addr and serves until the job ends.
# Envs: $PROJ/envs/sam3 (setup/setup_hexashell_bouchet.sh sam3). Steps: docs/sam3_service.md.
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=sam3
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/sam3_%j.out
set -euo pipefail
PROJ=/nfs/roberts/project/pi_dr877/ak3288
SCR=/nfs/roberts/scratch/pi_dr877/ak3288
HX=$PROJ/apollo-lab/HexaShell
RUA=$PROJ/apollo-lab/rua-harness
PY=$PROJ/envs/sam3/bin/python
MODE=${MODE:-smoke}
PORT=${PORT:-18300}
CKDIR=$SCR/models/sam3
CKPT=$CKDIR/sam3.pt
export HF_HOME=$SCR/cache/hf HF_HUB_CACHE=$SCR/cache/hf HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export PYTHONPATH=$RUA RUA_FRAMES=$SCR/frames

echo "== $(date '+%F %T') $(hostname) mode $MODE port $PORT"
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
    "$PY" "$HX/services/sam3_server.py" --port "$PORT" --checkpoint "$CKPT" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 600); do
    if curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
    kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -30 "$LOG"; exit 1; }
    sleep 1
done
echo "== server up after ${i} s"; tail -5 "$LOG"

if [ "$MODE" = smoke ]; then
    "$PY" -m policy.sam3_smoke --url "http://127.0.0.1:$PORT" --task TurnOnMicrowave --camera left --frame 0 \
        --text microwave "robot gripper" --save "$PROJ/apollo-lab/logs/sam3_smoke_${SLURM_JOB_ID:-local}"
    "$PY" -m policy.sam3_smoke --url "http://127.0.0.1:$PORT" --task PickPlaceCounterToCabinet --camera left \
        --frame 0 --text cabinet counter
else
    echo "$(hostname) $PORT ${SLURM_JOB_ID:-}" > "$PROJ/apollo-lab/logs/sam3_service.addr"
    wait $SERVER
fi
