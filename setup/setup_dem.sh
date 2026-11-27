#!/bin/bash
# One-shot setup + health check for the DEM agent-harness project (APOLLO Lab).
#
#   bash setup_dem.sh            # clone, build the env, run all checks (about 10 min, ~6 GB)
#   bash setup_dem.sh --assets   # same, then download RoboCasa kitchen assets (~10 GB) and run a sim rollout
#
# Safe to re-run: existing clones and the existing env are reused.
# Everything lands in ~/apollo-lab (override with APOLLO_DIR=/some/path).
# A full log is written to ~/apollo-lab/setup_log.txt.

ROOT="${APOLLO_DIR:-$HOME/apollo-lab}"
ENV_NAME="${DEM_ENV:-dem}"
PY_VER="3.11"
WITH_ASSETS=0
[ "${1:-}" = "--assets" ] && WITH_ASSETS=1

mkdir -p "$ROOT" || { echo "cannot create $ROOT"; exit 1; }
LOG="$ROOT/setup_log.txt"
SUMMARY="$ROOT/.setup_summary.tmp"
: > "$SUMMARY"
exec > >(tee "$LOG") 2>&1

note()  { printf '%-5s %s\n' "$1" "$2" >> "$SUMMARY"; }
title() { printf '\n========== %s ==========\n' "$1"; }
# run "label" cmd...  -> records PASS/FAIL, never aborts the script
run() {
  local label="$1"; shift
  if "$@"; then note PASS "$label"; return 0; else note FAIL "$label"; return 1; fi
}

title "0. System"
date
uname -a
command -v sw_vers >/dev/null 2>&1 && sw_vers
df -h "$ROOT" | tail -1
FREE_GB=$(df -g "$ROOT" 2>/dev/null | awk 'NR==2{print $4}')
[ -z "$FREE_GB" ] && FREE_GB=$(df -BG "$ROOT" | awk 'NR==2{gsub("G","",$4); print $4}')
echo "free disk: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -ge 25 ]; then note PASS "disk space (${FREE_GB} GB free)"; else note WARN "disk space low (${FREE_GB} GB free; want 25+ with assets)"; fi

title "1. git and GitHub"
if run "git installed" command -v git; then git --version; fi
GIT_NAME=$(git config --global user.name); GIT_EMAIL=$(git config --global user.email)
echo "git user.name : ${GIT_NAME:-<not set>}"
echo "git user.email: ${GIT_EMAIL:-<not set>}"
if [ -n "$GIT_NAME" ] && [ -n "$GIT_EMAIL" ]; then note PASS "git identity ($GIT_NAME <$GIT_EMAIL>)"; else note FAIL "git identity not set (git config --global user.name / user.email)"; fi

SSH_OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1)
echo "ssh  : $SSH_OUT"
SSH_USER=$(echo "$SSH_OUT" | sed -n 's/^Hi \([^!]*\)!.*/\1/p')
if [ -n "$SSH_USER" ]; then note PASS "GitHub SSH auth as '$SSH_USER'"; else note INFO "GitHub SSH key not set up (fine if you use HTTPS or gh)"; fi

GH_USER=""
if command -v gh >/dev/null 2>&1; then
  gh auth status 2>&1 | sed 's/^/gh   : /'
  GH_USER=$(gh api user --jq .login 2>/dev/null)
  if [ -n "$GH_USER" ]; then note PASS "GitHub CLI auth as '$GH_USER'"; else note INFO "gh installed but not logged in (gh auth login)"; fi
else
  note INFO "GitHub CLI (gh) not installed (optional: brew install gh)"
fi
if [ -z "$SSH_USER" ] && [ -z "$GH_USER" ]; then
  note WARN "no GitHub login found on this Mac: cloning public repos works, pushing or cloning private repos will not"
fi
run "reach DEM repo over HTTPS" git ls-remote --heads https://github.com/Apollo-Lab-Yale/decoupled-embodiment-model.git >/dev/null
if git ls-remote --heads https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark.git >/dev/null 2>&1; then
  note PASS "reach benchmark repo (read)"
else
  note INFO "benchmark repo not readable from this Mac (needs access from Xiatao; optional side thread)"
fi

title "2. Clone repos into $ROOT"
clone() {  # clone url dir
  if [ -d "$2/.git" ]; then echo "exists: $2 (pulling)"; git -C "$2" pull --ff-only || true
  else git clone "$1" "$2"; fi
  [ -d "$2/.git" ]
}
run "clone decoupled-embodiment-model" clone https://github.com/Apollo-Lab-Yale/decoupled-embodiment-model.git "$ROOT/decoupled-embodiment-model"
run "clone robosuite"                  clone https://github.com/ARISE-Initiative/robosuite.git "$ROOT/robosuite"
run "clone robocasa"                   clone https://github.com/robocasa/robocasa.git "$ROOT/robocasa"
for r in decoupled-embodiment-model robosuite robocasa; do
  [ -d "$ROOT/$r/.git" ] && echo "$r @ $(git -C "$ROOT/$r" log -1 --format='%h %ad %s' --date=short)"
done

title "3. Python environment '$ENV_NAME' (python $PY_VER)"
CONDA_BIN=$(command -v conda 2>/dev/null)
[ -z "$CONDA_BIN" ] && [ -x "$HOME/miniconda3/bin/conda" ] && CONDA_BIN="$HOME/miniconda3/bin/conda"
ENV_OK=0
if [ -n "$CONDA_BIN" ]; then
  CONDA_BASE=$("$CONDA_BIN" info --base)
  # shellcheck disable=SC1091
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PY_VER"
  else
    echo "conda env '$ENV_NAME' already exists, reusing"
  fi
  conda activate "$ENV_NAME" && ENV_OK=1
  ACTIVATE_HINT="conda activate $ENV_NAME"
else
  echo "conda not found, falling back to a venv at $ROOT/.venv"
  PYBIN=$(command -v "python$PY_VER" || command -v python3)
  [ -d "$ROOT/.venv" ] || "$PYBIN" -m venv "$ROOT/.venv"
  # shellcheck disable=SC1091
  . "$ROOT/.venv/bin/activate" && ENV_OK=1
  ACTIVATE_HINT="source $ROOT/.venv/bin/activate"
fi
if [ "$ENV_OK" = 1 ]; then note PASS "python env ($(python --version 2>&1), $(command -v python))"; else note FAIL "python env"; fi
python -m pip install -q --upgrade pip

# pip's build chatter goes to its own file so the main log stays readable
PIPLOG="$ROOT/pip_install.log"; : > "$PIPLOG"
qpip() { python -m pip install -q "$@" >> "$PIPLOG" 2>&1 || { tail -15 "$PIPLOG"; return 1; }; }

title "4. Install DEM and run its tests"
cd "$ROOT/decoupled-embodiment-model" || exit 1
run "pip install -e 'dem[dev,vit]'" qpip -e ".[dev,vit]"
run "DEM fast tests (pytest -m 'not slow')" python -m pytest -m "not slow" -q

title "5. Device check and a 10-step training run (synthetic data, no downloads)"
mkdir -p "$ROOT/runs"; cd "$ROOT/runs" || exit 1
DEVICE=$(python - <<'EOF'
import torch
print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
EOF
)
echo "torch $(python -c 'import torch; print(torch.__version__)'), best device: $DEVICE"
train() { python "$ROOT/decoupled-embodiment-model/examples/train_minimal.py" --steps 10 --batch 2 --offline --device "$1" --out "$ROOT/runs/dem_synthetic.pt"; }
if [ "$DEVICE" != "cpu" ]; then
  if PYTORCH_ENABLE_MPS_FALLBACK=1 train "$DEVICE"; then note PASS "10-step training on $DEVICE"
  else note WARN "training on $DEVICE failed (see log); use --device cpu locally, real training goes on lab GPUs"; DEVICE=cpu; fi
fi
[ "$DEVICE" = "cpu" ] && run "10-step training on cpu" train cpu
rm -f "$ROOT/runs/dem_synthetic.pt"   # 900 MB checkpoint of random weights, not worth keeping
run "inference example ($DEVICE)" env PYTORCH_ENABLE_MPS_FALLBACK=1 python "$ROOT/decoupled-embodiment-model/examples/inference.py" --offline --iters 5 --device "$DEVICE"

title "6. Hugging Face access (pretrained encoders)"
python - <<'EOF' ; HF_RC=$?
import sys
from huggingface_hub import hf_hub_download, whoami
rc = 0
try:
    print("logged in as:", whoami()["name"])
except Exception as e:
    print("not logged in to Hugging Face:", type(e).__name__); rc |= 1
try:
    hf_hub_download("chandar-lab/NeoBERT", "config.json"); print("NeoBERT (public): OK")
except Exception as e:
    print("NeoBERT: FAILED", type(e).__name__, e); rc |= 2
try:
    hf_hub_download("facebook/dinov3-convnext-base-pretrain-lvd1689m", "config.json"); print("DINOv3 ConvNeXt-B (gated): OK")
except Exception as e:
    print("DINOv3 ConvNeXt-B (gated): NO ACCESS YET ->", type(e).__name__); rc |= 4
sys.exit(rc)
EOF
if [ $((HF_RC & 2)) -eq 0 ]; then note PASS "download NeoBERT config"; else note FAIL "download NeoBERT (network?)"; fi
if [ $((HF_RC & 4)) -eq 0 ]; then note PASS "gated DINOv3 weights accessible"
else note TODO "DINOv3 is gated: accept terms at https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m then run: hf auth login"; fi

title "7. robosuite + RoboCasa in the same env"
cd "$ROOT" || exit 1
run "pip install -e robosuite" qpip -e "$ROOT/robosuite"
# --no-deps: robocasa pins lerobot==0.3.3, which would downgrade torch/transformers below what DEM needs.
run "pip install -e robocasa --no-deps" qpip -e "$ROOT/robocasa" --no-deps
run "robocasa runtime deps (pinned numpy/numba/scipy/mujoco)" qpip \
  "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
  pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio imageio-ffmpeg h5py lxml gymnasium
cd "$ROOT/runs" || exit 1
run "import dem + robosuite + robocasa together" python -c "
import numpy, torch, mujoco, dem, robosuite, robocasa
print('numpy', numpy.__version__, '| torch', torch.__version__, '| mujoco', mujoco.__version__, '| robosuite', robosuite.__version__)"
[ -f "$ROOT/robocasa/robocasa/macros_private.py" ] || python -m robocasa.scripts.setup_macros
cd "$ROOT/decoupled-embodiment-model" && run "DEM tests still pass after RoboCasa pins" python -m pytest -m "not slow" -q
python -m pip check 2>&1 | grep -v lerobot | grep -v tianshou | grep -v hidapi | sed 's/^/pip check: /'

title "8. RoboCasa assets and a simulator rollout"
ASSET_DIR="$ROOT/robocasa/robocasa/models/assets"
if [ "$WITH_ASSETS" = 1 ]; then
  cd "$ROOT/runs" || exit 1
  yes y | python -m robocasa.scripts.download_kitchen_assets
fi
if [ -d "$ASSET_DIR/objects/lightwheel" ] && [ -d "$ASSET_DIR/textures" ]; then
  cd "$ROOT/runs" || exit 1
  run "RoboCasa random rollout with offscreen render" python - <<'EOF'
import gymnasium as gym, numpy as np, robocasa
env = gym.make("robocasa/PickPlaceCounterToCabinet", split="pretrain", seed=0)
obs, info = env.reset()
print("obs keys:", sorted(obs.keys())[:12], "...")
for _ in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
imgs = {k: v.shape for k, v in obs.items() if hasattr(v, "shape") and len(v.shape) == 3}
print("camera observations:", imgs)
print("action space:", env.action_space)
env.close()
EOF
else
  note TODO "RoboCasa assets not downloaded yet (~10 GB): bash setup_dem.sh --assets"
fi

title "SUMMARY"
cat "$SUMMARY"
echo
echo "repos : $ROOT/{decoupled-embodiment-model,robosuite,robocasa}"
echo "env   : $ACTIVATE_HINT"
echo "log   : $LOG"
FAILS=$(grep -c '^FAIL' "$SUMMARY")
rm -f "$SUMMARY"
[ "$FAILS" = 0 ] && echo "RESULT: no failures" || echo "RESULT: $FAILS failure(s), see above"
