#!/bin/bash
# Build the project on Yale's Bouchet cluster. Run on a login node.
#
#   bash setup/setup_bouchet.sh env      # clones, Python environment, installs, import checks (about 15 min)
#   bash setup/setup_bouchet.sh assets   # RoboCasa kitchen assets, in the background (large)
#   bash setup/setup_bouchet.sh data     # demos for all 18 atomic-seen tasks, in the background (large)
#   bash setup/setup_bouchet.sh simtest  # submit a GPU job that renders one RoboCasa episode
#
# Layout:
#   project (backed up, never purged):  code, environment, checkpoints worth keeping
#   scratch (purged after 30 days):     demo datasets, caches
# Written Nov 26, 2026. Not yet run on Bouchet when written.

set -u
PROJ="${PROJ:-/nfs/roberts/project/pi_dr877/$USER}"
SCR="${SCR:-/nfs/roberts/scratch/pi_dr877/$USER}"
ROOT="$PROJ/apollo-lab"
ENVDIR="$PROJ/envs/dem"
LOGDIR="$ROOT/logs"
mkdir -p "$ROOT" "$LOGDIR" "$SCR/datasets" "$SCR/cache/pip" "$SCR/cache/hf" "$PROJ/conda_pkgs"

# keep caches out of home (home has a 500,000-file limit)
export PIP_CACHE_DIR="$SCR/cache/pip"
export HF_HOME="$SCR/cache/hf"
export CONDA_PKGS_DIRS="$PROJ/conda_pkgs"

activate() {
  module load miniconda
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENVDIR"
}

clone_at() {  # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2" || return 1
  [ -z "${3:-}" ] || git -C "$2" checkout -q "$3"
  echo "$(basename "$2") @ $(git -C "$2" log -1 --format='%h %ad' --date=short)"
}

case "${1:-}" in
env)
  set -e
  echo "== clones into $ROOT"
  clone_at https://github.com/Apollo-Lab-Yale/decoupled-embodiment-model.git "$ROOT/decoupled-embodiment-model" ""
  clone_at https://github.com/ARISE-Initiative/robosuite.git "$ROOT/robosuite" 5ce6643
  clone_at https://github.com/robocasa/robocasa.git "$ROOT/robocasa" 425684d

  echo "== environment at $ENVDIR"
  module load miniconda
  [ -d "$ENVDIR" ] || conda create -y -p "$ENVDIR" -c conda-forge --override-channels python=3.11
  activate
  python -m pip install -q --upgrade pip

  echo "== installs (quiet, a few minutes)"
  cd "$ROOT/decoupled-embodiment-model" && python -m pip install -q -e ".[dev,vit]"
  python -m pip install -q -e "$ROOT/robosuite"
  # --no-deps: robocasa's own pins would downgrade torch and transformers below what DEM needs
  python -m pip install -q -e "$ROOT/robocasa" --no-deps
  python -m pip install -q "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
    pygame Pillow opencv-python-headless pyyaml pynput tqdm termcolor imageio imageio-ffmpeg h5py lxml gymnasium pyarrow pandas av

  echo "== robocasa macros, datasets go to scratch"
  mkdir -p "$ROOT/runs" && cd "$ROOT/runs"
  [ -f "$ROOT/robocasa/robocasa/macros_private.py" ] || python -m robocasa.scripts.setup_macros
  MP="$ROOT/robocasa/robocasa/macros_private.py"
  if grep -q '^DATASET_BASE_PATH' "$MP"; then
    sed -i "s|^DATASET_BASE_PATH.*|DATASET_BASE_PATH = \"$SCR/datasets\"|" "$MP"
  else
    echo "DATASET_BASE_PATH = \"$SCR/datasets\"" >> "$MP"
  fi
  grep -n '^DATASET_BASE_PATH' "$MP"

  echo "== checks"
  MUJOCO_GL=egl python -c "
import numpy, torch, mujoco, dem, robosuite, robocasa
print('numpy', numpy.__version__, '| torch', torch.__version__, '| cuda build', torch.version.cuda, '| mujoco', mujoco.__version__)"
  cd "$ROOT/decoupled-embodiment-model" && python -m pytest -m "not slow" -q | tail -2
  echo
  echo "DONE. Next:"
  echo "  1. hf auth login            (once, for the gated DINOv3 weights; run after: module load miniconda; conda activate $ENVDIR)"
  echo "  2. bash setup/setup_bouchet.sh assets"
  echo "  3. bash setup/setup_bouchet.sh data"
  ;;
assets)
  activate; cd "$ROOT/runs"
  nohup bash -c "yes y | python -m robocasa.scripts.download_kitchen_assets" > "$LOGDIR/assets_download.log" 2>&1 &
  echo "started in background, pid $!. Watch: tail -f $LOGDIR/assets_download.log"
  ;;
data)
  activate; cd "$ROOT/runs"
  TASKS="CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet"
  nohup bash -c "yes y | python -m robocasa.scripts.download_datasets --split target --source human --tasks $TASKS" > "$LOGDIR/data_download.log" 2>&1 &
  echo "started in background, pid $!. Watch: tail -f $LOGDIR/data_download.log"
  echo "Data lands under $SCR/datasets"
  ;;
simtest)
  cat > "$ROOT/runs/simtest.sh" <<JOB
#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=$LOGDIR/simtest_%j.out
module load miniconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENVDIR"
export MUJOCO_GL=egl HF_HOME="$SCR/cache/hf"
cd "$ROOT/runs"
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'PY'
import time, torch, gymnasium as gym, robocasa
print("torch sees GPU:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
t = time.time()
env = gym.make("robocasa/PickPlaceCounterToCabinet", split="target", seed=0)
obs, info = env.reset()
for _ in range(20):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
print("cameras:", {k: v.shape for k, v in obs.items() if hasattr(v, "shape") and len(v.shape) == 3})
print("action space:", env.action_space)
print("20 steps incl. reset: %.1f s" % (time.time() - t))
env.close()
print("SIMTEST OK")
PY
JOB
  sbatch "$ROOT/runs/simtest.sh"
  echo "When it finishes: cat $LOGDIR/simtest_*.out"
  ;;
*)
  echo "usage: bash setup/setup_bouchet.sh {env|assets|data|simtest}"
  ;;
esac
