#!/bin/bash
# Build setup/rua.def into an Apptainer image on a compute node, then run the simulator test
# inside it with the GPU (the same test as `setup/setup_bouchet.sh simtest`).
#   cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness && sbatch setup/build_container.sbatch
# Rebuild over an existing image: REBUILD=1 sbatch setup/build_container.sbatch
# Result: $PROJ/containers/rua.sif and a log ending in CONTAINER SIMTEST OK.
# Written Nov 26, 2026. Not yet run.
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=build_container
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/build_container_%j.out
set -euo pipefail
PROJ=/nfs/roberts/project/pi_dr877/ak3288
SCR=/nfs/roberts/scratch/pi_dr877/ak3288
ROOT=$PROJ/apollo-lab
REPO=$ROOT/rua-harness
SIF=$PROJ/containers/rua.sif
mkdir -p $PROJ/containers $SCR/apptainer/cache $SCR/apptainer/tmp
# image layers and build files go to scratch, never home (500,000-file limit)
export APPTAINER_CACHEDIR=$SCR/apptainer/cache APPTAINER_TMPDIR=$SCR/apptainer/tmp
echo "host $(hostname) $(apptainer --version)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

cd $REPO   # %files in rua.def is relative to here
if [ ! -f $SIF ] || [ -n "${REBUILD:-}" ]; then
  t=$(date +%s)
  # no root on Bouchet: apptainer falls back to an unprivileged user-namespace build
  apptainer build --force $SIF setup/rua.def
  echo "built $SIF in $(( $(date +%s) - t )) s, size $(du -h $SIF | cut -f1)"
else
  echo "using existing $SIF (REBUILD=1 to rebuild)"
fi

RUN="apptainer exec --nv -B /nfs/roberts --env ROOT=$ROOT,HF_HUB_CACHE=$SCR/cache/hf $SIF"
cd $ROOT/runs   # not $ROOT: its robosuite/ folder would hide the package
$RUN python - <<'PY'
import numpy, torch, mujoco, dem, robosuite, robocasa
print("numpy", numpy.__version__, "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| mujoco", mujoco.__version__, "| robosuite", robosuite.__version__)
print("dem from", dem.__file__)
print("robocasa from", robocasa.__file__)
PY

# the simulator test from setup/setup_bouchet.sh simtest, unchanged
$RUN python - <<'PY'
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

# the repo's unit tests inside the image (no simulator needed)
cd $REPO && $RUN python -m pytest -q -m "not sim" tests | tail -3
echo "CONTAINER SIMTEST OK"
