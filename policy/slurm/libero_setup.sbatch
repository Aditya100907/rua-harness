#!/bin/bash
# Build the LIBERO env (own conda env, isolated from envs/dem), write LIBERO's config, run one
# random-action episode headless.   sbatch policy/slurm/libero_setup.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=libero_setup
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/libero_setup_%j.out
set -e
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/libero_common.sh
echo "host $(hostname)"
if [ -n "${SKIP_INSTALL:-}" ]; then conda activate $LIBERO_ENV; else
if [ ! -x $LIBERO_ENV/bin/python ]; then
  conda create -y -p $LIBERO_ENV -c conda-forge --override-channels python=3.11
fi
conda activate $LIBERO_ENV
python -m pip install -q --upgrade pip
# LIBERO's requirements.txt pins robosuite 1.4.0, numpy 1.22, torch-era packages. We keep robosuite 1.4.x
# and MuJoCo 2.3.x (what LIBERO was built on) and only the packages the env and data need.
python -m pip install -q "robosuite==1.4.1" "mujoco==2.3.7" "numpy==1.26.4" "bddl==1.0.1" "easydict==1.9" \
  "gym==0.25.2" "cloudpickle==2.1.0" "future" "hydra-core>=1.3" "matplotlib" h5py imageio imageio-ffmpeg pyyaml \
  "opencv-python-headless<4.11" pandas pytest "huggingface_hub>=1.0" scipy
python -m pip install -q -e $ROOT/LIBERO --no-deps
# the editable install does not expose the top-level `libero` package (no __init__.py there), so add the
# clone to the path directly
echo "$ROOT/LIBERO" > $(python -c "import site; print(site.getsitepackages()[0])")/libero_clone.pth
# DEM in the same env so evaluation can run policy and simulator in one process
python -m pip install -q "torch==2.15.0"
python -m pip install -q -e $ROOT/decoupled-embodiment-model
python -m pip install -q "numpy==1.26.4"   # in case a dependency upgraded it
fi
# LIBERO asks an interactive question on first import unless config.yaml exists
mkdir -p $LIBERO_CONFIG_PATH $LIBERO_DATA
B=$ROOT/LIBERO/libero/libero
cat > $LIBERO_CONFIG_PATH/config.yaml <<CFG
assets: $B/assets
bddl_files: $B/bddl_files
benchmark_root: $B
datasets: $LIBERO_DATA
init_states: $B/init_files
CFG
python - <<'PY'
import numpy, mujoco, robosuite, torch, h5py
print("numpy", numpy.__version__, "| mujoco", mujoco.__version__, "| robosuite", robosuite.__version__,
      "| torch", torch.__version__, "| h5py", h5py.__version__)
import dem; print("dem import ok")
PY
pip freeze > $LIBERO_ENV/pip_freeze.txt
python -m policy.libero.random_episode --suite libero_spatial --task 0 --steps 50
