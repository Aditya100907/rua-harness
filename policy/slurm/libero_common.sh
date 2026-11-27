# Sourced by LIBERO jobs: the separate libero env (robosuite 1.4.1, MuJoCo 2.3.7), paths, headless rendering.
module load miniconda >/dev/null 2>&1
source "$(conda info --base)/etc/profile.d/conda.sh"
export PROJ=/nfs/roberts/project/pi_dr877/ak3288 SCR=/nfs/roberts/scratch/pi_dr877/ak3288
export ROOT=$PROJ/apollo-lab REPO=$PROJ/apollo-lab/rua-harness LIBERO_ENV=$PROJ/envs/libero
conda activate $LIBERO_ENV 2>/dev/null || true
export LIBERO_CONFIG_PATH=$PROJ/libero_config LIBERO_DATA=$SCR/datasets/libero
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_CACHE=$SCR/cache/hf PIP_CACHE_DIR=$SCR/cache/pip
export CONDA_PKGS_DIRS=$SCR/cache/conda_pkgs TMPDIR=$SCR/cache/tmp PYTHONPATH=$REPO
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p $ROOT/runs/libero
cd $ROOT/runs/libero
