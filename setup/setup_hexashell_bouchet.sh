#!/bin/bash
# HexaShell (914d482) on Bouchet: its RoboCasa simulator env and the SAM 3 env. Run as a CPU batch job
# (the pip installs are too big for a login node):
#
#   sbatch --export=ALL,CMD="bash setup/setup_hexashell_bouchet.sh hexashell" policy/slurm/cpu_job.sbatch
#   sbatch --export=ALL,CMD="bash setup/setup_hexashell_bouchet.sh sam3" policy/slurm/cpu_job.sbatch
#
# hexashell: applies setup/patches/hexashell_target_split.patch, builds a CPython 3.12 (conda, project space)
#            and runs HexaShell's own scripts/setup/setup_robocasa.sh with VENV in project space and ASSETS =
#            our RoboCasa assets (no second 25 GB download).
# sam3:      a conda env (Python 3.11) with what HexaShell's README asks for: torch 2.7.0, torchvision 0.22.0
#            (CUDA 12.8 wheels, so B200 works), timm, sam3 from its repo, einops, pycocotools, psutil,
#            setuptools<81. No weights: those are gated (docs/sam3_service.md).
# Caches go to scratch. Idempotent.
set -euo pipefail
PROJ=/nfs/roberts/project/pi_dr877/ak3288
SCR=/nfs/roberts/scratch/pi_dr877/ak3288
HX=$PROJ/apollo-lab/HexaShell
RUA=$PROJ/apollo-lab/rua-harness
export PIP_CACHE_DIR=$SCR/cache/pip CONDA_PKGS_DIRS=$PROJ/conda_pkgs TMPDIR=$SCR/cache/tmp
mkdir -p "$TMPDIR"
SAM3_COMMIT=31cb7e624e4fe95301b1be1d4e809111593d2e45   # facebookresearch/sam3 main on Nov 27 2026

module load miniconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

case "${1:-}" in
hexashell)
    [ -d "$HX/.git" ] || { git clone https://github.com/HexaFuture/HexaShell "$HX"; git -C "$HX" checkout -q 914d482; }
    git -C "$HX" submodule update --init third_party/robocasa third_party/robosuite
    bash "$RUA/setup/patches/apply_hexashell_target_split.sh" "$HX"
    [ -x "$PROJ/envs/py312/bin/python3.12" ] || conda create -y -q -p "$PROJ/envs/py312" python=3.12
    cd "$HX"
    PY=$PROJ/envs/py312/bin/python3.12 VENV=$PROJ/envs/hexashell-robocasa \
        ASSETS=$PROJ/apollo-lab/robocasa/robocasa/models/assets bash scripts/setup/setup_robocasa.sh
    git -C "$HX/third_party/robocasa" log -1 --format='robocasa %h' ;;
sam3)
    E=$PROJ/envs/sam3
    [ -x "$E/bin/python" ] || conda create -y -q -p "$E" python=3.11
    "$E/bin/python" -m pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    "$E/bin/python" -m pip install -q timm einops pycocotools psutil "setuptools<81" "numpy<2" pillow huggingface_hub
    "$E/bin/python" -m pip install -q "git+https://github.com/facebookresearch/sam3@$SAM3_COMMIT"
    "$E/bin/python" -c "import torch, torchvision, sam3, timm; from sam3.model_builder import build_sam3_image_model
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__, 'timm', timm.__version__, 'sam3 ok')" ;;
*) sed -n 2,16p "$0"; exit 2 ;;
esac
