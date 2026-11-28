#!/bin/bash
# vLLM env on Bouchet for the open vision-language commander (docs/vlm_commander.md). Run as a CPU job:
#
#   sbatch --export=ALL,CMD="bash setup/setup_vllm_bouchet.sh" policy/slurm/cpu_job.sbatch
#
# Builds $PROJ/envs/vllm (conda Python 3.12, vllm pinned below, which brings its own CUDA 12.8 torch) and
# downloads the model weights to the Hugging Face cache on scratch. Idempotent.
set -euo pipefail
PROJ=/nfs/roberts/project/pi_dr877/ak3288
SCR=/nfs/roberts/scratch/pi_dr877/ak3288
E=$PROJ/envs/vllm
VLLM_VERSION=${VLLM_VERSION:-0.11.0}       # Oct 2025; first release line with Qwen3-VL support
MODELS=${MODELS:-"Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct"}
export PIP_CACHE_DIR=$SCR/cache/pip CONDA_PKGS_DIRS=$PROJ/conda_pkgs TMPDIR=$SCR/cache/tmp HF_HUB_CACHE=$SCR/cache/hf
mkdir -p "$TMPDIR"
module load miniconda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -x "$E/bin/python" ] || conda create -y -q -p "$E" python=3.12
"$E/bin/python" -m pip install -q "vllm==$VLLM_VERSION" "transformers==4.57.1" huggingface_hub   # vllm 0.11 predates transformers 5
"$E/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
for m in $MODELS; do
    "$E/bin/python" - "$m" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))
EOF
done
du -sh "$HF_HUB_CACHE"/models--Qwen--* || true
