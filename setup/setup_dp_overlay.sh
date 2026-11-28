#!/bin/bash
# Diffusion Policy baseline: LeRobot 0.4.4 in an overlay folder next to envs/dem (envs/dem is not changed).
# `pip install --no-deps --target` puts only these packages in the folder; policy/diffusion_policy.py appends it
# to sys.path, so envs/dem's torch, numpy, huggingface_hub, transformers keep priority. LeRobot's own pins
# (torch<2.11, huggingface_hub<0.36, ...) would otherwise downgrade envs/dem.
#   Mac:     conda activate dem && bash setup/setup_dp_overlay.sh                 (~/apollo-lab/envs/dp_overlay)
#   Bouchet: sbatch -p day -c 2 --mem=8G -t 1:00:00 --wrap "bash $REPO/setup/setup_dp_overlay.sh"
set -euo pipefail
if [ -d /nfs/roberts/project/pi_dr877/ak3288 ]; then
  source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
  O=${RUA_DP_OVERLAY:-/nfs/roberts/project/pi_dr877/ak3288/envs/dp_overlay}
  cd $REPO
else
  O=${RUA_DP_OVERLAY:-$HOME/apollo-lab/envs/dp_overlay}
  cd "$(dirname "$0")/.."
fi
PKGS="lerobot==0.4.4 diffusers==0.35.2 einops==0.8.1 draccus==0.10.0 typing-inspect==0.9.0 mypy-extensions==1.1.0
mergedeep==1.3.4 pyyaml-include==1.4.1 toml==0.10.2 importlib-metadata==8.7.0 zipp==3.23.0 datasets==4.4.1
dill==0.4.0 multiprocess==0.70.18 xxhash==3.6.0 requests==2.32.5 urllib3==2.5.0 charset-normalizer==3.4.4
certifi==2025.10.5 httpx==0.28.1 httpcore==1.0.9 sniffio==1.3.1 accelerate==1.11.0 pyserial==3.5
deepdiff==8.6.1 orderly-set==5.5.0 cachebox==6.2.11"
python -m pip install --no-deps --upgrade --target "$O" $PKGS
RUA_DP_OVERLAY=$O PYTHONPATH=. python -c "
from policy.diffusion_policy import import_lerobot_dp, lerobot_version, DPPolicy
import_lerobot_dp(); m = DPPolicy(['a'], ['left', 'wrist'])
import torch, numpy, huggingface_hub
print('DP_OVERLAY_OK lerobot', lerobot_version(), 'torch', torch.__version__, 'numpy', numpy.__version__,
      'hub', huggingface_hub.__version__, 'params', sum(p.numel() for p in m.parameters()))"
