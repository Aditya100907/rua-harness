#!/bin/bash
# Download LIBERO demo hdf5s from the Hugging Face mirror (yifengzhu-hf/LIBERO-datasets) to scratch.
#   SUITES="libero_spatial libero_object" sbatch policy/slurm/libero_download.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=libero_dl
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/libero_download_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/libero_common.sh
echo "host $(hostname)"
mkdir -p $LIBERO_DATA
for S in ${SUITES:-libero_spatial libero_object}; do
  python - "$S" <<'PY'
import sys, time, os
from huggingface_hub import snapshot_download
s, t = sys.argv[1], time.time()
p = snapshot_download(repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                      local_dir=os.environ["LIBERO_DATA"], allow_patterns=[f"{s}/*"])
files = sorted(os.listdir(os.path.join(p, s)))
print(s, len(files), "files", f"{time.time() - t:.0f}s")
PY
done
du -sh $LIBERO_DATA/*
ls $LIBERO_DATA/*/ | head -40
