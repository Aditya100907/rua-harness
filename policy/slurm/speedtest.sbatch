#!/bin/bash
# C2: 300 training steps on the smoke tasks, logging GPU utilization.
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --job-name=speedtest
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/speedtest_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$SCR/speedtest_$SLURM_JOB_ID
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 5 > $OUT.gpu.csv &
SMI=$!
python -m policy.train --tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave --steps ${STEPS:-300} \
  --batch ${BATCH:-64} --workers ${WORKERS:-5} --log-every 20 --save-every 100000 --out $OUT
kill $SMI
echo "GPU util (5 s samples, after the first minute):"
tail -n +13 $OUT.gpu.csv | awk -F', ' '{gsub(/ %/,"",$1); s+=$1; n++} END {print "mean util", s/n, "% over", n, "samples"}'
python - <<PY
import torch; from dem import DEM
p = DEM.load("$OUT/ckpt_$(printf %07d ${STEPS:-300}).pt"); print("RELOAD OK", sum(x.numel() for x in p.parameters()))
PY
