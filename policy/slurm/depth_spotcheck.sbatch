#!/bin/bash
# F2 test: on 20 random rendered demos per task, the point map at the projected hindsight grasp
# point (F1 labels, $SCR/labels/<task>.jsonl) must be within 5 cm of it (best camera, 5-pixel
# window) on >= 90% of demos. Reads files only, no simulator. Run after rerender_array.sbatch:
#   sbatch --dependency=afterany:<rerender job id> policy/slurm/depth_spotcheck.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=depthcheck
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/depthcheck_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
export RUA_DEPTH=$SCR/depth
python -m policy.check_depth --all-tasks --n ${N:-20} --labels ${LABELS:-$SCR/labels} \
  --out $ROOT/runs/depth_spotcheck/spotcheck_$(date +%Y-%m-%d).json
