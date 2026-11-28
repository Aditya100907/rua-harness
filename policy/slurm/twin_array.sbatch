#!/bin/bash
# CPU job array: twin-distractor data for PickPlaceCounterToCabinet (policy/twin_data.py).
# One array index = CHUNK demos. 502 demos / 10 = 51 indices (0..50); the last chunk is short.
#   sbatch --array=0-50%40 policy/slurm/twin_array.sbatch
# Output: $SCR/frames/PickPlaceCounterToCabinetTwin/parts/episode_NNNNNN.npz plus build_log.jsonl.
# Demos already on disk are skipped, so resubmitting resumes. Then merge once (CPU, a few minutes):
#   sbatch --wrap "source $PROJ/apollo-lab/rua-harness/policy/slurm/common.sh; python -m policy.twin_data merge" \
#     -p day -c 2 --mem 8G -t 01:00:00 -J twin_merge -o $PROJ/apollo-lab/logs/twin_merge_%j.out
# Rendering on day nodes: MUJOCO_GL=egl through Mesa llvmpipe (see rerender_array.sbatch).
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --job-name=twin
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/twin_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CHUNK=${CHUNK:-10}
echo "index $SLURM_ARRAY_TASK_ID chunk size $CHUNK on $(hostname) commit $(cat $REPO/COMMIT 2>/dev/null)"
python -m policy.twin_data build --chunk $SLURM_ARRAY_TASK_ID --chunk-size $CHUNK
