#!/bin/bash
# LIBERO replay test (CPU, no rendering) plus a rendered wiring check through the eval loop.
#   SUITES="libero_spatial libero_object" TASKS="0 1 2" N=10 sbatch policy/slurm/libero_replay.sbatch
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=libero_replay
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/libero_replay_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/libero_common.sh
echo "host $(hostname)"
python -m policy.libero.replay_test --suites ${SUITES:-libero_spatial libero_object} --tasks ${TASKS:-0 1 2} \
  --n ${N:-10} --extra-steps ${EXTRA:-0} --out $ROOT/runs/libero/replay_${SLURM_JOB_ID}.csv 2>&1 | grep -v -i warning
if [ -z "${NO_WIRING:-}" ]; then
  python -m policy.libero.eval --wiring --suite ${WIRING_SUITE:-libero_spatial} --tasks ${WIRING_TASKS:-0} --episodes ${WIRING_N:-3} --video 1 --device cpu \
    --out $ROOT/runs/libero/wiring_${SLURM_JOB_ID}.csv 2>&1 | grep -v -i warning
fi
