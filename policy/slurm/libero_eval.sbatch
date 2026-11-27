#!/bin/bash
# Evaluate a LIBERO DEM checkpoint (GPU for the policy; rendering on EGL).
#   CKPT=$PROJ/checkpoints/libero_spatial_dem/ckpt_0030000.pt SUITE=libero_spatial sbatch policy/slurm/libero_eval.sbatch
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=libero_eval
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/libero_eval_%j.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/libero_common.sh
python -m policy.libero.eval --ckpt $CKPT --suite ${SUITE:-libero_spatial} --episodes ${EPISODES:-50} \
  --out $REPO/results/$(date +%F)_libero_${SUITE:-libero_spatial}_$(basename $(dirname $CKPT))_$(basename $CKPT .pt).csv \
  2>&1 | grep -v -i warning
