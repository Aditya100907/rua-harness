#!/bin/bash
# Submit a training run, picking partitions by length. Prints the job id.
#   bash policy/slurm/launch_train.sh RUN "train.py args"
# Short runs (<= 30k steps): gpu partition, L40S or A40 (48 GB), 8 h, resubmits itself if needed.
# Long runs: whichever of gpu_h100, gpu_b200, gpu_h200 starts first, 2 days, resubmits itself.
RUN=$1; ARGS=$2
REPO=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
STEPS=$(echo "$ARGS" | grep -o -- '--steps [0-9]*' | awk '{print $2}'); STEPS=${STEPS:-20000}
mkdir -p /nfs/roberts/project/pi_dr877/ak3288/checkpoints/$RUN
echo "$ARGS" > /nfs/roberts/project/pi_dr877/ak3288/checkpoints/$RUN/args.txt
if [ "$STEPS" -le 30000 ]; then
  RUN=$RUN ARGS="$ARGS" PARTS=gpu CONSTRAINT="gpu:l40s|gpu:a40" sbatch --parsable --job-name=$RUN \
    --partition=gpu --constraint="gpu:l40s|gpu:a40" --time=08:00:00 $REPO/policy/slurm/train.sbatch
else
  RUN=$RUN ARGS="$ARGS" PARTS=gpu_h100,gpu_b200,gpu_h200 sbatch --parsable --job-name=$RUN \
    --partition=gpu_h100,gpu_b200,gpu_h200 $REPO/policy/slurm/train.sbatch
fi
