#!/bin/bash
# Priority 3 (Nov 27): write the format-study queue for the chosen objective, replacing the old queue.
#   bash policy/slurm/format_study_queue.sh fm      (or mse)
# 3 smoke tasks at 20k steps, same seed and data (the 452/464/493-demo split that holds out the last 50
# demos per task, which pose_path needs for evaluation), cameras left + wrist, shipped quantiles.
# Then the 18-task stock run in the sentence format.
OBJ=$1
R=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/runs
Q=$R/queue.txt
[ -f $Q ] && mv $Q $R/queue_before_format_study_$(date +%Y%m%d_%H%M).txt
COMMON="--tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave --steps 20000 --batch 64 --save-every 2500 --seed 0 --cams left wrist --quantiles dataset --objective $OBJ --episodes-json $R/smoke_train_minus50.json"
{
  echo "# format study, objective $OBJ, written $(date)"
  for f in sentence kv_text points_2d points_3d_rgb points_3d pose_path; do
    echo "fs_${OBJ}_${f} | --format $f $COMMON"
  done
  echo "fs_${OBJ}_sentence_all18 | --format sentence --all-tasks --steps 300000 --batch 64 --save-every 10000 --seed 0 --cams left wrist --quantiles dataset --objective $OBJ"
} > $Q
cat $Q
