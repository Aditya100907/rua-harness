# Experiment registry

Every run gets a row before it runs, with its prediction. Negative results stay. Checkpoints live in
`/nfs/roberts/project/pi_dr877/ak3288/checkpoints/<run>/` on Bouchet; results CSVs in `results/`.
Scoring: the watcher (`policy/slurm/watcher.sh`) evaluates every 5k steps for 3-task runs (50 episodes per task, 100 for the final checkpoint) and every 50k plus final for 18-task runs.

| Run | Job | Config | Purpose | Prediction | Status | Results |
|---|---|---|---|---|---|---|
| untrained_smoke | 34841791 | random head, pretrained towers, 3 smoke tasks | D1 zero test | about 0% | done: 0/28 | `results/2026-11-26_policy_only_zero_test_*.csv` |
| wiring | 34841796 | demo actions through the eval loop | D1 wiring test | 10/10 per task | done: 30/30 | `results/2026-11-26_policy_only_wiring_*.csv` |
| run1_smoke | 34842801 | stock DEM (sentence), left+right, 3 smoke tasks, 20k steps, batch 64, seed 0, L40S/A40 | D3 first real numbers; control for F4 | drawer and button well above 0 by 20k; pick-place low (paper category numbers: drawers 79.5%, buttons 81.5%, pick-place 33.2% with full 18-task training) | running | |
| run2_all18 | 34842802 | stock DEM, 18 tasks, 300k steps, batch 64, seed 0, H100/B200/H200 | U1: reproduce the paper's 55.6% | 45 to 60% at 300k | running | |
| run3_smoke_seed1 | 34842803 | run1 with seed 1 | G2 noise estimate | within about 10 points of run1 per task | running | |
| run4_smoke_seed2 | 34842804 | run1 with seed 2 | G2 noise estimate | within about 10 points of run1 per task | running | |
| run5_smoke_leftwrist | 34842805 | run1 with cameras left + wrist | does the wrist view help (DEM README example uses scene + wrist) | better on pick-place, similar on drawer/button | running | |
| f4_kv_text | queued | kv_text command text, color, 3 tasks, 20k, seed 0 | F4 format study | about equal to run1 (the text carries the same names) | queued | |
| f4_points_3d_rgb | queued | kv_text + 2 point tokens, color | F4 | better than run1 on pick-place if the policy uses the point | queued | |
| f4_points_2d | queued | kv_text + points drawn on the images | F4 | similar to points_3d_rgb | queued | |
| f4_pose_path | queued | kv_text + 8 future gripper poses (hindsight, train noise 2 cm / 3 deg) | F4 | highest with perfect paths; evaluated only on demo scenes | queued | |
