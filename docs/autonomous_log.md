# Autonomous session log

Every autonomous Claude session (overnight runs, loops, routines) adds a section here and updates it every few minutes while it works: what it is trying, bugs it hits, how it solved them, and when and why it stopped or stalled. Newest session last. Times are ET. Plain facts; failures stay in.

---

## Session 2026-11-26 23:15 to now: overnight run (branch `overnight-nov26`, `docs/OVERNIGHT.md`)

### What was tried, in order
- 23:15 to 00:00: verified downloads; wrote the replay test, data loader, frame store, training and eval scripts; launched training runs 1 to 5 and the watcher. Helpers (subagents) did reading, paper scaffold, geometry and simlib, depth re-render, scripted baseline, reproducibility, RPent, LIBERO, HexaShell tool.
- 00:00 to 01:10: format variants, labels for 18 tasks, held-out split, commander (agent with look + run_skill), code review fixes, M test (watcher runs without SSH).
- 06:30 onward: first trained-policy numbers near zero; Part K investigation (open-loop, conditioning, overfitting tests, ablations).

### Bugs hit and how they were solved
| When | Bug | Fix |
|---|---|---|
| 23:20 | Bouchet kitchen assets incomplete (login-node download killed at 82%); simtest failed | re-download as a `day` CPU job |
| 23:40 | gpu_devel allows 1 submitted job; gpu_h200 allows 6 CPUs per GPU | speed test on gpu_h100; jobs use 6 CPUs |
| 23:45 | 10 eval processes on one GPU ran out of memory | 6 processes plus a retry pass |
| 23:50 | 2-day jobs would not start on H100 until morning | short runs on `gpu` (L40S/A40), long runs on any of H100/B200/H200 |
| 00:10 | watcher counted CPU jobs toward the 12-job cap; GPU count matched partition names by accident | count jobs in gpu* partitions |
| 00:22 | code review: eval shards double-count on requeue, norm stats used held-out demos, crash loop on resubmit, and 8 more | all fixed before any checkpoint was scored |
| 00:27 | points_3d had no point-map input | wired it, with a train/eval match test |
| 00:38 | Mac: loading the model after two env resets corrupts every later render | load the policy before building envs; image check column in eval |
| 05:00 to 06:30 | watcher read progress from the 1.3 GB train_state.pt, which failed on the 8 GB agent node, so it resubmitted all finished runs every 10 min (filled the GPU cap with 1-step jobs and repeat evals) | read progress from checkpoint file names; killed the extra jobs |
| 02:15 to 06:15 | YCRC cancelled an eval job for 0% GPU use and warned on another (eval is simulator-bound) | all scoring moved to CPU nodes (`day`) |
| 06:30 | stock DEM scores near 0% | investigation, still open (see below) |
| 07:10 | SSH master connection expired after 8 h; Duo needed; a `!` command cannot answer Duo | asked Adi to run `ssh bouchet true` in his own terminal |

### When the session stalled or stopped, and why
- **01:10 to 06:30 (about 5 h):** this session ended its turn while waiting for a helper agent and nothing woke it until the helper finished. Effect: near-zero numbers and the watcher bug went unnoticed for hours. Lesson: never end a turn with only a long helper pending; keep a short background timer running.
- **07:10 to 15:55 (about 9 h):** the Mac went to sleep, and SSH needed a new Duo approval. Nothing ran on the Mac; the cluster kept going. Lesson: on the Mac, run long sessions under `caffeinate`; ask for Duo before the 8-hour connection expires.

### Open problem right now
Stock DEM trained here does not learn usable actions (run 1 20k: 0% drawer and pick-place; run 2 at 70k: 0%). On training frames its actions are worse than the mean action and dominated by sampling noise; images barely used. Ruled out: eval input mismatch, renders, the one-step sampler. Testing: dispersive weight, batch size, clipping, learning rate, longer overfitting. Details: `docs/research_journal.md`.

### Running updates
- 16:00 Reconnected. Collected: run2 70k 0/24 each; run1 20k with zero noise 0/30 each, with 4 sampling steps 0, 2, 2 of 30. Ablation raw_mse means are spike-dominated, so conditioning checks submitted on all ablation checkpoints and run2 180k. Watcher kept off (its queue would launch more runs on the failing recipe).
- 16:02 Measured outliers in normalized targets: under 2.5% of frames per dim exceed 5 sigma (max 29), mostly pick-place base and mode dims. Probably not the main cause.
- 16:03 Started this log at Adi's request; added the rule to `CLAUDE.md`.
- 16:04 Started `caffeinate -i -t 43200` so the Mac stays awake for 12 h. Waiting on 6 conditioning checks (CPU jobs 34891984 to 34891989).
- 16:05 Conditioning checks: all four ablations at 5k are still at mean-action level (input spread 0.01 to 0.06 vs true 0.23); run2 180k only slightly better (0.117, MAE 0.137 vs 0.155). No single tweak helps. Next: test whether plain regression (mse) or standard flow matching (fm) learns quickly; if yes, MeanFlow optimization is the bottleneck.
- 16:07 Added --objective mse|fm; launched abl_mse and abl_fm (5k steps, 3 tasks).
- 16:08 Launched overfit1_mse and overfit1_meanflow (1 OpenDrawer demo, 2000 steps, batch 64) on Bouchet.
- 16:09 Chained CPU checks after each run (cond_check for abl_mse with zero noise via act z0, abl_fm with 10 steps; open-loop for both overfits, zero noise for mse). Waiting for GPUs.
- 16:10 Adi: match the DEM paper: scene + wrist cameras, quantiles from the dataset's shipped meta/stats.json (not recomputed); compare shipped vs recomputed on base dims; rerun the one-demo test.
- 16:12 Quantile comparison (shipped meta/stats.json vs recomputed over all frames), 3 smoke tasks: identical on every dim, max difference about 1e-5 (TurnOnMicrowave base x q99, interpolation rounding). Base ranges are tiny in both (base z 0.00012 to 0.0067, PickPlace yaw 0.02). Full table: docs/quantile_comparison.txt. So the quantile source is not the cause.
- 16:13 Added --quantiles dataset (shipped stats); launched overfit1_paper on Bouchet: 1 OpenDrawer demo, cameras left + wrist, shipped quantiles, DEM recipe (meanflow, head lr 1e-4, dispersive 0.25), 2000 steps, batch 64 (job 34892487); open-loop checks chained with random and zero noise. Note: the dataset has no center camera; "scene" = left agent view. abl_mse, abl_fm, overfit1_mse are running now.
- 16:17 Correction: earlier timestamps this afternoon were written ahead of the real clock; fixed to approximate real times. From now on every entry uses `date`. Early results: overfit1_mse memorizes the demo (normalized MSE 0.0019 at 2000 steps); abl_mse at 2600 steps 0.55; abl_fm flow error 0.40 at 1920 steps vs MeanFlow about 1.1. Points to MeanFlow optimization as the bottleneck; waiting for the checks.
- 16:21 Mac one-demo MeanFlow test (head lr 3e-4, no dispersive, batch 8, 600 steps; the Mac slept through most of it): flow error stays about 0.7 to 1.0; open-loop arm MAE 0.211 vs 0.157 for the mean action. MeanFlow still cannot memorize one demo, with or without the dispersive term, while plain regression memorizes it on Bouchet (0.0019).
- 16:24 New priorities from Adi. Cancelled run2_all18 (STOP file set so nothing resubmits it). Pushed the branch. Gate evals chained: abl_mse 5k with zero noise, abl_fm 5k with 10 sampling steps, 50 episodes per smoke task (gate: either beats 20% on OpenDrawer or TurnOnMicrowave).
- 16:28 Bug: train.sbatch exited 1 after a finished run (my crash-loop fix), so Slurm marked abl_mse and overfit1_mse FAILED and cancelled the afterok-chained checks and the mse gate eval. Fixed (exit 0) and re-chained all checks with afterany; mse gate eval resubmitted (34893757). DDP 2-GPU test passed (job 34893602). Research-weights converter and --init-from done and tested on fake NeoBERT files; mmBERT test running on CPU (34893756).
- 16:28 Priority 4: launched mf_ddp4_bs256 (MeanFlow, 4 H100s, 64 per GPU = global 256, 3 smoke tasks, 5k steps; only change vs run1_smoke 5k is the global batch). cond_check chained.
- 16:28 Moved mf_ddp4_bs256 to gpu_b200 or gpu_h100 (B200 starts about 16:28, H100 not until 20:17).
- 16:29 STATUS.md rewritten for the new priorities. Depth re-render complete for all 9,126 demos; spot check submitted (34893778). Eval now follows each checkpoint's objective (mse: zero noise, fm: 10 steps). Format-study queue script ready (policy/slurm/format_study_queue.sh).
- 16:39 Checks done (see journal 16:40): regression memorizes 1 demo (MAE 0.009); paper-matched MeanFlow 1-demo 0.095; MeanFlow left+right lr 3e-4 0.21; abl_mse 0.128 and abl_fm 0.145 vs mean 0.155. mmBERT converter test passed. Gate evals running (mse and fm, 50 eps x 3 tasks).
- 16:42 Gate at 5k: abl_mse TurnOnMicrowave 0/50 (others still running). Launched fs_mse_sentence and fs_fm_sentence: 20k steps with the exact format-study settings (left + wrist, shipped quantiles, held-out-50 split, seed 0). They are the gate at 20k and the sentence arm of the format study for whichever objective wins.
- 16:43 Watcher restarted (CPU scoring, fixed progress reading) with an empty queue; all existing checkpoints marked scored. It will score fs_mse_sentence and fs_fm_sentence every 5k steps (50 eps; 100 at 20k).
- 16:45 Adi: add a target split to HexaShell's RoboCasa port matching our eval kitchens, objects and seeding, and stand up the SAM 3 service on Bouchet once access lands. Handed to a helper (pinned clone at 914d482, change kept as setup/patches/hexashell_target_split.patch, recorded in docs/hexashell_changes.md; scene-by-scene comparison against policy/eval.py on 3 tasks x 5 seeds; SAM 3 env, job script and smoke test ready, access checked read-only).
- 16:50 GATE PASSED: abl_mse 5k OpenDrawer 22/50 (44%), PickPlace 0/50, Microwave 0/50. Sanity-checked (rows, renders, success lengths, video). Waiting for abl_fm's drawer and pick-place before choosing the objective.
- 16:55 HexaShell 914d482 cloned (Mac and Bouchet, with submodules). Added a target split (setup/patches/hexashell_target_split.patch, docs/hexashell_changes.md) and policy/hexashell_target_check.py. Mac check (our env, 425684d): 12-13/15 scenes match; seed 1003 changes layout between two runs of our own eval (counter.py set order); with HexaShell's determinism fix monkeypatched, ours matches the target split 15/15. SAM 3: HF account adikothar1 access to facebook/sam3 still 'awaiting a review'.
- 16:55 fm gate: 0/50 on all three. Chose mse. Wrote the mse format-study queue and restarted the watcher: 5 formats + 18-task sentence launched (8/8 GPUs). Watcher now counts GPUs per job (the 4-GPU test counted as 1 before).
- 17:00 Batch-size test answered: global batch 256 does not make MeanFlow beat the mean action at 5k (raw_mse median 0.80 vs 0.89; MAE 0.152 vs mean 0.155). No further MeanFlow runs. HexaShell helper is setting up envs on Bouchet (hx_setup, sam3_setup jobs).
