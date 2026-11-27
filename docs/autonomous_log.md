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
- 16:05 Reconnected. Collected: run2 70k 0/24 each; run1 20k with zero noise 0/30 each, with 4 sampling steps 0, 2, 2 of 30. Ablation raw_mse means are spike-dominated, so conditioning checks submitted on all ablation checkpoints and run2 180k. Watcher kept off (its queue would launch more runs on the failing recipe).
- 16:10 Measured outliers in normalized targets: under 2.5% of frames per dim exceed 5 sigma (max 29), mostly pick-place base and mode dims. Probably not the main cause.
- 16:12 Started this log at Adi's request; added the rule to `CLAUDE.md`.
- 16:15 Started `caffeinate -i -t 43200` so the Mac stays awake for 12 h. Waiting on 6 conditioning checks (CPU jobs 34891984 to 34891989).
- 16:20 Conditioning checks: all four ablations at 5k are still at mean-action level (input spread 0.01 to 0.06 vs true 0.23); run2 180k only slightly better (0.117, MAE 0.137 vs 0.155). No single tweak helps. Next: test whether plain regression (mse) or standard flow matching (fm) learns quickly; if yes, MeanFlow optimization is the bottleneck.
- 16:30 Added --objective mse|fm; launched abl_mse and abl_fm (5k steps, 3 tasks).
- 16:33 Launched overfit1_mse and overfit1_meanflow (1 OpenDrawer demo, 2000 steps, batch 64) on Bouchet.
- 16:38 Chained CPU checks after each run (cond_check for abl_mse with zero noise via act z0, abl_fm with 10 steps; open-loop for both overfits, zero noise for mse). Waiting for GPUs.
- 16:45 Adi: match the DEM paper: scene + wrist cameras, quantiles from the dataset's shipped meta/stats.json (not recomputed); compare shipped vs recomputed on base dims; rerun the one-demo test.
