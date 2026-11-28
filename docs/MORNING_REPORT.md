# Morning report, Fri Nov 27

Last update: Sat Nov 28, 13:34 ET.

## 1. Bottom line (updated Sat Nov 28, 13:34)
DEM's MeanFlow recipe did not learn here, but the same model trained by plain regression works: 51.2% over all 18 tasks at 300k steps (paper: 55.6%), and the format study on 3 tasks now has 3-seed results. Drawing the grasp and place points on the camera images raises success on the drawer (65% vs 53%) and microwave (56% vs 44%) over the plain sentence, while structured text alone and learned 3D point tokens do not help. But the policy does not follow a point to a different object and collapses when the object name is removed: it relies on the text, so "points steer the policy" is not supported yet; a name-dropout test of that is running.

## 2. Numbers
**18-task stock DEM (sentence, regression, left + wrist, 300k steps, one seed, 100 episodes per task):** 51.2% (921/1800). Curve: 37.0% at 50k, 40.8% at 100k, 47.8% at 150k, 47.9% at 200k, 49.8% at 250k, 51.2% at 300k. Best: OpenStandMixerHead 88%, PickPlaceToasterToCounter 81%, OpenCabinet 78%, TurnOnElectricKettle 78%. Worst: NavigateKitchen 0%, CloseBlenderLid 10%, PickPlaceCounterToStove 16%.

**Format study (3 smoke tasks, regression, 20k steps, 100 episodes per seed per task; pooled 95% CI):**
| Format | Seeds | OpenDrawer | PickPlace | Microwave |
|---|---|---|---|---|
| sentence | 3 | 53% (48 to 59) | 22% (17 to 27) | 44% (39 to 50) |
| kv_text | 3 | 55% (49 to 60) | 23% (19 to 28) | 48% (42 to 53) |
| points_2d | 3 | **65% (59 to 70)** | 23% (19 to 28) | **56% (51 to 62)** |
| pose_path (demo scenes) | 1 | 56% | 16% | 38% |
| points_3d | 1 | 41% | 12% | 18% |
| points_3d_rgb | 1 | 33% | 7% | 25% |

**Part N (points formats):** success is flat for point errors of 0 to 10 cm. With the grasp point on a distractor, the gripper reaches the distractor in 1 to 5 of 50 episodes and the true object in 10 to 27. Removing or renaming the object name collapses OpenDrawer to 0 to 2 of 50 on most runs.

**What failed and why (details in `docs/research_journal.md`):** MeanFlow as trained here never beat predicting the mean action (0% on drawer and pick-place at 20k on 3 tasks and at 70k on 18; a 4x larger batch did not help; one-demo overfitting failed). Regression memorizes one demo (action error 0.009) and passed the gate at 5k (OpenDrawer 22/50).

**Other numbers:** replay 30/30; eval wiring 30/30; untrained 0/28; scripted oracle 46% / 82% / 100% (pick, drawer, microwave); training speed 3.27 steps/s (MeanFlow, H100), 5.4 steps/s (regression, H200); determinism fix changed OpenDrawer by 1 episode (21 vs 22 of 50).

## 3. Done
- A1 downloads verified (18 tasks, 500 to 543 each). A2 simtest OK after re-downloading assets. A3 dataset notes. A4 3 tasks on the Mac.
- B1 data loader + frame store, 6 tests pass. B2 replay test passes.
- C1 training script with resume (Mac test passes). C2 speed test. C3 runs launched.
- D1 eval script; wiring and zero tests pass. D2 eval job array works (used for D1 on Bouchet).
- E1 geometry self-check passes. E2 labels. E6 reading notes (`docs/related_notes.md`).
- F1 labels for 18 tasks. F3 five formats implemented (`policy/variants.py`); each trains 50 Mac steps with falling loss; a unit test shows moving a point changes the output. F6 eval with ground-truth commands, point error, distractor point, dropped fields, renames.
- I1 to I4 paper scaffolding builds a PDF. M1 to M3 watcher, queue, self-resubmitting training. P simlib with ground-truth parts for all 18 tasks. X lock files, container recipe (not built), CI workflow (not run on GitHub), clean-clone test passes.

## 4. Running (Bouchet; check with `ssh bouchet squeue -u ak3288`, `tail $ROOT/logs/watcher.log`)
| Job | What | Ends |
|---|---|---|
| 34842801 | run1_smoke: stock DEM, 3 tasks, 20k | about 05:00 |
| 34842802 | run2_all18: stock DEM, 18 tasks, 300k | about 01:30 Sat (resubmits itself at the 2-day limit) |
| 34842803, 34842804 | run1 with seeds 1 and 2 | about 05:00 |
| 34842805 | run1 with left + wrist cameras | about 05:00 |
| 34843699 | f4_kv_text | about 05:10 |
| 34843694 | watcher (agent partition, 7 days) | Dec 4 |
Queued in `runs/queue.txt`: f4_points_3d_rgb, f4_points_2d, f4_pose_path (launch as GPUs free; 2 GPUs are kept free for scoring).

## 5. Blocked or failed
- **D3 (first real numbers) failed:** see section 2. Ablations abl_nodisp, abl_bs128, abl_clip3, abl_lrhead3e4 (5k steps each, jobs 34864024 to 34864027) and eval tests of run1 20k with zero noise and with 4 sampling steps (34864030, 34864031) are running. A one-demo overfitting test is running on the Mac.
- **Watcher bug, 05:00 to 06:30:** it read each run's step from the 1.3 GB train_state.pt, which failed on the 8 GB agent node, so it resubmitted all five finished 20k runs every 10 minutes. Each resubmission trained one more step and triggered another 100-episode eval, filling the GPU cap. Fixed (progress from checkpoint names); the extra checkpoints (ckpt_0020001, ckpt_0020002) and their result CSVs are not deleted; ignore them.
- **YCRC cancelled one eval job (34848256_0) for 0% GPU use;** another was warned. Evaluation is simulator-bound, so all scoring now runs on CPU nodes.
- **I lost about 5 hours (01:10 to 06:30):** my session waited on a helper with nothing to wake it, so the near-zero numbers went unnoticed while the cluster kept scoring and the watcher misbehaved.
- Part W (RPent baseline): installed on Bouchet, simulator checks pass; blocked on a planner LLM choice (decision 5).
- Kitchen assets on Bouchet were incomplete (login-node download killed at 82%); fixed by job 34840169.
- ICML 2027 style files are not published; the paper uses the 2026 kit for now.

## 6. Needs Adi
- `docs/decisions_needed.md`: (1) HexaShell already runs on RoboCasa: switch William to it? (2) run_skill stops on the success check. (3) cameras. (4) the paper's claim. (5) planner LLM for the RPent baseline.

## 7. Things to review by hand
- `policy/sim.py`: action reordering (LeRobot to robosuite), gripper and base-mode snapped to +-1 by sign, image flip, env built from the dataset's kwargs.
- `policy/eval.py`: 16 predicted, 8 executed; success checked after every step; per-episode seed sets `env.rng`.
- `policy/data.py`: two-stage normalization (per-task q01/q99 to [-1, 1], then global z-score in the policy).
- `policy/commands_gt.py`: what a "perfect" command is at test time (body origin of the object), which differs from the training labels (gripper position at grasp).

## 8. Surprises
- The DEM README's example feeds "scene camera, wrist camera"; the plan says left and right agent views. Run 1 follows the plan; run 5 tests left + wrist.
- The four format variants gave nearly identical losses over their first 50 steps (same seed and data order; the head starts with zero output layers, so early learning is the mean action). Not a bug by itself, but check that the format curves separate later.
- gpu_devel allows only 1 submitted job; gpu_h200 allows only 6 CPUs per GPU.
- **HexaShell already ships a RoboCasa port** with skills for all 50 RoboCasa365 tasks (59.3% without skills, 85.6% with, pretrain split). `docs/agent_port_spec.md` assumed it did not. Notes: `docs/hexashell_robocasa_notes.md`.
- **The scripted oracle is strong** on the button and drawer tasks (100%, 82%). The learned policy may not beat it at 20k steps on those.
- **Mac render bug:** on the Mac, loading the model after an env has been reset twice corrupts every later render. Eval on Bouchet loads the model first, so it is not affected; the commander now does the same, and eval records an image check per episode (`img_rowdiff0`). The one commander smoke episode ran before the fix.
- **A code review found 11 issues**, all fixed before any trained checkpoint was scored (list in `docs/overnight_log.md`, 00:22). The most important: re-run eval shards could double-count, and object-split runs took normalization stats from held-out demos.
