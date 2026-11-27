# Morning report, Fri Nov 27

Updated through the night. Last update: Nov 27, 00:50 ET.

## 1. Bottom line
Every test before the gate passes: replay (30/30 on Bouchet), wiring (30/30) and zero test (0/28). Stock DEM is training on the 3 smoke tasks (run 1) and on all 18 tasks toward 300k steps (run 2), with two more seeds, a camera comparison and the first format-study runs alongside. The cluster now scores checkpoints and launches queued runs by itself. No trained-policy success numbers exist yet; the first (run 1 at 5k steps) should appear around 01:30.

## 2. Numbers
- **Replay test (B2):** Bouchet: PickPlaceCounterToCabinet 10/10, OpenDrawer 10/10, TurnOnMicrowave 10/10 (job 34841442). Mac: OpenDrawer 10/10, TurnOnMicrowave 10/10.
- **Wiring test (D1.1):** demo actions through the eval loop: 10/10 on each smoke task (job 34841796). Env state at reset equals the recorded state to 1e-8; rendered image vs video frame: 1.7 to 2.5 grey levels mean error.
- **Zero test (D1.2):** untrained head: 0/8, 0/10, 0/10 (job 34841793; 2 episodes lost to GPU out-of-memory, fixed).
- **Training speed (C2):** H100 80GB, batch 64, 2 cameras: 3.27 steps/s, GPU 83% busy, data wait 0%, 32 GB. Hours: 20k = 1.7 h, 100k = 8.5 h, 300k = 25.5 h. L40S: 1.16 steps/s (20k = 4.8 h). Run 2 on H100 runs at 3.29 steps/s.
- **Hindsight labels (E2, F1):** all 18 tasks labeled, counts equal episode counts. Grasp point to object origin, 20 demos per task: median 1.6 to 5.6 cm across the 6 pick-style tasks; every grasp point lies within the object's extent. Details: `docs/labels_notes.md`.
- **Geometry (E1):** pixel lifted to 3D lands 3.2 cm (left) and 3.5 cm (right) from the object origin; depth must be flipped (28 to 43 cm error otherwise). `docs/geometry_notes.md`.
- **Scripted oracle baseline (F7)**, ground-truth positions, no learning, 50 episodes each, seeds 1000 to 1049: PickPlaceCounterToCabinet 23/50 (46%), OpenDrawer 41/50 (82%), TurnOnMicrowave 50/50 (100%). This is the bar the learned policy has to clear on these tasks.
- **Test sets (Y), scripted baseline, small samples:** look-alike distractors 6/20 when it goes for the target, 3/20 when it picks one of the two at random; recombined kitchens 4/10, 8/10, 10/10; RoboCasa's pretrain kitchens 7/10, 5/10, 10/10.
- **Commander (Part O) smoke test, 1 episode, 50-step checkpoint:** 62 s, 11 turns, 283k input tokens (90% cache reads), 1.2k output, $0.29 at list price. Pointing error 81.5 cm with the arm in the way, 3.1 cm from the clear camera.
- **Policy timing on the Mac (M-series, MPS):** 47 ms per act call, 118 ms per robot second. GPU numbers pending (job 34847608).
- Learned-policy success rates: none yet. First at 5k steps around 01:10.

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
