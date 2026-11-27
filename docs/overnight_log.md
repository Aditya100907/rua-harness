# Overnight log, Nov 26 to 27

Times are ET.

- 23:18 Branch `overnight-nov26` created. SSH to Bouchet works with no prompt.
- 23:19 A1: no download process running. All 18 tasks present, 500 to 543 episodes each (`meta/info.json`). Table in `docs/dataset_notes.md`.
- 23:20 A2: simtest job 34840042 (H200, gpu_devel) FAILED: `UtensilRack002/model.xml` missing. Cause: the kitchen-asset download on the login node was killed at 82% of `aigen_objs.zip`; `objects/lightwheel` never downloaded.
- 23:22 Submitted 34840169 (day partition, 32G) to download `objs_aigen` and `objs_lw` again. Other asset folders match the Mac's sizes.
- 23:22 A4: rsync of the 3 smoke tasks to the Mac started. Slow (about 1 MB/s).
- 23:23 Started 3 helper agents: E6 reading plus M4 agent-partition docs, I1 to I4 paper scaffolding, E1 geometry plus P simlib.
- 23:25 A3: read OpenDrawer layout, parquet, modality.json, extras, env kwargs. Notes in `docs/dataset_notes.md`.
- 23:27 B2: `policy/replay_test.py` written. OpenDrawer 10/10 on the Mac (env built from the dataset's own kwargs, LeRobot actions reordered to robosuite order, gripper and mode snapped to +-1).
- 23:28 B1: `policy/extract_frames.py` (JPEG q95, one file per task and camera) and `policy/data.py` written. Frame extraction array 34840595 (18 tasks, day partition) submitted.
- 23:29 E6 and M4 done by helper: `docs/related_notes.md`, `docs/agent_on_cluster.md`.
- 23:35 I1 to I4 done by helper: `paper/` builds a PDF with tectonic (ICML 2026 style; 2027 kit not out). `scripts/make_tables.py`, `scripts/make_plots.py`, tests pass.
- 23:37 Asset re-download done (aigen_objs and lightwheel now present). SIMTEST OK (job 34841443, gpu_devel H200, EGL; 35.6 s for env creation plus 20 steps).
- 23:38 Frame extraction done for all 18 tasks, 0 length mismatches between videos and parquet. About 2 to 4 GB per camera per task.
- 23:40 Bouchet limits found: gpu_devel 1 job submitted at a time and 12 CPUs; gpu_h200 at most 6 CPUs per GPU. Training jobs use 6 CPUs and 5 loader workers.
- 23:41 C1: 50 Mac steps (MPS, batch 4, pretrained towers), loss 4.79 to 0.79; resume from step 30 continued to 50. Pass.
- 23:44 D1 wiring on the Mac: OpenDrawer 3/3; state at reset matches the recording to 1e-8; rendered left image vs video frame 0 mean error 1.7 to 2.5 grey levels.
- 23:46 C2 speed test (job 34841772, H100 80GB, batch 64, 3 tasks): 3.27 it/s steady, GPU 83% busy, data wait 0%, 32.2 GB peak. Loss 0.75 to -0.26 (MeanFlow loss with the dispersive term can go negative). Checkpoint reloads.
- 23:47 D1 wiring on Bouchet (job 34841796): 10/10 on each of the 3 smoke tasks. D1 zero test (untrained checkpoint, job 34841793): PickPlaceCounterToCabinet 0/8, OpenDrawer 0/10, TurnOnMicrowave 0/10. 2 PickPlace shards died of GPU out-of-memory with 10 processes on one GPU; now 6 processes plus a retry pass.
- 23:48 E2: hindsight labels for the 3 smoke tasks. PickPlaceCounterToCabinet, 20 demos checked in the simulator: eef from the state vector equals the simulator's eef (0.0 cm). Grasp point to object origin: median 2.1 cm, p95 8.3 cm, max 8.6 cm, 14/20 within 5 cm. All 20 lie inside the object's bounding sphere; the 6 over 5 cm are larger objects grasped away from their origin. OpenDrawer: all 514 by gripper close. TurnOnMicrowave: 492 by gripper close, 51 by success frame (no close).
- 23:49 C3: run 1 (3 tasks, 20k steps) and run 2 (18 tasks, 300k steps) submitted to gpu_b200 (H100 2-day jobs would not start until 06:13). Jobs 34842499, 34842500.
- 00:15 (cluster clock) M test started: no SSH from the Mac for 30 minutes. Deviation: the SSH master connection is not killed, because reconnecting needs a Duo approval nobody can give overnight; not using it tests the same thing (the watcher has to work alone).
- 00:08 F5 held-out object split (PickPlaceCounterToCabinet): 16 of 106 categories held out; 328 training demos, 35 val, 36 locked test, 103 dropped (held-out object as distractor). F5 runs queued.
- 00:05 E4 run_skill done (test on a live OpenDrawer env passes). F2 depth re-render array for the 3 smoke tasks submitted (34844300, day partition).
- 00:14 Watcher bugs fixed: CPU jobs counted toward the 12-job cap; GPU count matched partition names by accident. Now both count jobs in gpu* partitions.
- 00:18 Part O commander built and tested: 1 real agent episode (budget used: 1 of 60).
- 00:22 Independent code review of the policy core (subagent). Confirmed correct: normalization once per stage, action order and gripper, chunk indexing, seeding through env.rng, point frames, resume, 16 predicted / 8 executed. Fixed at 00:30: (1) eval shards appended on requeue and partial shards merged silently, now cleared at start, deduped, checked for completeness; (2) pose_path scenes marked as training scenes when no split exists; (3) points_3d had no point-map input (fixed 00:27 with a train/eval match test); (4) norm stats used held-out demos, now training episodes only; (5) CUDA and numpy RNG saved on resume; (6) torch seeded per eval episode; (7) training resubmitted itself after crashes, now only after a clean time-limit stop; (8) watcher counted every pending array element as a GPU; (9) held-out evals had no retry marker; (10) padded chunk steps train on the repeated last action, now documented as deliberate; (11) min distances now per step, every checkpoint records its format.
- 00:28 GAE identified from Adi's notes: arXiv 2510.03896, verified on arXiv and added to refs.bib and docs/related_notes.md.
- 00:29 Part R done (subagent): robosuite has XArm7; RoboCasa needs a wheeled-base subclass; gripper state differs from the Panda's. Depth-noise model with D435 and ZED presets. docs/sim_to_real.md.
- 00:45 M test: during 30 min without SSH from the Mac, the watcher cycled every 10 min (00:25, 00:35) and all six training runs kept going. Nothing came due for scoring in that window (first 5k checkpoints about 01:05); the scoring half of the test is confirmed when those are scored with no SSH involvement.
- 00:50 F2: depth re-render for the 3 smoke tasks running (about 65 s per demo on CPU, llvmpipe EGL); remaining 15 tasks submitted (180 chunks, 60 at a time). H1 timing job 34847608 (gpu_devel RTX 5000 Ada). Composite target datasets download 34847609 (Part Q). Part S LIBERO handed to a helper.
- 00:58 M test complete: the watcher scored f4_kv_text's 5k checkpoint on its own (eval 34848256), with no SSH from the Mac involved. Pass.
