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
