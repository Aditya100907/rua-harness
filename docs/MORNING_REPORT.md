# Morning report, Fri Nov 27

Updated as the night goes. Last update: Nov 26, 23:40 ET.

## 1. Bottom line
The data is all there (18 tasks, 9,126 demos) and the replay test passes on OpenDrawer, 10 of 10. The cluster's kitchen assets were incomplete, so the first simulator test failed; a re-download job is running. No training has run yet.

## 2. Numbers
- Replay test, Mac: OpenDrawer 10/10. TurnOnMicrowave and PickPlaceCounterToCabinet pending.
- No success rates for any policy yet.

## 3. Done
- A1 downloads verified: 18 tasks, 500 to 543 episodes each. `docs/dataset_notes.md`.
- A3 dataset read: state, action, camera and instruction layout in `docs/dataset_notes.md`.
- E6 reading: `docs/related_notes.md`. M4 docs: `docs/agent_on_cluster.md`.
- I1 to I4 paper scaffolding: `paper/` builds a PDF; tables are placeholders until results exist.

## 4. Running
- 34840169: kitchen asset re-download (day partition).
- 34840595_[0-17]: frame extraction, one task each (day partition).
- Check: `ssh bouchet squeue -u ak3288`.

## 5. Blocked or failed
- A2 simtest failed once: missing object assets (login-node download was killed). Fix running.

## 6. Needs Adi
- (none yet)

## 7. Things to review by hand
- `policy/sim.py`: action reordering and gripper snapping, image flip, env built from the dataset's kwargs.
- `policy/data.py`: two-stage normalization (per-task quantiles, then global z-score).

## 8. Surprises
- The DEM README's inference example feeds "scene camera, wrist camera". The plan says left and right agent views. Run 1 follows the plan; a left plus wrist run is queued as a comparison.
