# Twin-distractor data (PickPlaceCounterToCabinet)

Code: `policy/twin_data.py`. Tests: `tests/test_twin_data.py`. Cluster job: `policy/slurm/twin_array.sbatch`.
Decision: docs/decisions_log.md, Sat Nov 28 13:42 (research recommendation 1).

## Why

The distractor test (docs/research_journal.md, Nov 28) showed that our point-format policies
follow the object name, not the point. In the stock demos the name always picks out one object,
so the point never has to carry the choice. These data add a second, identical copy of the
target object (the "twin") to every demo scene. The instruction names a category that now fits
two objects. Only the point, which stays on the true object, says which one to pick.

## What one demo becomes

1. Reset to the demo scene with the stock model (`policy.sim.reset_to_episode`). Record the
   position of every body at every 10th frame and the last frame, the target's pose at frame 0
   and the counter height under it (downward ray, collision geoms only).
2. Draw up to 400 candidate spots. A spot passes the cheap checks when it is 15 to 45 cm from the
   target (xy), at least 15 cm from the grasp point (gripper position at the first gripper
   close, as in the labels), at least 10 cm from every gripper position of the demo, and when
   nine downward rays over the twin's footprint (bounding box plus 1 cm) all hit a free surface
   within 1.5 cm of the target's counter height. Orientation: the target's frame-0 orientation
   turned by a uniform random yaw about world z (seed 10000 + episode).
3. Add the twin to the model XML: a copy of the `obj_main` subtree, every name `obj_*` renamed
   `twin_*`, mesh and material references shared, free joint, appended last in the worldbody.
4. The saved states have the stock layout (243 numbers for episode 0: time, 124 qpos, 118 qvel).
   Each frame is mapped into the new layout by joint name (qpos and qvel addresses). The twin's
   qpos is its spot, its velocity zero.
5. A spot is kept when, at every frame of the demo, no twin geom touches anything except the
   counter body it stands on (robot, carried object, cabinet doors, other objects), and when the
   twin covers at least 40 pixels of the left camera at frame 0 (segmentation render). Among the
   first 4 kept spots, the one visible in the wrist camera at frame 0 wins, then the one with
   the most left-camera pixels.
6. Render left, right and wrist RGB at 256 x 256 for every frame, flip to dataset orientation,
   JPEG quality 95 (as `policy/extract_frames.py`).

## Checks (run on every demo; a demo that fails one is not written)

- Original bodies unchanged: max distance between every original body's position in the twin
  replay and in the stock replay, at the check frames, must be under 1 mm. It is exactly 0 on
  the Mac demos, as expected: same kinematic tree, same joint values. The test
  `test_state_map_by_joint_name` checks the mapping itself with the twin inserted in the middle
  of the joint order.
- Twin visible: segmentation pixels > 0 in at least one camera at frame 0 (the selection already
  asks for 40 in the left camera).
- Twin stays put: drift of the twin's body over the replay (0 here, since it is held).
- Demo still works with the twin as a free body (eval scenes): demo 3's recorded actions,
  played open loop in the twin scene, still succeed; the twin moved 0.003 mm.
- Frame 0 of the stored JPEGs against a live observation of the twin eval scene: mean absolute
  error 1.7 (0 to 255), JPEG noise.
- End to end on the Mac: a 2-step points_2d checkpoint trained on the twin task (5 demos)
  and evaluated on one twin scene with `--twin-point` writes all columns (smoke test only).
- Training data loads: `RoboCasaDataset(["PickPlaceCounterToCabinetTwin"])` gives the same state,
  actions and instruction as the source task, and twin images (merge test on 10 episodes, 5 with
  twin, 5 filled from the source store, which match the source exactly).

## Mac test, 5 demos (episodes 0 to 4)

| ep | category | twin to target (xy) | twin to grasp | twin to nearest gripper pos | twin pixels left / right / wrist at frame 0 | seconds |
|---|---|---|---|---|---|---|
| 0 | honey bottle | 44.4 cm | 45.6 cm | 39.8 cm | 636 / 363 / 2371 | 20.4 |
| 1 | milk | 27.3 | 29.9 | 17.9 | 552 / 552 / 1374 | 20.4 |
| 2 | sweet potato | 35.7 | 35.9 | 35.8 | 104 / 86 / 348 | 24.6 |
| 3 | cheese | 25.6 | 24.7 | 24.6 | 202 / 158 / 940 | 15.9 |
| 4 | carrot | 22.4 | 20.8 | 16.5 | 136 / 112 / 397 | 21.5 |

All 5 placed with no failures. About 20 s per demo on the Mac (stock reset 7 to 12 s, placement
3 to 6 s, rendering about 10 s). Frame 0 images: `docs/img/twin/twin_ep000.png` to
`twin_ep004.png` (left | right | wrist; in the left view the twin is outlined in magenta from
the segmentation render and the target is marked with a green cross).

## Outputs

- Frame store: `<frames>/PickPlaceCounterToCabinetTwin/{left,right,wrist}.bin`, `.idx.npy`,
  `DONE`, same format and global frame indexing as the source task, so `FrameStore` reads it
  unchanged. Demos whose twin failed hold the source task's frames (has_twin false).
- `twin_meta.jsonl`: episode, source episode, twin position and quaternion, yaw, category,
  distances, pixel counts, checks, candidate counts.
- Labels: `<labels>/PickPlaceCounterToCabinetTwin.jsonl`, the source labels (grasp and place
  points on the true object) plus `has_twin`, `twin_world`, `twin_cat`.
- Episode lists: `runs/twin_train.json` (training split episodes 0 to 451 that have a twin) and
  `runs/twin_heldout50.json` (held-out episodes 452 to 501 with a twin), keyed by the twin task.

## Virtual task

`paths.VIRTUAL_TASKS` maps `PickPlaceCounterToCabinetTwin` to `PickPlaceCounterToCabinet`.
`paths.task_dir`, `paths.horizon` and `commands_gt.perfect_command` resolve it to the source, so
`load_task_table`, the quantiles, the env and the instructions are the source task's, while the
frame store and labels are the twin task's own. Train with
`--tasks PickPlaceCounterToCabinetTwin --episodes-json runs/twin_train.json` (alone, or next to
the stock tasks).

## Evaluation on twin scenes

`policy/eval.py --tasks PickPlaceCounterToCabinetTwin --demo-scenes runs/twin_heldout50.json`
rebuilds each held-out demo's initial scene, adds its twin as a free body at the placement in
`twin_meta.jsonl` (or recomputes it the same way) and runs the policy. `--twin` does the same for
the stock task name. Extra columns: `min_dist_twin_cm` (gripper to the twin's live position),
`min_dist_true_cm` (gripper to the target's position at reset), `went_to` (the copy the gripper
came closer to, or "neither" when it came within 5 cm of neither, as in the N2 distractor test), `twin_moved_cm`. `--twin-point` puts the command's grasp point on the twin
instead of the target: the counterfactual command (same name, other copy). A checkpoint trained
only on the stock task is scored with the stock task's normalization.

Success is still the env's own check on the true object. A twin put in the cabinet is not
counted as success; `twin_moved_cm` shows whether the twin was picked up.

## Limits

- Twin frames are fresh renders; the stock frames are decoded from the dataset mp4s. They differ
  by about 2.3 (0 to 255) mean absolute error (docs/rerender_notes.md), video compression noise.
  Evaluation uses live renders, so the twin data are if anything closer to evaluation.
- The twin is held still in training data. The demo gripper never comes within 10 cm of it, so
  no demo shows a contact with it.
- Only PickPlaceCounterToCabinet. The same code should work for the other counter pick-place
  tasks with a different source name; not tried.
