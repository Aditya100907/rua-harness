# Re-rendering demos with depth (E3, prep for F2)

Run Nov 26-27, 2026 on Adi's Mac (conda env `dem`, Apple GPU through MuJoCo's default GL).
Code: `policy/rerender.py` (render + `PointMapStore` reader), `policy/check_depth.py` (grasp
point check), `tests/test_rerender.py` (codec and reader round trip, no simulator).
Cluster jobs: `policy/slurm/rerender_array.sbatch`, `policy/slurm/depth_spotcheck.sbatch`
(written, not submitted).

## What it does

For each demo: build the env from the dataset's own kwargs with cameras left and right (wrist
optional, `--cams left right wrist`) at 256x256 and `camera_depths=True`, reset to the demo scene
(`policy.sim.reset_to_episode`), then for every frame t set `states[t]`, call `forward()` and
render RGB + depth per camera with `sim.render`. Depth goes to metres with
`simlib.camera.metric_depth`, is flipped to the dataset orientation (top row first), rounded to
uint16 millimetres (0 = invalid, which also covers anything past 65.5 m) and saved.

One file per episode, `<base>/<task>/episode_NNNNNN.npz`, written to a temp name and renamed, so
a file on disk means the episode is done. It holds, per camera: intrinsics `K_<cam>` (fixed),
extrinsics `T_world_cam_<cam>` for every frame (the agentview cameras ride on the mobile base,
so they move), the depth, plus `T_world_base` per frame (from the parquet state) and, when
checked, `rgb_mae_<cam>` per frame. Base path: `$RUA_DEPTH`, else `$SCR/depth` on Bouchet, else
`~/apollo-lab/runs/depth`.

Depth compression: groups of 8 frames. The first frame of a group is differenced along rows,
the rest against the previous frame, then zlib level 6. Lossless on the millimetre values.
Measured on PickPlaceCounterToCabinet episode 0, left camera, KB per frame:

| method | KB/frame | random frame access |
|---|---|---|
| zlib on raw uint16 | 32.9 | no |
| 16-bit PNG per frame (level 3) | 22.3 | yes |
| 16-bit PNG per frame (level 9) | 17.1 | yes, but 10x slower to encode |
| lzma on the whole episode | 7.9 | no |
| **groups of 8, time differences, zlib 6** | **9.0** | **yes (decode 8 frames)** |

The outer npz is not deflated, so the reader seeks straight to one group.

## Results on PickPlaceCounterToCabinet, episodes 0 to 9

All 10 rendered with no failures (5 asked for; 10 run to get steadier numbers).

**RGB matches the dataset videos.** Mean absolute error between the re-rendered frames and the
mp4 frames, averaged over each episode, 0 to 255 scale: left 2.28, right 2.36 (mean over the 10),
worst episode 3.15. Per-frame range on episode 0: 2.6 to 3.2. That is video compression noise;
an unflipped image gives about 68 (`docs/geometry_notes.md`).

**Stored depth equals a fresh render.** At the grasp frame of each of the 10 demos, the decoded
depth differs from a new `render_view` by at most 0.5 mm (the rounding), both cameras.

**Base frame round trip.** The base-frame point map mapped back through `T_world_base` gives
the world-frame point to 0.0 mm.

**Point map at the grasp pixel.** Project the hindsight grasp point (`grasp_world`, the
end-effector position at `grasp_frame`, from `~/apollo-lab/runs/labels`) into each camera at
`grasp_frame`, round to a pixel, read the point map there. Distance to the grasp point in cm:

| episode | left: at pixel | left: best in 11x11 window | right: at pixel | right: best in window |
|---|---|---|---|---|
| 0 | 45.4 | 5.3 | 1.9 | 0.9 |
| 1 | 41.4 | 34.1 | 6.3 | 1.3 |
| 2 | 23.3 | 2.0 | 2.9 | 2.1 |
| 3 | 2.2 | 1.6 | 2.7 | 1.6 |
| 4 | 24.2 | 2.4 | 23.8 | 1.1 |
| 5 | 1.3 | 1.0 | 2.0 | 1.5 |
| 6 | 34.0 | 10.7 | 24.4 | 2.6 |
| 7 | 27.5 | 3.5 | 66.7 | 2.6 |
| 8 | 27.0 | 2.3 | 2.7 | 2.2 |
| 9 | 34.1 | 25.3 | 3.4 | 1.7 |

Summary: right camera, window: median 1.7 cm, max 2.6 cm, 10 of 10 within 3 cm. Left camera:
only 2 of 10 within 5 cm at the pixel, 6 of 10 in the window. Best camera per episode, window:
median 1.7 cm, max 2.6 cm, 10 of 10 within 5 cm.

**Why the left camera misses: the arm is in the way.** Segmentation at the projected pixel (grasp
frame) shows `robot0_link5` (7 demos) or `robot0_link3` (1) for the left camera. The depth there is
23 to 45 cm closer than the grasp point, which is what an occluding arm gives. The right camera
hits the object (`obj_main`) in 6 demos, a gripper finger in 1, and arm links in 3, but its 5-pixel
window always reaches the object. So this is occlusion, not a geometry error. In the episodes where
the arm does not block the view (left camera episodes 3 and 5) the pixel itself is within 2.2 cm.

**Side finding: some objects are see-through in depth.** In episode 0 (a honey bottle) only 76%
of the pixels the segmentation marks as the object lift to within 1 cm of the object's geoms;
the rest have depth 10 to 20 cm behind the bottle. Eroding the mask by a pixel does not change
it, so it is not edge blur. The bottle has a "Clear" part, and the depth buffer sees through it.
Other demos, where the object is visible: 96% to 100% (masks of 6 to 364 pixels; one 1-pixel mask left out). Treat transparent objects as a known gap
of simulated depth, as a real depth camera would also struggle with them.

## Speed and size (Mac)

| | per demo (mean of 10) |
|---|---|
| frames | 227 |
| reset to the demo scene (XML compile) | 6.1 s |
| render 2 cameras + encode | 4.2 s (18.5 ms per frame for both cameras) |
| total wall time | 11.0 s |
| file size, 2 cameras | 4.09 MB (18.0 KB per frame) |
| file size, 3 cameras (episode 0 only) | 7.70 MB, vs 4.79 MB with 2 (wrist adds about 11.8 KB per frame) |

Reader (`PointMapStore.point_map`, 256x256, base frame): 2.1 ms per random frame, 0.9 ms
sequential (one group decode serves 8 frames).

## Cost estimate for all ~9,100 demos (~2.23M frames)

- **Disk:** 2.23M x 18.0 KB = **about 40 GB** for left + right. About 66 GB with the wrist
  camera. 9,100 files. Fits on scratch, but scratch purges files after 30 days, so either finish
  F2 training inside that window or copy to project storage.
- **Mac, one process:** 9,100 x 11 s = **about 28 hours**.
- **Bouchet `day` partition:** not measured. The nodes have no GPU, so rendering uses Mesa's
  llvmpipe software renderer. The reset (about 6 s) is CPU bound and should be similar. If
  software rendering is 3 to 10 times slower than the Mac GPU, a demo takes about 20 to 50 s,
  so **about 50 to 125 job-hours** (4 cores each, 200 to 500 core-hours). The array is 216 jobs of
  50 demos, about 15 to 45 minutes each, under the 4 hour limit. With 60 running at once
  (`--array=0-215%60`) that is 1 to 2 hours of wall time. The first chunk's log gives the real
  rate: check `rerender_log.jsonl` (fields `reset_s`, `render_s`) before trusting this.

## Rendering on CPU nodes: EGL works, OSMesa does not

Checked on the Bouchet login node, which has no GPU (`nvidia-smi` fails), Nov 27:

- `MUJOCO_GL=egl` with a plain `mujoco.Renderer` works: GL renderer
  `llvmpipe (LLVM 19.1.7, 256 bits)`, Mesa 24.2.8, 1.2 ms per frame on a one-box scene. glvnd
  picks Mesa's EGL vendor (`/usr/share/glvnd/egl_vendor.d/50_mesa.json`) when there is no NVIDIA
  device.
- robosuite's own EGL context (`MjRenderContextOffscreen(sim, device_id=-1)`) also renders RGB and
  depth there.
- A harmless EGL error prints when the process exits (in `__del__`). Ignore it.
- `MUJOCO_GL=osmesa` fails: there is no `libOSMesa` in the env or the system. Do not use it.

So `common.sh`'s `MUJOCO_GL=egl` is right for the `day` partition. The job logs the GL renderer
first, and sets `LP_NUM_THREADS` (llvmpipe's thread count) to the job's cores. Not checked: that
`day` compute nodes have the same Mesa as the login node (the job's probe line will show it), and
that llvmpipe's RGB matches the dataset videos as closely as the Mac's (the job compares the first
demo of every chunk with its video; look at `rgb_mae_*` in the log).

## How to run

```bash
# Mac
PYTHONPATH=. python -m policy.rerender --task PickPlaceCounterToCabinet --episodes 0 1 2 3 4 --check-rgb
PYTHONPATH=. python -m policy.check_depth --tasks PickPlaceCounterToCabinet --episodes 0 1 2 3 4 \
    --labels ~/apollo-lab/runs/labels

# Bouchet (F2): all tasks, then the spot check
sbatch --array=0-215%60 policy/slurm/rerender_array.sbatch
sbatch --dependency=afterany:<jobid> policy/slurm/depth_spotcheck.sbatch
```

The F2 spot check (`policy/check_depth.py --all-tasks --n 20`) picks 20 random rendered demos per
task and runs the grasp-pixel check at `grasp_frame`. A task passes when the best-camera window
distance is at most 5 cm on at least 90% of them. It needs the F1 labels at `$SCR/labels/<task>.jsonl`.
That folder exists on Bouchet but has no label files yet, so run `policy/label_commands.py` there
(or rsync the Mac labels for the 3 smoke tasks) first.

Reader for a data loader:

```python
from policy.rerender import PointMapStore
store = PointMapStore()                                    # $RUA_DEPTH or the default base
pm = store.point_map("OpenDrawer", ep=3, t=40, cam="left", frame="base")   # [3, 256, 256] float32
d = store.depth("OpenDrawer", 3, 40, "left")               # [256, 256] metres, NaN = invalid
K, T_base_cam = store.camera("OpenDrawer", 3, 40, "left", frame="base")
```

`frame="base"` is the robot base frame at that same frame t. It keeps up to 64 episode files
open and drops open handles when pickled, so it is safe in DataLoader workers.

## Not done

- Only one task (PickPlaceCounterToCabinet) and 10 demos were rendered. OpenDrawer and
  TurnOnMicrowave were not.
- No run on Bouchet, so the cluster speed is a guess.
- The wrist camera was rendered for one demo only (size), not checked against the labels.
- No look at frames where the robot base drives (the extrinsics are stored per frame, and the RGB
  match holds over whole episodes, but no base-motion frame was checked on its own).
