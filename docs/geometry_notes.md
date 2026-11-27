# Geometry notes: RoboCasa cameras, depth and pixel to 3D

Checked Nov 26, 2026 on Adi's Mac (conda env `dem`, RoboCasa 425684d, robosuite 5ce6643,
MuJoCo 3.3.1) by running `scripts/render_geometry.py`. The code that applies these rules is
`simlib/camera.py`. Its tests are `tests/test_simlib.py` and `tests/test_simlib_sim.py`.

## What was run

`PYTHONPATH=. python scripts/render_geometry.py`: task `PickPlaceCounterToCabinet`, seed 0,
`split="target"`, three cameras at 256x256 with `camera_depths=True`. The target object (a
measuring cup) is masked with MuJoCo segmentation. The mask's centre pixel is lifted to 3D with
metric depth, intrinsics and extrinsics and compared with the simulator's object position.

Result: **SELF-CHECK PASSED**.

| camera | centre pixel lifted vs object origin | mask pixels that lift onto the object | true position projected onto the mask |
|---|---|---|---|
| robot0_agentview_left | 3.2 cm | 100% | yes |
| robot0_agentview_right | 3.5 cm | 100% | yes |
| robot0_eye_in_hand | 6.3 cm | 100% | yes |

With the depth NOT flipped, the agentview errors are 28.3 cm and 43.4 cm and 0% of mask pixels
land on the object. So the flip is required.

The few cm left over is expected: the centre pixel is on the cup's visible surface, the
simulator position is the body origin. The eye-in-hand camera is 0.48 m from a cup that fills a
big part of the image, so the gap there is larger (6.3 cm). The script does not hold the
eye-in-hand centre error to 5 cm; it does require 95% of mask pixels to lift onto the object
(within 1 cm of an object geom's bounding sphere). For the eye-in-hand camera the flip test is not
decisive in this scene (both versions pass), because the cup is near the image centre. The two
agentview cameras decide it.

The first run of the original script (split `pretrain`, beer bottle) gave the same answer:
2.5 cm with the depth flipped, 79.2 cm without.

## Conventions that were right

1. **Rows are flipped.** robosuite observations (`obs["<cam>_image"]`, `obs["<cam>_depth"]`) and
   `sim.render(...)` come bottom-row-first (OpenGL), because `robosuite.macros.IMAGE_CONVENTION`
   is `"opengl"`. The dataset videos are top-row-first. simlib flips every image once, in
   `simlib.camera.views()` / `render_view()`, and works top-row-first from then on.
   - Checked against the data: reset the sim to OpenDrawer target demo 0, rendered
     `robot0_agentview_left`, compared with frame 0 of the demo mp4. Mean absolute pixel
     difference: 1.7 flipped, 67.7 not flipped (0 to 255 scale). See
     `docs/img/geometry/dataset_frame0.png` and `render_flipped_demo0.png`.
   - `robosuite.utils.camera_utils.get_camera_segmentation` already flips. Do not flip its
     output again. `simlib.camera.segmentation()` flips the raw render itself.
   - robocasa's robomimic wrapper also flips RGB obs (`[::-1]`), which agrees.
2. **Depth is normalised.** MuJoCo returns depth in [0, 1]. Metres along the optical axis:
   `near / (1 - d * (1 - near / far))` with `near = model.vis.map.znear * model.stat.extent`,
   `far = model.vis.map.zfar * model.stat.extent`. In these kitchens extent is about 29.8, so
   near is about 3 cm and far about 1.5 km. This is z-depth (along the axis), not distance
   along the ray. Same as `robosuite.utils.camera_utils.get_real_depth_map`.
3. **Camera axes.** MuJoCo cameras look along their own -z with +y up. simlib uses the OpenCV
   frame: +z forward, +x right, +y down. `T_world_cam = [R_mujoco @ diag(1, -1, -1), p]`. This
   equals `get_camera_extrinsic_matrix` (checked to machine precision for all three cameras).
   The pose changes as the robot moves, so read it at the same step as the image.
4. **Intrinsics.** `f = 0.5 * H / tan(fovy / 2)`, `cx = W / 2`, `cy = H / 2`, square pixels.
   fovy is 60 deg for both agentview cameras and 75 deg for the eye-in-hand camera. Equals
   `get_camera_intrinsic_matrix`.
5. **Pixel centres.** simlib takes integer pixel indices `(u, v)` = (column, row) and uses the
   pixel centre `(u + 0.5, v + 0.5)`. `world_to_pixel` returns continuous index coordinates, so
   rounding gives the pixel index. `rua_harness.geometry.lift_pixel` skips the half-pixel shift.
   That is at most half a pixel (about 2 mm at 1 m for a 256 image), so it is left alone. It
   expects top-row-first depth too.

## Images

All in `docs/img/geometry/`, for each camera (`agentview_left`, `agentview_right`,
`eye_in_hand`): `_rgb`, `_depth` (clipped at 4 m, scaled to 0 to 255), `_depth_noisy`
(`rua_harness.geometry.add_depth_noise`), `_mask` (target object), `_point_map` (world x, y, z as
r, g, b, each scaled to 0 to 255), `_overlay` (mask in green, the projected true position as a red
square). Arrays for the left camera: `~/apollo-lab/runs/geometry_demo/scene.npz` (not in git).

## Things that were not checked

- Only one scene for the lift test (one task, one seed). `tests/test_simlib_sim.py` repeats it at
  128x128 on the same scene.
- Only OpenDrawer demos are downloaded on the Mac, so the video comparison covers one task.
- No camera randomisation (`randomize_cameras=False`, the create_env default).
