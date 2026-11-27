# Sim to real: what a real deployment needs

Written Nov 27, 2026 (Part R). Investigation and setup only. Nothing here ran on a real robot.

## 1. What the policy expects (the interface a real robot must match)

These are facts of the simulator setup we train in (RoboCasa `PandaOmron`,
`robosuite/controllers/config/robots/default_pandaomron.json`, `policy/sim.py`).

- **Control rate:** 20 Hz (`Kitchen` default `control_freq=20`). DEM predicts 16 actions and executes 8, so one chunk covers 0.4 s.
- **Arm command:** operational space control (OSC, a controller that turns a desired gripper motion into joint torques), pose, **delta** input, in the **robot base frame** (`input_ref_frame: base`). Each step asks for at most 5 cm of translation and 0.5 rad of rotation (axis-angle) for an input of 1. Gains: kp 150, critically damped.
- **Gripper:** one number, closed if > 0, open otherwise.
- **Mobile base:** 3 numbers (forward, side, yaw velocity, `JOINT_VELOCITY_LEGACY`), 1 torso height number, and 1 mode bit that switches between arm and base control.
- **Action vector (robosuite order, 12):** eef position 3, eef rotation 3, gripper 1, base 3, torso 1, mode 1.
- **State (16):** base position 3, base quaternion 4, gripper position in base frame 3, gripper quaternion in base frame 4, gripper finger positions 2 (Panda: metres, about 0 to 0.04 per finger).
- **Images:** robot-mounted cameras `robot0_agentview_left`, `robot0_agentview_right` (fixed to the base, about 1 m up, 60 degree vertical field of view) and `robot0_eye_in_hand` (wrist). Rendered at 256 x 256.
- **Depth and points:** the point-based command formats lift pixels to 3D with metric depth and the camera pose (`simlib/camera.py`, `docs/geometry_notes.md`). Simulator depth is perfect.

A real robot therefore needs:

1. **A low-level controller that accepts delta end-effector poses in the base frame at 20 Hz.** On a real arm this is usually a Cartesian impedance or OSC controller on the vendor's torque or velocity interface, or a servo-mode Cartesian stream. Translation and rotation limits must be rescaled to match 5 cm and 0.5 rad per unit input. If the arm only takes position targets, convert each delta into an absolute target and stream it.
2. **Camera calibration.** Intrinsics for every camera (the vendor's factory values are often enough for RealSense and ZED). Extrinsics: hand-eye calibration for the wrist camera (camera to flange), and camera-to-base calibration for the fixed cameras (for example with an AprilTag or ChArUco board held by the gripper). Errors here add directly to the 3D point error, on top of depth noise.
3. **A depth source.** A stereo camera (RealSense D435 or D455, ZED 2 or ZED X) or a learned depth model on RGB. Missing pixels (0) must be handled: the point lifter must refuse a 0-depth pixel or take the median of valid neighbours.
4. **The same camera placement as in training**, or retraining. DEM sees the two base cameras from fixed poses about 1 m up and 0.35 m to each side of the base. A real rig that differs will need new demos or rendered data from the new poses.
5. **A mobile base with a velocity interface**, or a fixed-base setup where the base action is ignored and tasks are chosen so the arm reaches everything.
6. **A gripper state in the same units.** The state includes two finger positions in metres. A different gripper must be mapped to the same range, or the policy retrained.

## 2. xArm 7 in our simulator

### What exists at our pins
- robosuite `5ce6643` ships `XArm7` (`robosuite/models/robots/manipulators/xarm7_robot.py`), meshes in `models/assets/robots/xarm7/`, and `XArm7Gripper`. It is registered as a `FixedBaseRobot`. Its default base is `RethinkMount` (a fixed pedestal).
- There is **no** `default_xarm7.json` controller file. `XArm7.default_controller_config` names one, but `load_composite_controller_config` falls back to `controllers/config/default/composite/basic.json` (seen in the log).
- `robosuite_models` is **not** installed, and we do not need it: its current GitHub page (fetched Nov 27) lists no xArm model and no mobile base (ALOHA, ARX5, G1, H1, PR2, VX300s, YuMi, Z1; bases Aloha mount, B1, Go2). I did not install anything.
- There is no xArm mobile variant. `PandaOmron` (Panda on the Omron LD-60 base with a torso lift) is the only wheeled arm RoboCasa uses.

### What RoboCasa assumes
- `Kitchen._setup_model` sets a special start pose only when the robot is a `PandaOmron`.
- `Kitchen.__init__` sets the controller part order (`right, right_gripper, base, torso`) and the legacy base controller only when `robots[0] == "PandaOmron"`.
- The fixed cameras in `robocasa/utils/camera_utils.py` are attached to the body `mobilebase0_support`. A robot without the Omron base has no such body.
- Many helpers (`env_utils.set_robot_to_position`, `kitchen_navigate.py`, subtask labelers) read joints named `mobilebase0_joint_mobile_*`.
- `create_env` loads the controller config by robot name.

### What I ran (Mac, `PickPlaceCounterToCabinet`, split target, seed 0)
| Setup | Result |
|---|---|
| `create_env(robots="PandaOmron")` | Works. 12-number action, 16-number state, three cameras. Baseline for comparison. |
| `create_env(robots="XArm7")` | **Breaks at `env.reset()`**: `ValueError: No "camera" with name robot0_agentview_center exists. Available: robot0_robotview, robot0_eye_in_hand`. The kitchen cameras hang off `mobilebase0_support`, which the fixed mount lacks. |
| `create_env(robots="XArm7", base_types="OmronMobileBase")` | Builds, resets, steps, renders all three cameras. But the robot is still a `FixedBaseRobot`: 7-number action (arm 6 + gripper 1), no base or torso control, and no `robot0_base_pos`, `robot0_base_quat`, `robot0_base_to_eef_*` observations. DEM's 16-number state cannot be built. The default start pose stretches the arm straight forward. |
| A 15-line subclass `XArm7Omron(XArm7)` with `default_base="OmronMobileBase"`, `default_arms={"right": "XArm7"}`, registered with `@register_robot_class("WheeledRobot")`, plus PandaOmron's controller config with RoboCasa's part order and legacy base controller set by hand, built with `robosuite.make` (not `create_env`) | **Works.** 12-number action with the same layout as PandaOmron, 16-number state builds, +z arm command moves the gripper up 10 cm in 20 steps, base forward command moves the base 10 cm in 20 steps. No robosuite or RoboCasa file was edited. Script: see "Reproduce" below. |

### Remaining gaps for an xArm 7 policy (not fixed)
1. **Gripper state differs.** `XArm7Gripper` reports 6 joint angles in radians (about 0.38 open, 0.81 closed). The Panda gripper reports 2 finger positions in metres (about 0.04 open, 0 closed). `policy.sim.state16` takes the first 2 numbers, so the state would be out of distribution. Map it or retrain.
2. **Start pose.** RoboCasa's Panda start pose is not used. The stock xArm pose reaches far forward (gripper 0.47 m ahead and 0.30 m up in the base frame, against 0.25 m and 0.59 m for the Panda). A matching tucked pose has to be chosen.
3. **Wrist camera pose** is Panda-specific (`pos [0.05, 0, 0]` on `robot0_right_hand`). On the xArm the gripper fills more of the view.
4. **Reach and mount height.** xArm 7 reach is 700 mm against the Panda's 855 mm (vendor figures, from memory, not checked). The Omron torso height may need a different value.
5. **No xArm demos.** The 500 demos per task are Panda demos. Options: replay the Panda end-effector paths on the xArm with OSC (works only where the xArm can reach), or collect new demos. The replay test must pass on the xArm before any xArm number counts.
6. **Real xArm control:** UFactory's SDK offers Cartesian position and servo modes. The 20 Hz delta OSC interface would be emulated by streaming absolute Cartesian targets. Rate and latency on the lab's unit are unknown.

### Reproduce
```bash
cd ~/apollo-lab/rua-harness
PYTHONPATH=. ~/miniconda3/envs/dem/bin/python - <<'EOF'
import numpy as np, robosuite, robocasa  # noqa
from robosuite.models.robots import XArm7
from robosuite.robots import register_robot_class
from robosuite.controllers import load_composite_controller_config

@register_robot_class("WheeledRobot")
class XArm7Omron(XArm7):
    @property
    def default_base(self): return "OmronMobileBase"
    @property
    def default_arms(self): return {"right": "XArm7"}
    @property
    def init_torso_qpos(self): return np.array([0.0])

cc = load_composite_controller_config(controller=None, robot="PandaOmron")
cc["composite_controller_specific_configs"] = {"body_part_ordering": ["right", "right_gripper", "base", "torso"]}
cc["body_parts"]["base"]["type"] = "JOINT_VELOCITY_LEGACY"
env = robosuite.make(env_name="PickPlaceCounterToCabinet", robots="XArm7Omron", controller_configs=cc,
    camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
    camera_heights=128, camera_widths=128, has_renderer=False, has_offscreen_renderer=True,
    use_camera_obs=True, ignore_done=True, use_object_obs=True, seed=0,
    obj_instance_split="target", layout_and_style_ids=list(zip(range(1, 11), range(1, 11))))
obs = env.reset(); print(env.action_dim, obs["robot0_base_to_eef_pos"])
EOF
```

## 3. Noise model

### Why the old one is not enough
`rua_harness.geometry.add_depth_noise` (simlib has no depth noise of its own) adds Gaussian noise times depth squared, zeroes pixels where the depth gradient is above 5 cm per pixel, and zeroes 1% of pixels at random. It misses: quantization, spatially correlated error, flying pixels, the occlusion shadow beside near objects, holes on dark or shiny surfaces, and range limits. Its gradient test also marks slanted surfaces (a counter seen at a grazing angle) as edges.

### The new model: `policy/noise/depth_noise.py`
`stereo_noise(depth, rng, cam, rgb=None, fx=None, scale=1.0)` applies, in order:

1. **Disparity noise.** Depth to disparity `d = f B / z` with the real camera's focal length `f` (pixels) and baseline `B`. Add Gaussian noise of `subpixel_rms` pixels, correlated over a small box (`corr_px`), times `scale`. This gives depth error `sigma_z = z^2 * subpixel / (f B)`, growing with depth squared.
2. **Quantization.** Disparity rounded to `subpixel_step` pixels, depth rounded to `depth_unit_m`.
3. **Edges.** A pixel is at a discontinuity when the second difference of inverse depth, relative to its inverse depth, exceeds `edge_rel`. Inverse depth is linear in pixels on any plane, so slanted surfaces are not flagged. Edge pixels are dropped (`edge_drop_prob`) or become flying pixels with a random depth between the near and far neighbour (`edge_fly_prob`).
4. **Occlusion shadow.** With depth aligned to the left imager, background pixels to the left of a nearer surface that the right imager cannot see get no depth. The band is as wide as the disparity jump in rendered pixels (needs `fx = K[0, 0]` of the rendered camera).
5. **Dark and shiny holes.** With `rgb`, pixels darker than `dark_lum` or brighter than `bright_lum` drop out with set probabilities.
6. **Random holes and range.** Independent holes (`hole_prob`), and no depth nearer than `min_z` or farther than `max_z`.

Missing depth is 0.0, as the RealSense SDK reports it.
`gaussian_noise(depth, rng, sigma_m)` is the GAE-style i.i.d. setting. `apply_level(name, ...)` applies one named sweep level.

### Presets
| | D435 (`D435`) | ZED 2 / 2i (`ZED2`) |
|---|---|---|
| Baseline | 50 mm | 120 mm |
| Horizontal FOV, native width | 87 deg at 848 px (f = 447 px) | 110 deg at 1280 px (f = 448 px) |
| Sub-pixel RMS | 0.08 px | 0.18 px (fitted to "1% at 3 m"; pessimistic) |
| Disparity step, depth unit | 1/32 px, 1 mm | none, none |
| Range | 0.195 to 3 m | 0.3 to 20 m |
| Depth RMS at 0.5 / 1 / 2 m | 0.9 / 3.6 / 14 mm | 0.8 / 3.3 / 13 mm |
| Edges | drop 0.5, fly 0.3 | drop 0.2, fly 0.5 (neural depth fills edges with flying pixels) |
| Dark / shiny holes | 0.3 / 0.3 | 0.05 / 0.1 (neural depth fills) |

Edge, hole and correlation numbers are chosen by hand to look plausible. They are not measured.

On one real RoboCasa frame (`PickPlaceCounterToCabinet`, left camera, 256 px, depths 0.39 to 1.79 m) the D435 preset leaves 88% of pixels valid, with median absolute error 2.1 mm and 90th percentile 8.8 mm among valid pixels. The 1 cm Gaussian setting gives median 6.7 mm; the 10 cm one gives median 68 mm. The robot arm casts a visible shadow band on its left, as a real D435 does.

### References
Marked **[fetched]** when I read a source on Nov 27, **[search]** when I saw it only in search-result summaries, **[memory]** when not checked.

- Intel RealSense depth RMS formula `RMS = z^2 * subpixel / (f * B)`, `f = 0.5 * Xres / tan(HFOV / 2)`, D435 baseline about 50 mm, well-textured sub-pixel RMS below 0.1 px: Intel self-calibration white paper (intelrealsense.com/download/9277) and Intel community threads **[search]**. The original "Tuning depth cameras for best performance" page (dev.intelrealsense.com) did not resolve. A community example with 0.08 px, 848 px and 50 mm gives 2.9 mm at 1.07 m, consistent with the preset.
- Keselman, Woodfill, Grunnet-Jepsen, Bhowmik, "Intel RealSense Stereoscopic Depth Cameras", CVPR Workshops 2017, arXiv 1705.05548 **[fetched abstract only]**.
- Ahn, Chae, Noh, Nam, Hong, "Analysis and Noise Modeling of the Intel RealSense D435 for Mobile Robots", Ubiquitous Robots 2019: axial and lateral noise as Gaussians in distance and incidence angle **[search]**.
- Nguyen, Izadi, Lovell, "Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking", 3DIMPVT 2012: axial noise grows with distance squared, lateral linearly **[search; coefficients not checked]**.
- D435 HFOV 87 deg, min-Z about 0.195 m at 848 x 480, 1 mm default depth unit, 1/32 px disparity step: Intel D400 datasheet **[memory]**.
- ZED 2i: baseline 120 mm, range 0.3 to 20 m, FOV 110 x 70 deg with 2.1 mm lens, depth up to 100 Hz: stereolabs.com/en-us/products/zed-2 **[fetched]**. Accuracy "below 1% up to 3 m, below 5% up to 15 m" **[memory]**.
- ZED depth is least reliable at object edges (flying pixels, removed by the confidence threshold) and on textureless areas: docs.stereolabs.com depth settings page **[fetched]**.
- GAE (Liu et al., ICML 2026, arXiv 2510.03896) depth noise ablation 0 / 1 / 10 / 50 cm: success 0.57 / 0.49 / 0.33 / 0.21 with their action expert, 0.42 / 0.30 / 0.11 / 0.04 with IK. From our full-read notes of Nov 26; the notes do not state the exact noise form, so I assume i.i.d. Gaussian with that std.

### Evaluation-time noise sweep (design, not run)
Goal: show how each command format degrades when the depth behind its 3D points is realistic. Training stays on clean depth unless a run says otherwise.

| Level | What | Why |
|---|---|---|
| `clean` | simulator depth | upper bound |
| `gauss_1cm` | i.i.d. Gaussian, 1 cm | matches GAE's 1 cm point |
| `gauss_10cm` | i.i.d. Gaussian, 10 cm | matches GAE's 10 cm point |
| `d435_no_holes` | D435 disparity noise and quantization only | separates noise from missing data |
| `d435` | full D435 model | realistic short-range stereo |
| `d435_x2`, `d435_x4` | disparity noise times 2 and 4 | bad texture, poor calibration, far objects |
| `zed2` | full ZED 2 model | realistic wide-baseline neural stereo |

Protocol:
- Apply noise only where depth is consumed: the agent's pixel-to-3D lifting (`simlib.camera.pixel_to_world`) and any point-map input to the policy. Images are unchanged. Grading still uses simulator ground truth.
- Noise seed per (episode, step) from the episode seed, so every format sees the same noise.
- Formats: `points_3d` and `pose_path` (depth-dependent), with `sentence`, `structured` and `points_2d` as controls that should not move (a sanity check: if they move, depth is leaking somewhere it should not).
- Tasks: the 3 smoke-test tasks first, then all 18 `atomic_seen`. Episodes: 50 per task per level at the start; this is 8 levels x 5 formats x 3 tasks x 50 = 6,000 episodes for the smoke round.
- Missing depth at the chosen pixel: take the median of valid depth in a 5 x 5 window; if none, count the command as failed and log it. Report the rate of such failures per level.
- Also report the 3D error of lifted command points against simulator truth per level, so success drops can be tied to point error.
- Every run writes a CSV to `results/` with the noise level in a column, per `results/README.md`.
- Optional second sweep: add calibration error (camera pose off by 1 cm and 1 deg, then 2 cm and 2 deg) on top of `d435`, since on a real robot extrinsic error is often as large as depth error.

## 4. Lab facts to collect (for Xiatao)
Questions to put to him, written as facts we need.

1. **Arm:** which arm the lab has for this project (xArm 7, xArm 6, Lite 6, Panda, other), its controller box firmware version, and whether it exposes Cartesian servo or velocity streaming at 20 Hz or faster.
2. **Gripper:** model (xArm gripper, Robotiq 2F-85, other), and how its open width is read.
3. **Base:** mobile base or fixed table mount. If mobile, which base and its velocity interface. If fixed, table height relative to the arm base.
4. **Cameras:** which depth cameras the lab owns (D435, D455, D405, ZED 2i, ZED X, other), how many, and where they can be mounted (wrist, fixed on the base, on a tripod). Distances from the cameras to the work area.
5. **Calibration:** whether the lab has a hand-eye calibration routine and a calibration board.
6. **Compute:** which GPU machine sits next to the robot, and the network path from it to the arm controller (for latency).
7. **Control rate and latency:** measured command-to-motion delay on the arm.
8. **Scene:** whether a real kitchen counter, cabinet, drawer and microwave are available, or a tabletop proxy.
9. **Safety:** workspace limits, speed limits, and who must be present when the robot runs.
10. **Timeline:** whether a real-robot experiment is expected for this paper or later.
