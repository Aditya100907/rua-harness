# Dataset notes: RoboCasa target, human demos (LeRobot format)

Checked Nov 26, 2026 on Bouchet under `/nfs/roberts/scratch/pi_dr877/ak3288/datasets/v1.0/target/atomic/`.

## A1. Episode counts and sizes
No download process was still running. Counts are `total_episodes` from each task's `meta/info.json`. Every task has at least 500.

| Task | Date folder | Episodes | Frames | Size |
|---|---|---|---|---|
| CloseBlenderLid | 20250822 | 502 | 183,099 | 658M |
| CloseFridge | 20250816 | 513 | 155,443 | 590M |
| CloseToasterOvenDoor | 20250818 | 505 | 86,401 | 350M |
| CoffeeSetupMug | 20250813 | 502 | 117,168 | 453M |
| NavigateKitchen | 20250821 | 500 | 72,786 | 451M |
| OpenCabinet | 20250813 | 500 | 184,024 | 750M |
| OpenDrawer | 20250816 | 514 | 137,431 | 455M |
| OpenStandMixerHead | 20250818 | 502 | 58,100 | 275M |
| PickPlaceCounterToCabinet | 20250811 | 502 | 131,904 | 534M |
| PickPlaceCounterToStove | 20250818 | 501 | 129,560 | 520M |
| PickPlaceDrawerToCounter | 20250820 | 501 | 155,749 | 590M |
| PickPlaceSinkToCounter | 20250813 | 501 | 194,952 | 651M |
| PickPlaceToasterToCounter | 20250817 | 512 | 148,353 | 582M |
| SlideDishwasherRack | 20250820 | 502 | 83,544 | 437M |
| TurnOffStove | 20250812 | 500 | 114,027 | 402M |
| TurnOnElectricKettle | 20250817 | 520 | 84,679 | 315M |
| TurnOnMicrowave | 20250813 | 543 | 80,439 | 324M |
| TurnOnSinkFaucet | 20250812 | 506 | 113,688 | 378M |

Total: 9,126 episodes, 2.23M frames, about 8.7 GB.

## A3. Layout of one task (OpenDrawer)
```
lerobot/
  data/chunk-000/episode_000000.parquet ...   one parquet per episode
  videos/chunk-000/observation.images.robot0_agentview_left/episode_000000.mp4
                   observation.images.robot0_agentview_right/...
                   observation.images.robot0_eye_in_hand/...
  meta/info.json modality.json tasks.jsonl episodes.jsonl episodes_stats.jsonl stats.json embodiment.json
  extras/dataset_meta.json                      env name, version, full env kwargs (controller config)
  extras/episode_000000/{ep_meta.json, model.xml.gz, states.npz}
```
Sizes for OpenDrawer: data 23M, videos 264M, extras 168M.

- **fps:** 20. Control frequency is also 20 Hz (`embodiment.json`: record, body and hand controller all 20).
- **Robot:** `PandaOmron` (Panda arm on a mobile base). `embodiment_tag: robocasa_panda_omron`.
- **Cameras:** 3 videos, each 256 x 256, h264, yuv420p: `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`. Videos are already upright. robosuite renders are upside down; RoboCasa's gym wrapper flips them with `img[::-1]`. `policy/sim.py:images` does the same.
- **Parquet columns:** `observation.state` (float64 [16]), `action` (float64 [12]), `annotation.human.task_description` (int64, index into `tasks.jsonl`), `annotation.human.task_name` (int64, index of the bare task name, e.g. 2 = "OpenDrawer"), `next.reward`, `next.done`, `timestamp`, `frame_index`, `episode_index`, `index` (global frame), `task_index`.
- **Instructions:** `tasks.jsonl` maps an index to a string. OpenDrawer has two: 0 "Open the right drawer.", 1 "Open the left drawer.", plus 2 "OpenDrawer" (the task name). `episodes.jsonl` also lists each episode's instruction and length. `ep_meta.json["lang"]` holds the same string.

### State vector (16), from `meta/modality.json`
| Slots | Key | Env observation |
|---|---|---|
| 0:3 | base_position | `robot0_base_pos` |
| 3:7 | base_rotation (quaternion) | `robot0_base_quat` |
| 7:10 | end_effector_position_relative (to base) | `robot0_base_to_eef_pos` |
| 10:14 | end_effector_rotation_relative (quaternion) | `robot0_base_to_eef_quat` |
| 14:16 | gripper_qpos | `robot0_gripper_qpos` |

The mapping comes from `robocasa/wrappers/gym_wrapper.py:PandaOmronKeyConverter.map_obs`. Note the base position is in the world frame, so it differs between kitchens. In OpenDrawer ep 0 the base did not move (slots 0:7 constant).

### Action vector (12), LeRobot order
| Slots | Key | Range in OpenDrawer ep 0 |
|---|---|---|
| 0:4 | base_motion (3 base velocity + 1 torso) | all 0 |
| 4 | control_mode | always -1 (arm mode) |
| 5:8 | end_effector_position (delta, OSC input in [-1, 1], scaled to 5 cm) | -0.87 .. 0.76 |
| 8:11 | end_effector_rotation (delta, scaled to 0.5 rad) | -0.55 .. 0.30 |
| 11 | gripper_close | -1 (open) or +1 (closed) |

robosuite's env takes a different order: eef pos 0:3, eef rot 3:6, gripper 6, base 7:11, mode 11 (`robocasa/utils/lerobot_utils.py:ACTION_KEY_ORDERING_HDF5`). `policy/sim.py:lerobot_to_env_action` reorders. Actions are relative (delta) OSC commands in the robot base frame (`input_type: delta`, `input_ref_frame: base`).

### Saved simulator states (`extras/`)
- `states.npz["states"]`: [T, 210] flattened MuJoCo state, one per frame, same T as the parquet.
- `model.xml.gz`: the full scene XML. `ep_meta.json`: layout id, style id, object configs, fixture refs (`{'drawer': 'stack_1_left_group_4'}`), camera configs, the instruction, initial base pose.
- `dataset_meta.json`: `env_version 0.5.1`, `robosuite_version 1.5.2`, `mujoco_version 3.3.1`, env kwargs: `obj_instance_split: target`, `layout_and_style_ids: (1,1)..(10,10)`, `clutter_mode: 1`, controller `HYBRID_MOBILE_BASE` with OSC_POSE arm (kp 150, delta, base frame), torso JOINT_POSITION, base JOINT_VELOCITY.

## Compared with what DEM expects
DEM (`dem/policy.py`, `DEMConfig`): `num_cameras=2`, `image_size=256`, `proprio_dim=16`, `action_dim=12`, `chunk=16`, `max_language_tokens=32`.
- 16-number state and 12-number action match the dataset exactly.
- Images are already 256, so `preprocess_images` does no resize.
- 2 cameras: the dataset has 3. The plan uses left and right agent view. The DEM README's inference example says "scene camera, wrist camera", which suggests the paper may have used one agent view and the wrist. See "Surprises" in the morning report. The frame store keeps all three so both can be trained.
- Normalization: the DEM research runs scaled each task's q01..q99 to [-1, 1], then z-scored (`DEM.load_research_checkpoints` docstring). `policy/data.py` does the same: stage 1 in the data loader, stage 2 in the policy's normalizers, both in `norm_stats.json`.
- Chunk: 16 predicted, 8 executed (20 Hz, so 0.4 s per replan).
