# LIBERO as a second benchmark (Part S)

LIBERO (https://github.com/Lifelong-Robot-Learning/LIBERO) is a tabletop benchmark: a Panda arm, an OSC_POSE controller at 20 Hz, 130 language-conditioned tasks in suites. We use `libero_spatial` and `libero_object` first (10 tasks each, 50 human demos per task, 50 fixed init states per task for evaluation).

## Where things are on Bouchet
| What | Path |
|---|---|
| LIBERO code | `$PROJ/apollo-lab/LIBERO` (commit `8f1084e`, 2025-03-15) |
| Conda env | `$PROJ/envs/libero` (Python 3.11; separate from `envs/dem`; `pip freeze` in `envs/libero/pip_freeze.txt`) |
| LIBERO config | `$PROJ/libero_config/config.yaml` (`LIBERO_CONFIG_PATH`; without it LIBERO asks a question on first import) |
| Demos | `$SCR/datasets/libero/<suite>/<task_name>_demo.hdf5` (Hugging Face mirror `yifengzhu-hf/LIBERO-datasets`) |
| Job scripts | `policy/slurm/libero_{common.sh,setup,download,replay,train,eval}.sbatch` |
| Code | `policy/libero/` (`sim.py`, `data.py`, `replay_test.py`, `eval.py`, `train.py`, `random_episode.py`) |

`envs/libero` pins: robosuite 1.4.1 (LIBERO needs the 1.4 API; RoboCasa's robosuite 1.5 fork would break it), MuJoCo 2.3.7, numpy 1.26.4, bddl 1.0.1, gym 0.25.2. It also has torch 2.15.0 and DEM (editable) so evaluation runs the policy and the simulator in one process. LIBERO's own `requirements.txt` (torch-era transformers 4.21, robomimic, wandb, numpy 1.22) is not installed: those are only for LIBERO's lifelong-learning baselines. `envs/dem` was not touched.

## State and action layout, and how they map to DEM
DEM has no fixed layout; `DEMConfig(action_dim, proprio_dim)` is set per benchmark. RoboCasa uses 12 / 16. LIBERO uses 7 / 8.

**Action (7)**, passed to the env as is: `0:3` delta end-effector position, `3:6` delta end-effector rotation (axis-angle), both in [-1, 1] and scaled by the OSC controller; `6` gripper, -1 open, +1 close. No reordering (RoboCasa needs one). At evaluation the gripper is snapped to -1 / +1 by sign and the action clipped to [-1, 1].

**State (8)**, the layout openpi and OpenVLA-OFT use: `0:3` eef position (world frame), `3:6` eef orientation as axis-angle, `6:8` the two finger joint positions.
- From demos: `obs/ee_states` (6) and `obs/gripper_states` (2).
- From the live env: `robot0_eef_pos`, `quat2axisangle(robot0_eef_quat)`, `robot0_gripper_qpos`.
- The replay test checks that the two agree at t = 0 (column `state8_err0`).
- Not used: `obs/joint_states` (7 arm joints), `robot_states`.

**Cameras (2)**: `agentview` (scene) and `robot0_eye_in_hand` (wrist), the pair the DEM README describes. Demos store them at 128 x 128 upside down (OpenGL origin). `policy/libero/sim.py:upright` flips the rows (as `policy/sim.py:images` does for RoboCasa) and resizes to 256 bilinear. The same function serves the data loader and the live env, and the env renders at 128 like the demos, so training and evaluation see the same pixels. Note: OpenVLA rotates LIBERO images 180 degrees (flips both axes). Our choice (rows only) is the physically upright image; it only has to match between training and evaluation, which it does by construction.

**Normalization**: as RoboCasa (`policy/data.py`): per-task q01..q99 to [-1, 1], then a z-score over all tasks on the policy's normalizers. Task keys are `<suite>/<task_name>`.

**Language**: the task's `language_instruction` from the hdf5 `problem_info` (same string as `benchmark.get_task(i).language`). One instruction per task.

## Evaluation protocol (`policy/libero/eval.py`)
Standard LIBERO: per task, episode i starts from benchmark init state i (50 per task); 10 no-op steps so objects settle; predict 16 actions, execute 8; stop on `env.check_success()` or the suite step limit (spatial 220, object 280, goal 300, libero_10 520). One CSV row per episode, results/README.md columns plus `suite`, `model_seconds`. `--wiring` plays demo i's actions through the same loop with rendering, no policy.

## What was run and the numbers (Nov 27, 2026)
| Step | Job | Result |
|---|---|---|
| Env build (`libero_setup.sbatch`, day CPU) | 34847921 | env built and imports pass (numpy 1.26.4, MuJoCo 2.3.7, robosuite 1.4.1, torch 2.15.0+cu132, DEM). The final random-episode step failed (`libero` not importable, see "Fixes" below) |
| Download (`libero_download.sbatch`, day CPU) | 34848501 | libero_spatial 10 files, 5.9 GB; libero_object 10 files, 7.0 GB; about 25 s in total. 50 demos per task, 1,000 demos, 136,757 frames |
| Random-action episode, headless EGL on a CPU node (llvmpipe) | 34848528 (srun) | `RANDOM EPISODE OK`: env built and reset in 29.5 s, 50 random steps in 21 s with two 128 px cameras (about 0.4 s per step on 2 CPUs). Saved frames look upright: `runs/libero/random_episode.png` |
| Replay test, first 10 demos of tasks 0 to 2 per suite, no scene XML | 34848692 | spatial 10, 9, 9; object 9, 9, 7 (of 10) |
| Same, salad dressing task, with the demo's scene XML | 34849157 (srun) | 10/10 (was 7/10) |
| **Replay test, all 50 demos of all 20 tasks, with scene XML** | 34849177 (spatial), 34849178 (object) | see table below |
| Eval wiring: demo actions through `policy/libero/eval.py` with rendering | 34849041, 34849179 | spatial task 0: 3/3; object tasks 0 and 2: 5/5 each. Videos in `runs/libero/wiring_34849179/` |
| Data loader on real demos | login node | one task: 5,068 samples, images [2, 256, 256, 3] uint8, state [8], actions [16, 7]; loader frame matches the env frame orientation (`runs/libero/loader_sample.png`) |
| Training smoke test, `envs/dem`, CPU, `--offline` (random towers), 3 steps, both suites | 34849479 | `TRAIN DONE`, checkpoint has `action_dim 7`, `proprio_dim 8`; 1,000 episodes, 136,757 samples, 20 instructions |

Replay test with the scene XML. "first 10" is the requested test (at least 9 of 10 demos per task):

| Suite | Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| spatial | first 10 | 10 | 10 | 10 | 10 | 10 | **6** | 10 | 10 | 9 | 10 | |
| spatial | all 50 | 48 | 48 | 50 | 49 | 49 | 41 | 49 | 43 | 45 | 43 | 465/500 (93.0%) |
| object | first 10 | 9 | 10 | 10 | 10 | 9 | 9 | 9 | 10 | 10 | 10 | |
| object | all 50 | 47 | 48 | 49 | 47 | 44 | 45 | 48 | 48 | 50 | 48 | 474/500 (94.8%) |

**Result: 19 of 20 tasks pass the 9-of-10 test. libero_spatial task 5 ("pick up the black bowl on the ramekin and place it on the plate") fails: 6/10 with the scene XML, 7/10 without (job 34853733), 41/50 over all demos.** Holding still 30 steps after the last action adds only one success (object task 9). The bowl starts perched on the ramekin, so small contact differences likely change the outcome. Open-loop action replay of raw LIBERO demos is known not to be exact (OpenVLA re-recorded LIBERO and dropped demos that failed replay). This does not block training: the loader trains on the recorded observations, not on replays. Per-demo rows: `runs/libero/replay_34849177.csv`, `replay_34849178.csv` (also copied to `~/apollo-lab/runs/libero/` on the Mac). If a clean subset is wanted, these CSVs list the 61 demos that fail replay.

Other measured facts:
- State check at t = 0: the env's 8-number state after loading `states[0]` differs from the recorded `ee_states + gripper_states[0]` by a median of 0.005 (max 0.026). LIBERO's `scripts/create_dataset.py` records observations after each `env.step(action)` while `states` are before it, so the recorded obs may lead the stored state by one step. The measurement after one step (`state8_err_after1`) is about the same size, so this is not settled. Not checked: whether shifting actions by one step helps a trained policy.
- `max_state_err` (MuJoCo state drift from the recording) is large (0.2 to 10) even in successful replays, so the flattened states do not match exactly. Success is what counts.

### Fixes needed along the way (all in the repo)
- `pip install -e LIBERO` does not make `libero` importable (the top folder has no `__init__.py`); a `.pth` file adds the clone to the path.
- `benchmark.get_task_init_states` fails on torch 2.6 and later (`weights_only` default); `sim.init_states` loads the same file with `weights_only=False` (file from the LIBERO clone).
- The demo scene XML points at the LIBERO authors' machine (`/Users/yifengz/.../chiliocosm/assets/`); `sim.reset_to_demo` rewrites those paths to the clone's assets.
- `imageio-ffmpeg` was missing for wiring videos; installed.

### Not verified
- Any trained or untrained DEM checkpoint on LIBERO (no GPU job was launched). The eval runner's `act` was run only on the Mac with an offline checkpoint, which stops at tokenization as expected (no tokenizer offline); a real checkpoint uses the same code path as RoboCasa's `PolicyRunner`.
- Evaluation speed on a GPU node (rendering is EGL; on a GPU node it should use the GPU, not llvmpipe).
- libero_goal and libero_10 (not downloaded).

## Launch command for a stock-DEM LIBERO run (NOT launched)
Training runs in `envs/dem` (it needs only h5py and opencv, both there, plus the cached DINOv3 / NeoBERT weights), through `policy/libero/train.py`, which calls `policy.train.main` unchanged with the LIBERO dataset and `DEMConfig(action_dim=7, proprio_dim=8)`. Same recipe, checkpoints, resume and self-resubmission as RoboCasa runs.

```bash
# on a login node, after: squeue -u ak3288   (check the GPU cap and that no libero_train job exists)
cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
RUN=libero_spatial_dem ARGS="--suites libero_spatial --steps 30000 --batch 64" \
  sbatch --job-name=libero_spatial_dem --partition=gpu --constraint="gpu:l40s|gpu:a40" --time=08:00:00 \
  policy/slurm/libero_train.sbatch
# then evaluate (in envs/libero, GPU):
CKPT=/nfs/roberts/project/pi_dr877/ak3288/checkpoints/libero_spatial_dem/ckpt_0030000.pt SUITE=libero_spatial \
  sbatch policy/slurm/libero_eval.sbatch
```
Zero test first (rule: tests before trust): train with `--steps 0` is not supported, so build an untrained checkpoint by running the same command with `--steps 1 --lr-head 0 --lr-vision 0` into a separate RUN and evaluate it with `EPISODES=5`; expect about 0%.

The step count is a starting guess, not tuned: 30k steps at batch 64 is about 1.9M samples. libero_spatial plus libero_object have 1,000 demos and 136,757 frames together (from the training smoke test), and libero_spatial alone has about 50k (task 0 has 5,068 frames), so 30k steps is about 38 epochs of it.
