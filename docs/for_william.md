# For William: what simlib gives the agent side

Written Nov 26, 2026 (overnight run). Draft from Adi's side; nothing was sent to you.

`simlib/` is a small library in this repo that both of us can use, so the agent and the policy
read cameras, check success and log results the same way. It does not touch `agent/`. Full
reference: `docs/simlib.md`. Geometry checks: `docs/geometry_notes.md`.

Run everything from the repo root with `PYTHONPATH=.` and the `dem` env.

## 1. Build an env and read the cameras

```python
from simlib.tasks import make_env, language, check_success, horizon
from simlib.camera import views

env = make_env("PickPlaceCounterToCabinet", seed=0)   # split="target", 3 cameras, 256x256, depth on
obs = env.reset()                                      # raw robosuite env, not the gym wrapper
print(language(env))                                   # "Pick the ... and place it in the cabinet."
v = views(env, obs)["robot0_agentview_left"]           # also robot0_agentview_right, robot0_eye_in_hand
v.rgb          # [256, 256, 3] uint8, row 0 = top (same as the dataset videos)
v.depth        # [256, 256] metres along the optical axis
v.K            # 3x3 intrinsics
v.T_world_cam  # 4x4 camera pose in the world, OpenCV axes (+z forward, +y down)
```

These are the pieces `EpisodeAPI` needs for images, depth and calibration. The robosuite obs
images are upside down (row 0 = bottom); `views()` fixes that, so do not flip again.
`T_world_cam` moves with the robot (the cameras sit on the mobile base and the wrist), so read it
at the same step as the image.

## 2. Pixel to 3D and back

```python
p = v.pixel_to_world([u, v_row])        # u = column, v_row = row, integer pixel indices
uv, z = v.world_to_pixel(p)             # continuous pixel coordinates, depth in metres
pts = v.point_map()                     # [H, W, 3] world points for every pixel
```

Plain-array versions are in `simlib.camera` (`pixel_to_world`, `world_to_pixel`,
`depth_to_points`). These were checked in the simulator: a pixel on an object lifts to within
3.2 to 3.5 cm of the object's origin from the agentview cameras. Use this to turn a SAM 3 mask
into a 3D point.

## 3. Success, horizon, grading

```python
check_success(env)   # env._check_success(), the task's own test
horizon("OpenDrawer")   # 750 control steps, from robocasa's dataset registry
```

For grading only (never feed these to the agent, CLAUDE.md rule):

```python
from simlib import truth
truth.object_position(env, "obj")             # true position of the object to move
part = truth.target_part(env, "OpenDrawer")   # the drawer handle, knob, button... for each task
part.pos, part.names, part.how
```

Use these to score your tool tests from the port spec ("a located object should be within
about 2 cm of the simulator's true position").

## 4. Run episodes in parallel and write results

Write one function per episode:

```python
# agent/episode.py (yours)
def run(task: str, seed: int, episode: int, **kwargs) -> dict:
    env = make_env(task, seed=seed)
    ...  # your LLM loop, stop at horizon(task) steps
    return {"success": check_success(env), "llm_calls": n, "input_tokens": i,
            "output_tokens": o, "cost_usd": c, "seconds": rollout_seconds}
```

Then:

```bash
PYTHONPATH=. python -m simlib.runner --fn agent.episode:run \
    --tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave --episodes 1 \
    --workers 3 --timeout 1800 --out results/2026-11-28_agent_only_smoke.csv
```

You get one process and one log per episode (`results/<csv name>_logs/<task>_epNNN.log`),
seed = episode index (shift with `--seed0`), and CSV rows with exactly the columns in
`results/README.md`, including `code_commit` (marked `-dirty` if there are uncommitted
changes). A crash or timeout gives `success=False`, and the log says why. `--tasks atomic_seen`
runs all 18. `--kwargs '{"model": "..."}'` passes options to your function.

The runner uses no LLM itself. `simlib.episodes:random_actions` is a no-LLM example to copy.

## What is not there yet

- No motion commands or robot-state helpers (EpisodeAPI's motion side is still yours).
- No gym wrapper: `make_env` returns the raw robosuite env (`env.step(a)` returns 4 values).
  `gym.make("robocasa/...")` also works but its images are not run through simlib.
- `simlib.tasks.reset_to_demo(env, dataset_dir, episode)` restores a demo scene, which may help
  you debug a tool on a known scene.
