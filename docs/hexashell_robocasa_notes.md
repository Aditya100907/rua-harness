# HexaShell's RoboCasa port: how it works, and how to add `run_skill`

Source: the public HexaShell repo at commit `914d482` (Nov 25), shallow clone. I read the code; I did not run
any of it. Each point is marked **[read]** (seen in the code or docs, file named) or **[inferred]** (my
conclusion, not checked by running). Paths are relative to the HexaShell repo.

## 1. How the model calls the robot

* **[read]** The LLM has exactly two tools: `python(code)` and `finish(status, summary)`
  (`hexashell/runtime/episode.py`, `TOOLS`). Robot calls are not LLM tools. They are methods of a `robot`
  object that the model's Python cell calls, e.g. `robot.goto((0.6, 0, 0.9), pitch=90)`.
* **[read]** The model learns the calls from the system prompt, `prompts/robocasa/01_rules.md` to
  `06_checking.md`. `03_api.md` is a table: call, what it does, what it returns, plus printed examples of
  each result. `help(robot.goto)` in a cell prints the method's docstring (`kernel/worker.py`, `_help`).
* **[read]** The "14 general calls" in the README are goto, gripper, home, wait, look, find, measure,
  to_world, to_pixel, task, state, things, image, camera. The RoboCasa robot actually answers 19:
  `ARM_CALLS` in `hexashell/benches/armhost.py` (those 14 plus arc and depth) and RoboCasa's own
  drive, push_arc, rim_pick (`benches/robocasa/host.py`, `PUBLIC`).

### Where a call is declared

| layer | file | what it holds |
|---|---|---|
| proxy (worker side) | `hexashell/benches/robocasa/proxy.py`, class `Robot` | one method per call: signature, docstring, body `return self._call("goto", (xyz,), dict(...))` |
| host | `hexashell/benches/armhost.py` (`ArmHost`), `benches/robocasa/host.py` (`CasaHost`) | `public` tuple of names; `dispatch(name, args, kwargs)` calls `self._<name>(...)`; checks arguments, turns pixels to 3-D, plans motions |
| server | `hexashell/benches/armserver.py` (`ArmServer`), `benches/robocasa/server.py` (`Server`) | `handle(op, args)` calls `op_<name>`; owns the RoboCasa env and steps it |
| results | `hexashell/benches/results.py`, `hexashell/robot/results.py` | dataclasses (Move, Grip, Homing, Drive, Waited, Status, TaskInfo, Spot, Thing, Mask, Picture) whose `__repr__` is the text the model reads |
| prompt | `prompts/robocasa/03_api.md` | the table the model reads |

### How a call travels **[read]**

1. The cell runs in a sandboxed worker process (`python -I`, `kernel/worker.py`). `Robot._call` is
   `Worker._rpc`: it pickles `("call", id, name, args, kwargs)` to the host over a Unix socket pair.
2. The episode process unpickles with a whitelist (`kernel/wire.py`, `HostUnpickler`: plain data, numpy
   arrays, and classes from the two results modules) and calls `CasaSession.dispatch` ->
   `CasaHost.dispatch` (`armhost.py`, line 226).
3. `dispatch` refuses unknown names (`RobotError`), runs `_<name>`, and maps errors to
   `CallFailed(kind, text)`: `Refused` -> ValueError, `NotFound`, `NotExecuted` (a motion that cannot
   run), `TypeError`. If the episode is over it raises `Stop`, which ends the cell.
4. Motions go on to the simulator server: `ArmHost._call(op, **args)` -> `BenchSession.call` ->
   `SimServer.call` (`benches/session.py`) over a second socket, with its own non-pickle format
   (`benches/wire.py`: JSON header plus numpy blobs, because the two processes run different Python and
   numpy versions). The server's `serve` loop calls `op_<name>` and replies `{ok, result, status, log, proprio}`.
5. The host builds the result object (e.g. `R.Move(...)`) and returns it; the worker gets it by pickle and
   raises the matching exception type for errors (`Worker._rpc`).

### What a call returns to the model **[read]**

* Text: the cell's printed output plus the value of its last expression, then a status line with the step count and the reply count
  (`runtime/episode.py`, `_cell`). Result objects print as one line or a
  few, problems first, then `note:` lines, e.g.
  `Move: at (0.612, -0.083, 0.981) pitch 90 yaw 12 roll 0  ±2 mm 1°  18 steps  [842 steps left]`.
* Images: only when the cell calls something that uses `attach(kind, image, caption)`, which in the
  RoboCasa host is only `robot.look`. The pictures go in one `user` message after the tool results
  ("Pictures from robot.look() in this turn:"), at most 8 per turn (`PICTURES_PER_TURN`), JPEG quality 90,
  with a window of 90 live pictures (`config/benches.yml`). At the start, one picture per camera is
  pinned in the first message (`session.start_pictures`).
* `robot.image(cam)` and `robot.depth(cam)` return numpy arrays to the cell, not pictures to the model.
* The success check ends the episode by itself: the call running at that moment reports "The episode is
  over: the environment reports SUCCESS" (`prompts/robocasa/01_rules.md`). So the model does learn the
  outcome, through the episode ending.

## 2. Processes **[read]**

1. **Episode process** (`hexashell.benches.run --bench robocasa --task T --seed S`): the LLM loop
   (`runtime/episode.py`), the kernel host, `CasaSession` and `CasaHost`. **[inferred]** Its Python is separate from
   RoboCasa's (the server is started with `--python`, the benchmark's own interpreter).
2. **Sandbox worker**: runs the model's cells; no network, no API key, CPU limit.
3. **Simulator server**: `python -m hexashell.benches.robocasa.server <fd>` in RoboCasa's own Python
   (CPython 3.12, MuJoCo 3.3.1, numpy 2.2.5), started per episode or kept warm by `benches/daemon.py`.
   Rendering on `$HEXA_EGL_DEVICE`.
4. **SAM 3 service**: HTTP at `sam3_url` (default `127.0.0.1:18300`, `services/`), shared by episodes;
   used by `find` and `measure`. Without it, `find`/`measure` refuse and `to_world`/pictures still work
   (`armhost.py`, `_need_scene`).

## 3. The environment **[read]** (`benches/robocasa/sim.py`, `protocol.py`)

* `robosuite.make(env_name=task, robots="PandaOmron", controller_configs=load_composite_controller_config(controller=None, robot="PandaOmron"), camera_names=[left, right, wrist], camera_widths=512, camera_heights=512, has_offscreen_renderer=True, use_camera_obs=False, use_object_obs=True, translucent_robot=False, generative_textures=None, randomize_cameras=False, obj_instance_split="pretrain", ignore_done=True, ...)`.
* Cameras: `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`, rendered on demand at
  512 x 512 with depth (`CasaSim.observe`, flipped to top-row-first, depth in metres via robosuite's
  `get_real_depth_map`) and a robot mask from segmentation. Not part of the env's obs.
* Frame: everything the model sees is in the **start frame** (base at episode start, floor origin, x
  forward, y left), not MuJoCo world. `CasaSim.to_start` converts.
* Control: 20 Hz, the official 12-number action of HYBRID_MOBILE_BASE in robosuite order, always arm
  mode. HexaShell's own controllers turn goto/arc/drive into those actions (`control.py`, `benches/arm.py`).
* Episode end: `CasaSim.step` calls `env._check_success()` after every step; the episode is over at the
  first success or at the official horizon (`protocol.HORIZON`, e.g. OpenDrawer 750 steps).
* Splits: `pretrain50` (default; 50 fixed pretraining kitchens, seed s -> kitchen `PRETRAIN50[s % 50]`,
  seeds 0-9 learn, 10-19 validation, 20-49 test) and `pretrain` (RoboCasa's own evaluation split, layouts
  and styles 11-60). **No `target` split.** Objects: `obj_instance_split="pretrain"`.
* Seeds: `reset(seed)` sets `env.seed`, `env.rng = default_rng(seed)`, `np.random.seed(seed)`, and makes
  unseeded `np.random.default_rng()` calls inside the reset draw from a seeded stream (`_seeded_fallbacks`).
* Limits: 40 replies per atomic task, 100 per composite task, 2 h wall clock per episode.

### Compared with our setup

| | HexaShell | ours (DEM training and `policy/eval.py`) |
|---|---|---|
| kitchens | pretrain layouts/styles 11-60 | target layouts 1-10 (dataset env kwargs) |
| objects | `obj_instance_split="pretrain"` | `"target"` |
| `clutter_mode` | not set | 1 |
| controller config | robosuite default for PandaOmron | dataset's; **[read, checked here]** the only difference is `composite_controller_specific_configs` (IK offsets), which **[inferred]** HYBRID_MOBILE_BASE with OSC does not use |
| images | 512, rendered on demand | 256, camera obs |
| reset seeding | the four steps above | `env.rng = default_rng(seed); env.reset()` |
| frame | start frame | world |
| RoboCasa | `b4684e6` + patches | `425684d`, unpatched |

**[inferred]** To get agent-alone numbers on our split, `protocol.py` needs a `target` split (the 10
target kitchens, `obj_instance_split="target"`, `clutter_mode=1`, i.e. the dataset env kwargs). Same seed
numbers will not give the same scenes as `policy/eval.py`, because the reset procedures differ. Agent
alone and agent plus policy are comparable as long as both run through HexaShell.

## 4. RoboCasa patches **[read]** (`THIRD_PARTY_VERSIONS.md`, `scripts/patches/robocasa.patch`, `scripts/patch_third_party.py`)

* `robocasa/__init__.py`: the exact-version asserts on MuJoCo and numpy are commented out.
* `robocasa/models/__init__.py`, `environments/kitchen/kitchen.py`, `utils/texture_swap.py`: the asset
  folder can be moved (`$HEXA_ROBOCASA_ASSETS`); the kitchen.py hunk only rewrites asset file paths in
  the model XML.
* `models/objects/objects.py`, `utils/model_zoo/mjcf_obj.py`: per-episode object XML goes to a temp file.
* `models/fixtures/counter.py` (`[roboshell-determinism]`): `list(set(valid_geoms))` becomes
  `list(dict.fromkeys(valid_geoms))`, so counter regions do not depend on memory addresses.
* Their statement: "None of the changes touches tasks, success checks, controllers or physics."
  **[read]** The patch file touches only those six files, and none of the hunks is in a task file or a
  `_check_success`. **[inferred]** Success checks are unchanged. The determinism patch can change which
  scene a seed builds compared with unpatched RoboCasa, so our unpatched `425684d` and their patched
  `b4684e6` will not build the same scenes from the same seed. Whether `425684d` and `b4684e6` differ in
  task code I did not check.

## 5. Adding a 15th call (`run_skill`)

The code for it is `policy/hexashell_tool/` (interface: `policy/hexashell_tool/interface.md`). It mirrors
the shape above without importing HexaShell. What William would change in HexaShell:

1. **Proxy** `benches/robocasa/proxy.py`: add `run_skill(self, skill, *, object=None, target=None, source=None, direction=None, max_seconds=None)`
   with its docstring, body `return self._call("run_skill", (skill,), dict(...))`. Copy it from
   `RunSkillProxy`. Add the name to `Robot.__repr__`.
2. **Result type** `benches/results.py`: add `Skill`. **[read]** Results cross host -> worker by
   pickle, and the worker imports the results modules before its sandbox closes
   (`proxy.py` imports them for that reason), so a result class must live in one of those modules.
   Arguments cross worker -> host through the whitelist; `run_skill`'s arguments are plain dicts, lists and
   strings, so they pass.
3. **Host** `benches/robocasa/host.py`: `PUBLIC += ("run_skill",)`; `_run_skill` sends the call to the server
   and maps `ToolError(kind, text)` to `CallFailed(kind, text)`. It should `attach` the final pictures and
   `log` the command and the grading flag (the record), and return the `Skill`.
4. **Server** `benches/robocasa/server.py`: `op_run_skill`. The policy has to run where the env is, i.e. in
   the server process. Build `RunSkillHost(RenderedObsEnv(self.sim.env, step_fn=self.sim.step), checkpoint=..., task=self.sim.task, pixel_size=512, to_agent_frame=<world to start frame via self.sim.to_start>)`.
   * **[read]** The server's env has `use_camera_obs=False`, so `RenderedObsEnv` renders left and right
     at 256 for the policy after each step.
   * **[read]** Stepping must go through `CasaSim.step`, not `env.step`, or HexaShell's step count,
     horizon and success check miss the skill's steps. `RenderedObsEnv(step_fn=sim.step)` does that.
   * **[inferred]** The server Python (3.12) needs torch, the DEM package and this repo importable, and a
     GPU for speed. The model loads once per server (`run_skill` caches it), so use warm servers.
   * The reply carries the text fields and the final images as arrays (`benches/wire.py` sends numpy).
5. **Prompt** `prompts/robocasa/03_api.md`: add `API_ROW` (in `tool.py`) to the motion table, and one
   example print of `Skill` under it. Decide whether skills/*.md mention it.
6. Record every change in `docs/hexashell_changes.md` (does not exist yet).

Choices William should confirm:
* **[inferred]** The model's pixels refer to 512 pictures; the harness lifts them on a 512 depth render.
  The policy still sees 256.
* The printed points are in the start frame (the model's frame), lifted from depth, not ground truth.
* `max_seconds` cap 30 s. One call can use 600 of OpenDrawer's 750 steps.
* `reason == "success_check"` is shown as "stopped before its time limit". In HexaShell the episode ends at
  that step anyway and the next call (or the same cell) reports the episode over, so the skill adds no
  leak beyond HexaShell's own (decisions item 2).

## 6. Found while testing (our side, not HexaShell)

**[read, reproduced]** On Adi's Mac, after a second hard reset of a RoboCasa env, loading the DEM model
makes every later render corrupt (striped or noisy frames). Reproduced with plain
`policy.run_skill.run_skill` and with `PolicyRunner` alone, on cpu and mps; one reset is clean, and two
resets with random actions and no model are clean. **[inferred]** Likely the old sim's render context
being freed by garbage collection while the new one is current. `policy/eval.py` resets the env every
episode, so Mac evaluations after the first episode may feed the policy garbage images. Not checked on
Bouchet (EGL). Worth a frame-sanity check in `policy/eval.py` (mean row-to-row pixel change: clean about
6, corrupt 25 to 80) before trusting any Mac number. `tests/test_hexashell_tool.py` resets once and
asserts frames are clean.
