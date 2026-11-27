# `robot.run_skill`: the trained policy as HexaShell's 15th robot call

Code: `policy/hexashell_tool/tool.py`. Test: `tests/test_hexashell_tool.py`. Background on HexaShell's
RoboCasa port: `docs/hexashell_robocasa_notes.md`.

## What the model writes

HexaShell's model has two LLM tools, `python(code)` and `finish(...)`. Robot calls are methods of
`robot` inside the Python cell. `run_skill` is one more method:

```python
r = robot.run_skill("open", object={"name": "drawer", "camera": "left", "uv": (72, 212)}, max_seconds=10)
print(r)
```

```
Skill: open ran 200 steps (10.0 s), stopped at its time limit  [550 steps left]
  sent: 'skill open object drawer'
  object point (0.612, -0.083, 0.795) from left (72, 212) on the depth picture
[picture attached: left, right at the end]
```

Arguments (same fields as `rua_harness.command.TOOL_SCHEMA`):

| argument | value |
|---|---|
| `skill` | `open`, `close`, `pick_place`, `slide`, `press`, `turn`, `insert`, `navigate` |
| `object`, `target` | a name (`"drawer"`), or `{"name": ..., "camera": "left" \| "right" \| "wrist", "uv": (x, y)}`. `uv` is HexaShell's pixel order: column, row, from the top left. `"xy"` is accepted for `"uv"`. |
| `source` | a name, e.g. `"counter"` (pick_place only) |
| `direction` | `on off in out up down left right` |
| `max_seconds` | robot time, default 30, capped at 30 and at the episode's steps left |

Which fields each skill takes is `rua_harness.command.SKILLS`; a wrong set raises `ValueError` and nothing moves.

## What the model gets back

* A `Skill` object that prints as above (HexaShell's style: one line first, then `note:` lines). Fields:
  `.skill .words .reason .steps .seconds .points .cams .notes .steps_left`.
* The final left and right pictures (and wrist, if the env renders it), through the host's
  `attach("run_skill", image, caption)`. HexaShell sends them after the cell's text, as with `robot.look`.
* Errors as HexaShell's kinds: `ValueError` (bad fields, pixel outside the picture), `NotFound` (no surface
  behind the pixel), `RobotError` (no steps left, checkpoint not trained for this task, `pose_path` checkpoint).

Never returned: `grading_only` (the simulator's success flag). It goes only to the harness record
(`log` callback and `host.last`). `reason == "success_check"` prints as "stopped before its time limit".
In HexaShell the episode then ends at once and its own "episode is over" message tells the model, as it
does for every call (docs/decisions_needed.md item 2).

## What the harness does

1. Checks the fields with `rua_harness.command.validate`.
2. Reads the checkpoint's format from its `config.json`.
3. For point formats (`points_3d`, `points_3d_rgb`, `points_2d`): lifts each given pixel to a world point
   from the depth picture (`simlib.camera`). The depth comes from the obs when it has depth at the
   model's picture size; otherwise the camera is rendered at that size (`pixel_size=512` in HexaShell).
   A ref without a pixel runs with that point masked (`has` false) and a note says so.
   For `sentence` and `kv_text` checkpoints the pixel is not used, and a note says so.
4. Builds the command: `kv_text = to_policy_text(cmd)`, `grasp_world` = object point, `place_world` =
   target point, `has` = which are present (same keys as `policy.commands_gt.perfect_command`). Sentence
   checkpoints get `to_sentence(cmd)` instead.
5. Calls `policy.run_skill.run_skill(env, words, max_seconds, checkpoint, task, command, obs)` on a fresh obs.

## Seams (constructor arguments of `RunSkillHost`)

| seam | our env | HexaShell |
|---|---|---|
| `env` | robosuite env from `policy.sim.make_env(..., depth=True)` | `RenderedObsEnv(sim.env, cams=("left","right"), size=256, step_fn=sim.step)` on the server's `CasaSim` |
| `attach(kind, image, caption)` | test collects them | `ArmHost.attach` |
| `log(event)` | test collects them | `ArmHost.log` (the episode record) |
| `steps_left()` | constant | `ArmHost._left` |
| `pixel_size` | obs size (256) | 512 (HexaShell's pictures) |
| `to_agent_frame(p)` | identity (world) | world -> start frame: `sim.to_start @ [p, 1]` |
| `runner` | `policy.run_skill.run_skill` | same |
| errors | `ToolError(kind, text)` | `raise CallFailed(e.kind, e.text)` |

`call_tool(host, arguments)` runs the JSON form (TOOL_SCHEMA arguments) and returns
`{"text", "images", "error"}`, for a harness that exposes `run_skill` as its own LLM tool instead of a
`robot.*` method.

## Dropping it into HexaShell (not done; HexaShell was read, not run)

1. `hexashell/benches/robocasa/proxy.py`: add the `run_skill` method (copy `RunSkillProxy.run_skill`, its
   docstring is what `help(robot.run_skill)` prints) and add it to `__repr__`.
2. `hexashell/benches/results.py`: add the `Skill` dataclass. Results cross host -> worker by pickle and the
   worker imports result modules before its sandbox closes, so the class must live there (or be imported
   by the proxy module at load).
3. `hexashell/benches/robocasa/host.py`: `PUBLIC += ("run_skill",)` and a `_run_skill` that forwards to the
   server (`self._call("run_skill", ...)`), turning a `ToolError` into `CallFailed`.
4. `hexashell/benches/robocasa/server.py`: `op_run_skill` builds a `RunSkillHost` on
   `RenderedObsEnv(self.sim.env, step_fn=self.sim.step)` and runs it in the server process, which owns the
   env. It must step through `CasaSim.step` so HexaShell's step count, horizon and success check stay its
   own. The server reply carries the text fields and the final images as numpy blobs (`benches/wire.py`
   sends arrays). The server needs torch, DEM and this repo on its Python path.
5. `prompts/robocasa/03_api.md`: add `API_ROW` from `tool.py` to the motion table.
6. Record each change in `docs/hexashell_changes.md`.

## Tested

`tests/test_hexashell_tool.py` (OpenDrawer, seed 6, 2 checkpoints at step 50, so the motions mean nothing):
* the proxy call pickles its arguments and result across, as HexaShell's worker does;
* points checkpoint (`f3_points_3d_rgb`): a pixel on the drawer handle (found from simulator truth to play
  the agent) is lifted to within 4 cm of the handle (1.2 cm measured), `has == [True, False]`,
  `kv_text == "skill open object drawer"`, 30 steps run, two 256 x 256 pictures attached, no "success" in the text;
* sentence checkpoint (`train_test`): sends "Open the drawer.", notes the unused pixel, 20 steps;
* bad fields return `ValueError`.

Not tested: `RenderedObsEnv` with HexaShell's `CasaSim` (its images matched the obs images on our env),
`pixel_size=512`, `to_agent_frame`, the wrist camera.

Known problem (Mac, not this code): after a second hard reset of the env, loading the model makes later
renders striped or noisy (reproduced with plain `policy.run_skill.run_skill`, on cpu and mps). The test
resets once and checks that frames are not garbage. See `docs/hexashell_robocasa_notes.md`.
