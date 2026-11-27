# simlib: shared simulator library

`simlib/` holds the RoboCasa helpers that both the agent side (`agent/`) and the policy side
(`policy/`) use. It has no LLM code and no policy code. Written Nov 26, 2026.

| module | what it does |
|---|---|
| `simlib/camera.py` | images, metric depth, intrinsics, extrinsics for every camera; pixel to world and world to pixel; segmentation |
| `simlib/tasks.py` | the 18 `atomic_seen` tasks, horizons, `make_env`, `check_success`, `language`, `reset_to_demo` |
| `simlib/truth.py` | ground truth for grading only: object positions, object geoms, the fixture part each task is about |
| `simlib/runner.py` | parallel episode runner: one process and one log per episode, results CSV rows |
| `simlib/episodes.py` | example episode functions (`random_actions`, `noop`, and `fake` for tests) |

Tests: `PYTHONPATH=. python -m pytest -q tests` runs everything (about 80 s on the Mac).
`-m "not sim"` skips the tests that build a real env (about 6 s). The `sim` marker is registered
in `pytest.ini`.

## Image and geometry conventions

Every image simlib hands out is **top-row-first** (row 0 = top), like the dataset videos.
robosuite gives bottom-row-first images; `views()`, `render_view()` and `segmentation()` flip
them once. Depth is metres along the optical axis. `T_world_cam` uses OpenCV axes (+z forward,
+x right, +y down). Pixels are `(u, v)` = (column, row). Details and the checks behind them:
`docs/geometry_notes.md`.

```python
from simlib.tasks import make_env
from simlib.camera import views, render_view, segmentation, pixel_to_world, world_to_pixel
env = make_env("OpenDrawer", seed=0)          # target split, 3 cameras, 256x256, depth on
obs = env.reset()
v = views(env, obs)["robot0_agentview_left"]  # View(name, rgb, depth, K, T_world_cam)
p = v.pixel_to_world([120, 140])              # world xyz of pixel column 120, row 140
uv, z = v.world_to_pixel(p)                   # back to (120, 140), z = depth
```

## Tasks, horizons, success

`simlib.tasks.ATOMIC_SEEN` is a copy of `TARGET_TASKS["atomic_seen"]`; `horizon(task)` reads
`ATOMIC_TASK_DATASETS[task]["horizon"]` (control steps). `check_success(env)` calls
`env._check_success()`. A test checks the copies against robocasa.

## Ground truth parts (grading only)

`simlib.truth.target_part(env, task)` returns a `Part` with `pos` (mean world position),
`positions` and `names` (one per matched element), `element` (geom, site, body, object,
interior, base_goal), `fixture`, `fixture_class`, `how`, and for tasks that move an object
`object`, `object_pos`, `dest_pos`. The rules live in `simlib.truth.TASK_TARGETS`.

How the part is found: take the fixture the task itself uses (the env attribute set in the
task's `_setup_kitchen_references`), then match regexes against the names of that fixture's geoms,
then sites, then bodies (fixture name prefix stripped). First match wins. For a body the
position is the mean of its geoms, because body origins are often hinges.

### The 18 tasks

Verified Nov 26, 2026: every task built with `make_env(task, seed)` and `target_part` found its
part for **all 18 tasks at seeds 0 to 4 (90 scenes, 0 failures)**. None of the 90 scenes counts
as a success at reset. "Seen" = the part's position projected into an agentview camera lands on
a pixel whose depth agrees within 6 cm (handle region boxes sit a few cm off the surface), in at least one of the two agentview cameras, in
that many of the 5 seeds. `tests/test_simlib_sim.py::test_target_part_found` repeats seed 0.

| task | horizon | target part | fixture classes seen | how it is located | seen |
|---|---|---|---|---|---|
| CloseBlenderLid | 900 | blender lid, loose beside the blender (dest: pitcher) | Blender | geom `auxiliary_*handle_main` (the lid's handle); dest body `pitcher_main` | 5/5 |
| CloseFridge | 900 | fridge door handle(s) | FridgeSideBySide, FridgeFrenchDoor | geom `fridge_*door_handle_main`. French door gives 2 handles; use `positions` (the mean is in mid-air) | 4/5 |
| CloseToasterOvenDoor | 450 | toaster oven door handle | ToasterOven | geom `door_handle_main` | 5/5 |
| CoffeeSetupMug | 600 | receptacle under the dispenser (object `obj` = mug) | CoffeeMachine | site `receptacle_place_site` | 5/5 |
| NavigateKitchen | 450 | robot base goal next to `env.target_fixture` | StandMixer, FridgeBottomFreezer, Oven, Sink, Microwave | `env.target_pos` (x, y; z set to 0). Success = base within 0.20 m and facing it | n/a |
| OpenCabinet | 1050 | cabinet door handle(s) | SingleCabinet, HingeCabinet | geom `door_handle_reg_main` (handle region box). HingeCabinet gives 2 handles, both doors must open | 5/5 |
| OpenDrawer | 750 | drawer handle | Drawer | geom `door_handle_reg_main` | 5/5 |
| OpenStandMixerHead | 450 | stand mixer head | StandMixer | body `head`, position = mean of its geoms (body origin is the hinge) | 5/5 |
| PickPlaceCounterToCabinet | 750 | cabinet interior (object `obj`) | HingeCabinet, OpenCabinet | centre of `fixture.get_int_sites(all_points=True, relative=False)` | 1/5 (cabinet above the cameras' view) |
| PickPlaceCounterToStove | 600 | pan on the stove (object `obj`) | Stovetop, Stove | `env.objects["container"]` body position | 5/5 |
| PickPlaceDrawerToCounter | 750 | the object in the open drawer; destination = any counter | Drawer | `env.objects["obj"]` body position; `dest_pos` is None | 5/5 |
| PickPlaceSinkToCounter | 900 | plate on the counter (object `obj`) | Sink | `env.objects["container"]` body position | 5/5 |
| PickPlaceToasterToCounter | 600 | plate (object `obj`, the toast) | Toaster | `env.objects["plate"]` body position | 5/5 |
| SlideDishwasherRack | 450 | top rack (rack1, the joint the task checks) | Dishwasher | geom `reg_rack1` (rack region box) | 0/5: found, but always behind something (inside the dishwasher) |
| TurnOffStove | 750 | knob of the burner named in the instruction | Stovetop, Stove | geom `knob_{env.knob}_main` | 5/5 |
| TurnOnElectricKettle | 450 | switch lever | ElectricKettle | geom `switch_main` | 5/5 |
| TurnOnMicrowave | 450 | start button | Microwave | geom `start_button` (the same geom the success check uses) | 5/5 |
| TurnOnSinkFaucet | 600 | faucet handle | Sink | geom `handle_main` | 4/5 |

Tasks whose part could not be located: none. Caveats: SlideDishwasherRack's rack is located but
hidden from both agentview cameras at reset; PickPlaceCounterToCabinet's interior is usually out
of frame; CloseFridge (French door) and OpenCabinet (HingeCabinet) have two handles, so use
`Part.positions`, not `Part.pos`.

## Episode runner

```bash
PYTHONPATH=. python -m simlib.runner --fn simlib.episodes:random_actions \
    --tasks OpenDrawer TurnOnMicrowave --episodes 5 --workers 4 --timeout 900 \
    --out results/2026-11-27_policy_only_random.csv
```

- `--fn module:function`: called in a fresh child process as `fn(task=, seed=, episode=, **kwargs)`.
  It returns a dict with `success` and optionally `llm_calls`, `input_tokens`, `output_tokens`,
  `cost_usd`, `seconds`. Without `seconds` the runner uses the child's wall time (env build
  included).
- Seeds: episode i uses seed `seed0 + i`. `--tasks atomic_seen` runs all 18.
- One log per episode: `<csv stem>_logs/<task>_epNNN.log` (stdout, stderr, and a final
  `# runner: status=...` line), plus the returned dict as `.json`.
- Crash or timeout: the child is killed, the row has `success=False`, the log says why.
- Rows go to the CSV in job order as each job finishes (appended, header written once) with exactly the columns of
  `results/README.md`: `task, episode, seed, success, seconds, llm_calls, input_tokens,
  output_tokens, cost_usd, code_commit`. `code_commit` is the short hash, `-dirty` if there are
  uncommitted changes to tracked files.
- Python API: `simlib.runner.run_episodes(fn_spec, tasks, episodes, out_csv, seed0, workers,
  timeout, kwargs)`.

Tested: fake episodes (parallel rows, crash, 3 s timeout) in `tests/test_simlib.py`; a real
OpenDrawer env for 5 random steps in `tests/test_simlib_sim.py`; and by hand 3 smoke tasks x 2
episodes x 30 random steps (0/6 success, 8 to 13 s each including env build, CSV in the
scratchpad, not in `results/`). No full-horizon run was made.

## Reset to a demo

`reset_to_demo(env, dataset_dir, episode, step=0)` loads a demo's model xml and ep_meta from
`extras/` and sets the recorded state, as robocasa's `playback_dataset.py` does. Used in
`scripts/render_geometry.py` to compare our render with the dataset video. The replay test
(do a demo's recorded actions succeed) is not in simlib yet.
