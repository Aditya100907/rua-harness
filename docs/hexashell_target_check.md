# HexaShell target split vs our evaluation: scene check

Question: does `python -m hexashell.benches.run --bench robocasa --split target --seed S` (HexaShell
`914d482` plus `setup/patches/hexashell_target_split.patch`) build the same scene as `policy/eval.py`
for the same task and seed? Written Nov 27 2026. Script: `policy/hexashell_target_check.py`.

Compared per (task, seed): layout id, style id, instruction, ep_meta objects (name, category, model),
fixture refs, every object's initial body position, robot base position, and the task's target part
position (`simlib.truth.target_part`). Positions must agree to 0.1 mm. Both sides loop the seeds in
order in one env per task, as `policy/eval.py` and HexaShell's warm server do.

## Bouchet, each side in its own environment (the real test)

CPU job 34895814 (`day`, MUJOCO_GL=egl on Mesa). Ours: `dem` env, RoboCasa `425684d` unpatched,
`policy.sim.make_env` + `env.rng = default_rng(seed); env.reset()`. HexaShell: its own venv
(`envs/hexashell-robocasa`, CPython 3.12, built by its `setup_robocasa.sh` in CPU job 34894783), RoboCasa
`b4684e6` with HexaShell's patches, `CasaSim(task, split="target").reset(seed)`. Seeds 1000 to 1004.
Each side run twice.

| comparison | result |
|---|---|
| ours run 1 vs HexaShell run 1 | **15/15 match** |
| ours run 1 vs ours run 2 | 15/15 |
| HexaShell run 1 vs run 2 | 15/15 |
| ours with HexaShell's counter fix vs HexaShell | 15/15 |

All position differences were 0.0 (to 5 decimals). Kitchens drawn (layout = style): seed 1000: 3
(OpenDrawer, TurnOnMicrowave), 6 (PickPlace); 1001: 10; 1002: 7; 1003: 3; 1004: 8.
CSVs: `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/runs/hexashell_target_check_5/`.

A 50-seed version (seeds 1000 to 1049, same five comparisons) is running as job 34894785 (about
2.5 h); results will land in `runs/hexashell_target_check/`.

## Mac (first pass, HexaShell's sim code in our env)

Run with our `425684d` (HexaShell's RoboCasa patches not applied). Two runs of our own evaluation code
disagreed on seed 1003 for all 3 tasks (layout 1 in one run, 3 in the other). Repeated 4 times in fresh
processes for OpenDrawer seed 1003: 2 runs drew layout 3 in one try, 2 runs failed a placement in layout 3
and retried into layout 1. With HexaShell's determinism fix monkeypatched in (`--counter-fix`), 6 of 6
runs agreed, and our scenes matched the target split 15/15.

Cause: `robocasa/models/fixtures/counter.py` line 813 in `425684d`, `list(set(valid_geoms))`, iterates
XML elements in memory-address order, so counter regions come in a different order per process.
HexaShell's `[roboshell-determinism]` patch uses `dict.fromkeys`. On Bouchet the 5 seeds happened to
agree; the 50-seed run will show how often this bites on Linux. Decision for Adi:
`docs/decisions_needed.md` item 6.

## RoboCasa `b4684e6` (HexaShell) vs `425684d` (ours)

`b4684e6` is 10 commits before `425684d`. In the `robocasa/` package the only differences are
`composite/setting_the_table/arrange_drinkware.py` (more excluded layouts), `models/fixtures/sink.py`
(`check_obj_under_water` present in `b4684e6`, gone in `425684d`), and new subtask-labeling scripts. None is
used by the atomic tasks, so scene sampling code is the same for the 3 smoke tasks; the Bouchet check
agrees.

## HexaShell's own tests

On the Mac in a venv with its `requirements.txt` (two RoboColiseum tests skipped at collection, they
need extra packages): 590 passed, 22 failed with the patch; 592 passed, 20 failed without it. The
failures are in sandbox and scripted-episode tests (`test_sandbox`, `test_runtime`, scripted episodes in
`test_benches`, `test_simple`) and vary between runs on the Mac with or without the patch (the sandbox
is built for Linux). No RoboCasa protocol or split test failed. Not run on Linux.
