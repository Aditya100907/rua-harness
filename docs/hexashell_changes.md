# Changes from the original HexaShell

Every change we make to HexaShell (https://github.com/HexaFuture/HexaShell). Base commit for all of
them: `914d482` (Nov 25 2026). Each change lives as a patch file in `setup/patches/`, so the HexaShell
checkout stays a clean clone plus `git apply`.

Checkouts: `~/apollo-lab/HexaShell` (Mac) and `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/HexaShell`
(Bouchet), both at `914d482` with submodules `third_party/robocasa` (`b4684e6`) and
`third_party/robosuite` (`5ce6643`).

## 1. `target` split for the RoboCasa bench (Nov 27 2026)

**Why.** HexaShell's RoboCasa bench only has the pretraining kitchens (`pretrain50`, `pretrain`). Our
policy is trained and evaluated on RoboCasa's target split. To compare agent alone with agent plus
policy on the same scenes, HexaShell must build the scenes that `policy/eval.py` builds for the same
task and seed.

**Patch.** `setup/patches/hexashell_target_split.patch` (`git diff 914d482 -- hexashell`). Apply with
`bash setup/patches/apply_hexashell_target_split.sh [HEXASHELL_DIR]` (idempotent; checks the base).
Use with `python -m hexashell.benches.run --bench robocasa --split target --task T --seed S`
(or `HEXASHELL_ROBOCASA_SPLIT=target`).

**What changed.**

| file | change |
|---|---|
| `hexashell/benches/robocasa/protocol.py` | `SPLITS` gains `"target"`; new `TARGET_SCENES` = (1, 1) to (10, 10) and `TARGET_CLUTTER_MODE = 1`; docstring |
| `hexashell/benches/robocasa/sim.py` | `make(split="target")` builds the env with the target datasets' env kwargs: `layout_and_style_ids` (1, 1)..(10, 10), `obj_instance_split="target"`, `clutter_mode=1`, and the datasets' HYBRID_MOBILE_BASE controller config (`TARGET_CONTROLLER`, copied from `extras/dataset_meta.json`; differs from robosuite's PandaOmron default only in `composite_controller_specific_configs`). `reset` is unchanged: for `target`, as for `pretrain`, the seed draws the kitchen through `env.rng = default_rng(seed)` |
| `hexashell/benches/run.py` | `--split` help text and docstring list `target` |

The existing splits are untouched: `pretrain50` and `pretrain` take the same code paths with the same
arguments as before.

**Seeding.** HexaShell's reset does `env.seed = seed; env.rng = default_rng(seed); np.random.seed(seed)`
and runs `env.reset()` with unseeded `np.random.default_rng()` calls redirected to a seeded side stream.
Ours does only `env.rng = default_rng(seed); env.reset()`. The extra steps do not touch `env.rng`, so the
scene draws are the same; they only make fallbacks reproducible.

**Differences that remain (not part of the patch).**
- Cameras: HexaShell renders left, right and wrist at 512 on demand (`use_camera_obs=False`); our eval
  uses 256 camera observations. Neither touches scene sampling.
- RoboCasa: HexaShell runs `b4684e6` with its patches (`scripts/patches/robocasa.patch`). `b4684e6` is
  an ancestor of our `425684d`. Between them the `robocasa/` package differs only in
  `composite/setting_the_table/arrange_drinkware.py` (more excluded layouts), `models/fixtures/sink.py`
  (`check_obj_under_water` removed), and new subtask-labeling code. None of it is used by the 18 atomic
  tasks. HexaShell's asset-path and temp-file patches do not touch sampling. Its determinism patch does
  (see the check below).

**Check.** `policy/hexashell_target_check.py` dumps, for each (task, seed), layout and style ids, the
ep_meta objects (name, category, model), fixture refs, instruction, every object's initial position,
the robot base position and the task's target part (`simlib.truth`), from both sides, and compares
them (positions to 0.1 mm). Results: `docs/hexashell_target_check.md`.
