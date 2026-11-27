# rua-harness: instructions for AI agents working on this code

## The project in three lines
An LLM agent hands physical jobs to a small fast robot policy (DEM), in the RoboCasa kitchen simulator. The research question: what should the agent tell the policy (a sentence, structured text, 2D points, 3D points, a pose path), measured with the policy retrained for each format. The paper needs two tables: which command format works best, and agent alone against agent plus policy on success, time and tokens.

## Start here
1. `docs/STATUS.md`: what is done, running, next and blocked. Work from the top of "Next". Update it as you go.
2. `README.md`: layout and versions.
3. `docs/agent_port_spec.md`: the agent side (William).
4. On Adi's Mac only: `~/Desktop/APOLLO-Lab/CLAUDE.md` and `PROJECT_CONTEXT.md` hold the full project record (meetings, decisions, research notes). Read them when you need the why. That folder is not in this repo. Never copy anything from its `private/` folder here.

## People
- Adi Kothari: project lead. Owns `policy/` (data, training, evaluation).
- William Gu: owns `agent/` (HexaShell's tools on RoboCasa, LLM loop, episode runner).
- Xiatao Sun: PhD advisor. Hands-off. Gets a weekly update.

## Environment
- Everything lives under `~/apollo-lab/`: this repo, `decoupled-embodiment-model/`, `robosuite/`, `robocasa/`, `runs/`.
- Conda env `dem`, Python 3.11. Install with `bash setup/setup_dem.sh --assets`.
- Pinned: RoboCasa v1.0.1 at `425684d`, robosuite `5ce6643`, MuJoCo 3.3.1, numpy 2.2.5. RoboCasa is installed with `--no-deps`, because its own pins would downgrade what DEM needs. Do not change these pins.
- Do not run Python from `~/apollo-lab` itself. The `robosuite` folder there hides the installed package. Use this repo or `~/apollo-lab/runs`.
- On the Mac GPU set `PYTORCH_ENABLE_MPS_FALLBACK=1`. On a Linux cluster with no screen set `MUJOCO_GL=egl`.
- Real training runs on Yale's Bouchet cluster through its job queue. Submit jobs and check logs. Do not hold a terminal open waiting.

## Bouchet (Yale's cluster): how to work on it from Adi's Mac
- Connect with `ssh bouchet`. It reuses one approved connection for 8 hours, so commands like `ssh bouchet "squeue -u ak3288"` run with no prompt. If a command asks for Duo or says the connection is closed, stop and ask Adi to run `ssh bouchet` once and approve it. The Yale VPN must be on when off campus.
- Paths: project `/nfs/roberts/project/pi_dr877/ak3288` (code, environment, checkpoints to keep), scratch `/nfs/roberts/scratch/pi_dr877/ak3288` (datasets and caches, purged after 30 days). Home has a 500,000-file limit, so install nothing there.
- Code on the cluster: `$PROJ/apollo-lab/{rua-harness,decoupled-embodiment-model,robosuite,robocasa,runs,logs}`. Environment: `module load miniconda; conda activate /nfs/roberts/project/pi_dr877/ak3288/envs/dem`.
- Demos: `$SCR/datasets/v1.0/target/atomic/<Task>/<date>/lerobot/`.
- Send code with `rsync -a --exclude .git --exclude __pycache__ ./ bouchet:/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/`. Edit on the Mac, commit, rsync, then run on the cluster. Do not edit code on the cluster.
- Login nodes are for editing, installs and submitting jobs. Anything that uses a GPU or runs longer than a few minutes goes through `sbatch`. `gpu_devel` is for tests (6 hours, 1 running job). `gpu`, `gpu_h100`, `gpu_h200` are for training (2 days per job, so save checkpoints and support resume). Request a GPU with `--gpus=1` or a type such as `--gpus=rtx_5000_ada:1`.
- Jobs with no screen need `MUJOCO_GL=egl`.
- Before starting any background download or job, check it is not already running (`pgrep -u ak3288 -af <name>` or `squeue -u ak3288`). Never start a second copy.
- `setup/setup_bouchet.sh` has `env`, `assets`, `data`, `simtest`.
- YCRC support is closed until Mon Nov 30.

## Key facts
- DEM: `dem/policy.py` in the DEM repo. 2 cameras at 256, 32 language tokens (frozen NeoBERT), 16-number robot state, 12-number action, predicts 16 actions and executes 8.
- Data: RoboCasa "target, human" split, 500 demos per task, LeRobot format (parquet state and actions, mp4 per camera, `tasks.jsonl`, plus `extras/` with saved simulator states and per-episode metadata). Download with `python -m robocasa.scripts.download_datasets --split target --source human`.
- The 18 tasks are `TARGET_TASKS["atomic_seen"]` in `robocasa/utils/dataset_registry.py`. Evaluate with `split="target"`.
- Smoke-test tasks: `PickPlaceCounterToCabinet`, `OpenDrawer`, `TurnOnMicrowave`.

## Rules
- **Tests before trust.** No evaluation number counts until the replay test (a demo's recorded actions succeed in the simulator) and the zero test (an untrained checkpoint scores about 0%) both pass.
- **Every run writes a CSV to `results/`** with the columns in `results/README.md`, including the code commit.
- **Work on a branch, open a pull request, keep `main` passing.** `PYTHONPATH=. python -m pytest -q tests` must pass before a merge.
- **No secrets in the repo.** Keys go in `.env`, which git ignores.
- **Do not spend money on LLM APIs** beyond a single smoke-test episode without Adi's say-so.
- **Do not delete datasets, checkpoints or results.**
- **Do not message anyone** (Slack, email, GitHub comments to people outside the repo). Draft for Adi.
- **Simulator ground truth is for grading only.** The agent never acts on it.
- **Record every change from the original HexaShell** in `docs/hexashell_changes.md`.
- **If a number looks too good, or is exactly zero, stop and check** before building on it.
- **Blocked on something only a person can do** (an access request, a budget, a decision about the claim): write it under "Blocked" in `docs/STATUS.md` and move to the next item.

## Autonomous sessions (overnight runs, loops, routines)
- **Keep a running log** in `docs/autonomous_log.md`: start a dated section for the session, and update it every few minutes with what you are trying, bugs you hit, how you solved them, and when and why you stop or stall (including idle gaps you notice afterwards). Commit and push it with your other work.
- **Commit and push after every piece of progress**, even small ones, on the working branch.
- **Never end a turn with only a long helper or job pending.** Keep a short background timer so the session wakes up and checks.
- **On the Mac, run long sessions under `caffeinate`** so the machine does not sleep, and ask Adi to approve Duo (`ssh bouchet true` in his own terminal) before the 8-hour SSH connection expires.

## Writing style
Short, plain, no em dashes, active voice. Define a term the first time. State what was run and what was not.
