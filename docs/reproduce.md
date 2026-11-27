# Reproducing the policy results

How to rebuild the environment and rerun the Nov 26 to 27 policy runs from nothing, on the Mac and on Bouchet. Written Nov 27, 2026.

## 1. Versions

| Piece | Version |
|---|---|
| Python | 3.11.18 (conda-forge) |
| RoboCasa | v1.0.1, commit `425684d332f135f5b30a40b321f52d15a0fe9250`, installed `--no-deps` |
| robosuite | commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| DEM | commit `e5c8f6a81dca59bdfea2d1de067a7ec6f8a3e0ea` (`origin/main`, same on Mac and Bouchet) |
| MuJoCo, numpy, scipy, numba | 3.3.1, 2.2.5, 1.15.3, 0.61.2 |
| torch, torchvision | 2.15.0, 0.29.2 (Bouchet: `+cu132`, CUDA 13.2 wheels) |

Lock files (the full `pip freeze` of each env, editable clones replaced by `name @ git+URL@commit`):
- `setup/lock_mac.txt`: Mac, macOS arm64.
- `setup/lock_mac_conda.txt`: the Mac env's conda layer (`conda list --explicit`; only Python and its libraries).
- `setup/lock_bouchet.txt`: Bouchet, RHEL 9.6, x86_64.

Install from a lock with `pip install --no-deps -r <lock>`. `--no-deps` matters: it stops pip from re-resolving, and RoboCasa's own pins would downgrade torch and transformers. To keep the three clones editable, skip their lines and install them by hand as in `setup/setup_dem.sh`.

### Mac against Bouchet
The pinned pieces match exactly (Python, torch, numpy, scipy, numba, MuJoCo, gymnasium, the three commits). Differences, from `pip freeze` on Nov 27:
- Only on Bouchet: the CUDA stack that torch pulls in on Linux (`nvidia-*` cu13 packages, `cuda-toolkit 13.2.2`, `cuda-bindings`, `triton 3.9.0`), `opencv-python-headless 5.1.0.95` (`setup_bouchet.sh` installs it), and `evdev`, `python-xlib` (Linux back ends of `pynput`).
- Only on the Mac: `pyobjc-*` 13.0 (macOS back end of `pynput`).
- Small version drift, because the two envs were built a few hours apart without a lock: `opencv-python` 5.0.0.93 Mac, 5.1.0.95 Bouchet; `transformers` 5.23.0 / 5.24.1; `huggingface_hub` 2.4.0 / 2.5.0; `packaging` 26.3 / 26.4; and patch bumps of `annotated-doc`, `cloudpickle`, `filelock`, `httpcore2`, `httpx2`, `imageio`, `pydantic`, `pydantic_core`, `pyyaml`, `regex`, `typer`. None of these touch the simulator or the model maths as far as we know, but `transformers` loads NeoBERT, so prefer the Bouchet lock for anything that produces a reported number.

## 2. Mac from scratch
```bash
bash setup/setup_dem.sh --assets           # clones, env 'dem', kitchen assets, a rollout check
git -C ~/apollo-lab/robosuite checkout 5ce6643
git -C ~/apollo-lab/robocasa checkout 425684d
# optional, exact versions: pip install --no-deps -r setup/lock_mac.txt (minus the 3 git lines)
hf auth login                              # gated DINOv3 weights
python -m robocasa.scripts.download_datasets --split target --source human \
  --tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave
```
Note: `setup_dem.sh` clones the default branches and does not check out the pinned commits; the two `checkout` lines do that. `setup_bouchet.sh env` does check them out.

Checks, in this order (all from this repo, never from `~/apollo-lab` itself):
```bash
PYTHONPATH=. python -m pytest -q tests                       # unit + sim tests
python -m policy.replay_test --tasks OpenDrawer TurnOnMicrowave --n 10     # replay test
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m policy.eval --wiring --tasks OpenDrawer --episodes 3
bash scripts/clean_clone_test.sh                              # fresh clone works (section 5)
```

## 3. Bouchet from scratch
On a login node, after `rsync`ing this repo (`bash scripts/sync_bouchet.sh` from the Mac):
```bash
bash setup/setup_bouchet.sh env        # clones at the pinned commits, env, installs, checks
hf auth login
bash setup/setup_bouchet.sh assets     # kitchen assets (background; a login-node download was
                                       # killed once on Nov 26, so check the log finished)
bash setup/setup_bouchet.sh data       # 18 tasks of target human demos to scratch
bash setup/setup_bouchet.sh simtest    # gpu_devel job; log must end in SIMTEST OK
```
For exact versions, after `env`: `pip install --no-deps -r <(grep -vE '^(#|dem @|robosuite @|robocasa @)' setup/lock_bouchet.txt)`.

Then the policy pipeline, all through `sbatch` (scripts in `policy/slurm/`, each sources `common.sh` for the env, `MUJOCO_GL=egl` and the scratch paths):

| Step | Command | What it makes |
|---|---|---|
| Frames | `sbatch --array=0-17 policy/slurm/extract_frames.sbatch` | `$SCR/frames`, decoded camera frames for training |
| Replay test | `sbatch policy/slurm/replay.sbatch` | `runs/replay/bouchet_<job>.csv`; must be at least 9/10 per task |
| Zero test checkpoint | `sbatch policy/slurm/untrained.sbatch` | `checkpoints/untrained_smoke` |
| Zero test | `CKPT=$PROJ/checkpoints/untrained_smoke/ckpt_0000000.pt TASKS="PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave" sbatch --array=0-2 policy/slurm/eval_array.sbatch` | about 0% |
| Eval wiring | `TASKS="..." EVAL_ARGS=--wiring sbatch --array=0-2 policy/slurm/eval_array.sbatch` (no `CKPT`) | demo actions through the eval loop |
| Speed test | `sbatch policy/slurm/speedtest.sbatch` | 300 steps, GPU utilization |
| Hindsight labels | `sbatch --array=0-17 policy/slurm/label_array.sbatch` | `$SCR/labels` (format study) |
| Training | `bash policy/slurm/launch_train.sh RUN "ARGS"` | `checkpoints/RUN/ckpt_*.pt`; resubmits itself until `--steps` |
| Everything after | `sbatch policy/slurm/watcher.sbatch` | resubmits dead runs, scores each new checkpoint, launches `runs/queue.txt`, rebuilds tables |

Evaluation writes `results/<date>_policy_only_<run>_<step>_<task>.csv`; `python scripts/make_tables.py` turns `results/*.csv` into `results/tables.md` and `paper/tables/*.tex`.

### The runs launched on Nov 26 to 27
Arguments as saved in `$PROJ/checkpoints/<run>/args.txt` (read Nov 27; all running then, no results CSV yet):

| Run | `policy.train` arguments |
|---|---|
| `run1_smoke` | `--tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave --steps 20000 --batch 64 --save-every 2500 --seed 0` |
| `run2_all18` | `--all-tasks --steps 300000 --batch 64 --save-every 10000 --seed 0` |
| `run3_smoke_seed1` | as run1 with `--seed 1` |
| `run4_smoke_seed2` | as run1 with `--seed 2` |
| `run5_smoke_leftwrist` | as run1 with `--cams left wrist` |

To rerun one: `bash policy/slurm/launch_train.sh run1_smoke "--tasks PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave --steps 20000 --batch 64 --save-every 2500 --seed 0"`, then score its checkpoints with `eval_array.sbatch` (50 episodes, 100 for the final one, seeds `1000 + i`), or let the watcher do both. Every CSV row carries `code_commit`; check out that commit before rerunning.

## 4. Container (Apptainer)
Bouchet has Apptainer 1.5.5 (`/usr/bin/apptainer`, also as `singularity`). No root and no `/etc/subuid` entry for `ak3288`, so the build runs as an unprivileged user-namespace build. The NVIDIA EGL library and `10_nvidia.json` are in Bouchet's `nvliblist.conf`, so `--nv` brings in headless EGL.

- `setup/rua.def`: Ubuntu 24.04 (`nvidia/cuda:13.0.0-base-ubuntu24.04`), Python 3.11.18 from conda-forge, every package in `setup/lock_bouchet.txt` installed `--no-deps`, plus glvnd, EGL, OSMesa and ffmpeg. The three clones, kitchen assets and datasets are **not** in the image: they come from the cluster disk through `-B /nfs/roberts` and `PYTHONPATH` (set in `%environment` from `$ROOT`).
- `setup/build_container.sbatch`: a gpu_devel job that builds `$PROJ/containers/rua.sif` (cache and temp files in `$SCR/apptainer`), prints versions inside the image, runs the same simulator test as `setup_bouchet.sh simtest` with `--nv`, then the repo's `not sim` unit tests.

Run it (from the Mac: commit, `bash scripts/sync_bouchet.sh`, then on Bouchet):
```bash
cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
squeue -u ak3288 -p gpu_devel         # gpu_devel allows one job at a time
sbatch setup/build_container.sbatch    # REBUILD=1 to rebuild over an existing image
# done when the log ends in CONTAINER SIMTEST OK:
tail -5 /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/build_container_<job>.out
```
Use the image in any job: `apptainer exec --nv -B /nfs/roberts --env ROOT=$ROOT $PROJ/containers/rua.sif python -m policy.train ...`, run from `$ROOT/runs` or the repo, with `MUJOCO_GL=egl` (set in the image).

**Not run yet.** Unknowns: whether compute nodes reach Docker Hub, GitHub and PyPI (the login node reaches Docker Hub), whether the unprivileged build's `apt-get` works, and whether the `nvidia/cuda:13.0.0-base-ubuntu24.04` tag resolves (any Ubuntu base works: torch ships its own CUDA). If compute nodes have no network, the build has to happen somewhere else (a Linux machine with Apptainer) and the `.sif` copied over, or ask YCRC (support reopens Mon Nov 30).

## 5. Clean-clone test
`scripts/clean_clone_test.sh` clones this checkout (committed files only, branch `overnight-nov26` by default) into a temp folder and, with the existing `dem` env:
1. runs `PYTHONPATH=. python -m pytest -q -m "not sim" tests`;
2. trains 5 steps, batch 2, on CPU, on 2 OpenDrawer demos (`--episodes-json`, `--video` so no frame store is needed);
3. evaluates that checkpoint on 1 OpenDrawer episode capped at 16 steps (`--horizon 16`) in the real simulator.

It needs the OpenDrawer demos, the kitchen assets and the NeoBERT and DINOv3 weights in the Hugging Face cache. It writes only to the temp folder.

**Result on Adi's Mac, Nov 27, commit `a413f50`: CLEAN CLONE OK in 100 s.** Tests: 30 passed, 21 deselected (7 s). Training: 5 steps in 67 s (0.1 step/s on CPU), checkpoint written. Evaluation: 1 episode, 0/1 success (expected for 5 steps), 22 s including model load, CSV row written with `code_commit a413f50`.

Found while writing it: a checkpoint trained with `--offline` cannot be evaluated. `--offline` builds NeoBERT without a tokenizer, and `policy.eval` fails with `RuntimeError: no tokenizer available`. So the test trains with the pretrained towers. `OFFLINE=1` trains offline and skips the evaluation.

The clone does not include uncommitted work. At the time of the run that left out `policy/variants.py`, `policy/rerender.py`, `policy/slurm/label_array.sbatch`, `tests/test_variants.py` and the edit to `policy/train.py`.

## 6. Continuous integration
`.github/workflows/tests.yml` runs on every push and pull request: Ubuntu, Python 3.11, only `numpy==2.2.5 scipy==1.15.3 pytest`, then `python -m pytest -q -m "not sim" -rs tests --ignore=tests/test_variants.py` with `RUA_DATA` pointed at an empty folder.

Checked locally in a bare venv with just those three packages (Nov 27): 23 passed, 7 skipped, 21 deselected. Skips: the 6 dataset tests in `tests/test_data.py` (no demos) and 1 test in `tests/test_simlib.py` (no RoboCasa, `importorskip`). Without `RUA_DATA` set, on a machine that has the demos but not `av`, `test_data.py` errors instead of skipping, which is why the workflow sets it.

`tests/test_variants.py` (not yet committed) imports `policy.variants`, which imports torch at the top, so collection fails without torch. The workflow ignores it. To run it in CI later, either add `pytest.importorskip("torch")` before its import of `policy.variants`, or install CPU torch in the workflow. The workflow itself has not run on GitHub yet.
