# Status

Keep this current. A new session reads it first and continues from "Next".
Last updated: Fri Nov 27, 2026, 00:20 ET (overnight agent).

## Goal for Sun Nov 29, night
Three conditions on RoboCasa's 18 atomic-seen tasks: policy alone, agent alone, agent plus policy. Success, seconds, LLM calls and tokens per episode.

## Done
- Repo created, first commit `b180b55`. William invited.
- `rua_harness/command.py` and `geometry.py`: command format v0 and geometry helpers. 8 tests pass.
- `docs/agent_port_spec.md`: William's spec.
- DEM, robosuite and RoboCasa installed on Adi's Mac. DEM's tests pass, a 10-step run on fake data works, a random-action RoboCasa episode renders.

- Bouchet login works (Nov 26, through the web portal with the Yale VPN on). User `ak3288`, group `pi_dr877`. Home `/nfs/roberts/home/ak3288` (125 GiB, 500,000 files). Project `/nfs/roberts/project/pi_dr877/ak3288` (group 4 TiB, backed up). Scratch `/nfs/roberts/scratch/pi_dr877/ak3288` (group 10 TiB, purged after 30 days). Plan: code and environment in project, datasets and simulator assets in scratch, checkpoints worth keeping in project.

- SSH from the Mac works: `ssh bouchet` (config in `~/.ssh/config`, one Duo approval lasts 8 hours through a shared connection).
- Test GPU job on `gpu_devel` started with no queue wait (Nov 26, 9:24 pm, holiday evening) and got an RTX 5000 Ada with 32 GB. The cluster has `miniconda/26.9.4` as a module and `git`. No `conda` on the path until `module load miniconda`.
- Bouchet environment built with `setup/setup_bouchet.sh env` (Nov 26, about 9:55 pm): numpy 2.2.5, torch 2.15.0 with CUDA, MuJoCo 3.3.1, DEM tests pass. Hugging Face login done on the cluster.
- Kitchen assets and the 18-task demo download started on Bouchet at about 10:05 pm (one copy each). **Not yet verified:** that both finished, that each task has 500 episodes, and `setup_bouchet.sh simtest`.

- Nov 26 overnight (branch `overnight-nov26`): downloads verified (18 tasks, 500 to 543 episodes, `docs/dataset_notes.md`). Kitchen assets were incomplete on Bouchet (login-node download killed); re-downloaded by job 34840169. `SIMTEST OK` (job 34841443, gpu_devel, EGL works; 35.6 s for env creation plus 20 steps).
- Replay test (`policy/replay_test.py`) on the Mac: OpenDrawer 10/10, TurnOnMicrowave 10/10.
- Data loader `policy/data.py` and frame store `policy/extract_frames.py`; `tests/test_data.py` passes (6 tests). Frames for all 18 tasks extracted to `$SCR/frames` (job 34840595).
- Training script `policy/train.py`: 50 steps on the Mac (MPS) with falling loss (4.79 to 0.79), stop and resume continues the step count.
- Eval wiring test (`policy/eval.py --wiring`) on the Mac: OpenDrawer 3/3, env state matches the recording to 1e-8, rendered image matches the video within 2 grey levels.
- Nov 27 overnight: replay 10/10 on all 3 smoke tasks on Bouchet; wiring 30/30 and zero test 0/28 on Bouchet; speed test 3.27 steps/s on H100 (batch 64). Hindsight labels for all 18 tasks (`docs/labels_notes.md`). Five command formats (`policy/variants.py`), eval with ground-truth commands and error modes (`policy/commands_gt.py`), held-out object split (`policy/heldout.py`), `run_skill`, the commander (`policy/commander/`), simlib, depth re-render, paper scaffolding. Full list: `docs/MORNING_REPORT.md`.

- Nov 27 (Part S): LIBERO set up on Bouchet as a second benchmark, in its own env `envs/libero` (robosuite 1.4.1, MuJoCo 2.3.7, plus DEM). Demos for libero_spatial and libero_object on scratch. Random episode, replay test, eval wiring and a CPU training smoke test pass. No LIBERO training launched. Details, numbers and the launch command: `docs/libero.md`.

## Never run
- Any evaluation of a trained policy (first checkpoints are training now).
- The container build, the GitHub workflow, the depth re-render on Bouchet beyond the 3 smoke tasks (running).

## Running (Nov 27, 00:20; live view: `bash scripts/status.sh`)
- run1_smoke 34842801, run3/run4 seeds 34842803/34842804, run5 left+wrist 34842805: 3 tasks, 20k steps, L40S/A40, end about 05:00.
- run2_all18 34842802: 18 tasks, 300k steps, H100, about 25 h; resubmits itself.
- f4_kv_text 34843699 (format study). Queued: f4_points_3d_rgb, f4_points_2d, f4_pose_path, f5_sentence_objsplit, f5_points_3d_rgb_objsplit.
- watcher (agent partition, 7 days): scores checkpoints, resubmits dead runs, launches the queue (cap 8 GPUs, 2 kept for scoring).
- F2 depth re-render for the 3 smoke tasks: 34844300 (day partition).

## Overnight Nov 26 to 27
An agent works through `docs/OVERNIGHT.md` on branch `overnight-nov26`. It supersedes the short list below where they differ. Its log is `docs/overnight_log.md` and its summary is `docs/MORNING_REPORT.md`.

## Next: policy track (Adi)
Each item is done only when its test passes.
1. **Verify the Bouchet downloads and run the simulator test.** Check no download is still running. Count episodes per task under `$SCR/datasets/v1.0/target/atomic/` (expect 18 tasks with 500 each) and confirm each has saved simulator states. Then run `bash setup/setup_bouchet.sh simtest` and confirm the log ends with `SIMTEST OK`. Record sizes and results here.
2. **Look at one dataset before writing code.** Print the folder tree of one task, `meta/info.json`, `meta/modality.json`, one parquet file's columns and shapes, and the video list. Write what you find to `docs/dataset_notes.md`. Download the 3 smoke-test tasks on the Mac for local debugging.
3. **Data loader** in `policy/data.py`. Reads LeRobot files and yields what `DEM.loss` expects: two camera images, robot state, instruction, a 16-step action chunk. Test: one batch has the right shapes and value ranges, and the two cameras are the left and right agent views in that order.
4. **Replay test** in `policy/replay_test.py`. Reset the simulator to a demo's saved state, play its recorded actions, check `env._check_success()`. Test: succeeds on at least 9 of 10 demos for each smoke-test task.
5. **Training script** in `policy/train.py`, built from the DEM repo's `examples/train_minimal.py`. Real loader, action and state normalization saved with the checkpoint, checkpoints every N steps, loss logging. Test: 50 steps on the Mac with falling loss, then a job on Bouchet.
6. **Launch training** on all 18 tasks by Friday evening.
7. **Evaluation script** in `policy/eval.py`. Load a checkpoint, run episodes in the target split, predict 16 actions and execute 8, stop on success or the task's horizon, write a results CSV. Test: the zero test, then the replay test's actions through the same environment wrapper.
8. **Score** the latest checkpoint: 3 smoke-test tasks first, then all 18 at 50 episodes each as parallel jobs.
9. **`run_skill` tool** in `agent/tools/run_skill/`, in HexaShell's tool format. Runs the trained DEM on the live environment until success, timeout or the step limit.
10. **Combined run** with William: all 18 tasks, same seeds as his agent-alone run.

## Next: agent track (William)
See `docs/agent_port_spec.md`.

## Blocked
- Nov 27, 07:10: SSH to Bouchet needs a new Duo approval (the 8-hour shared connection expired). Asked Adi to run `ssh bouchet` once. Until then: no syncs, the fixed watcher is not restarted (it was stopped at 06:30 after its resubmit bug); jobs already queued keep running.
- Part W (RPent baseline): needs a planner LLM with credentials, which costs money or needs a terms check (Adi: `docs/decisions_needed.md` item 5). RPent is installed and its simulator checks pass (`docs/rpent_notes.md`).
- HexaShell already has a RoboCasa port; whether William still ports it is Adi's call (`docs/decisions_needed.md` item 1).
- (Resolved Nov 26) No lab API key or budget. No private DEM data loader or evaluation script. See Decisions.
- GPU access for SAM 3: not asked yet.
- Bouchet limits found tonight: gpu_devel allows 1 job submitted at a time and 12 CPUs; gpu_h200 allows at most 6 CPUs per GPU.

## Decisions
- Nov 26: all 18 tasks, not 3. The 3 are the smoke test.
- Nov 26: the agent is HexaShell's real tools ported through its `EpisodeAPI`, with its own perception.
- Nov 26: stock DEM for the baseline. New inputs come with the format study.
- Nov 26: the real robot waits until the format study is done.

- Nov 26: no lab LLM budget. The agent runs through a coding-agent CLI on a personal subscription. Log tokens per episode anyway and report cost at API list prices. Size the episode count to the subscription's limits after the smoke test.
- Nov 26: no private DEM code exists. The data loader and evaluation script are built here from scratch.

## After the weekend
- Dataset for the format study: replay demos with depth, build point maps, label each demo's command in every format.
- Read X-Harness and GR00T 2 in full before fixing the list of formats.
