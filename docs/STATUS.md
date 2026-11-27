# Status

Keep this current. A new session reads it first and continues from "Next".
Last updated: Fri Nov 27, 2026, 16:30 ET.

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

## Key finding (Nov 27)
Stock DEM trained with the MeanFlow recipe does not learn usable actions here: run1 (3 tasks, 20k) 0% on drawer and pick-place; run2 (18 tasks) 0% at 70k; on training frames its actions are worse than the mean action and dominated by sampling noise. Plain regression through the same pipeline memorizes one demo (normalized MSE 0.0019) and learns on 3 tasks, so the data, vision and head pipeline is sound; MeanFlow's optimization is the suspect. Shipped and recomputed quantiles are identical (`docs/quantile_comparison.txt`). Details: `docs/research_journal.md`, `docs/autonomous_log.md`.

## Never run
- A policy that scores above 20% on any smoke task (the gate).
- The container build and the GitHub workflow.

## Running (Nov 27, 16:55; live view: `bash scripts/status.sh`)
- **Gate passed (16:52):** abl_mse (regression, 5k steps) OpenDrawer 22/50 (44%), PickPlace 0/50, Microwave 0/50. abl_fm (flow matching) 0/50 on all three. Objective chosen: mse.
- **Format study on mse** (3 smoke tasks, 20k steps, seed 0, held-out-50 split, left + wrist, shipped quantiles): fs_mse_sentence 34894170, fs_mse_kv_text 34894764, fs_mse_points_2d 34894765, fs_mse_points_3d_rgb 34894766, fs_mse_points_3d 34894767, fs_mse_pose_path 34894768; 18-task sentence fs_mse_sentence_all18 34894769 (300k). Watcher (34894756) scores every 5k steps on CPU nodes and holds the cap at 8 GPUs.
- SAM 3 service 34895235 (gpu_devel, until 23:04): http://a1125u42n01:18300, address in `runs/sam3_service.txt`, smoke test passed (`docs/sam3_service.md`).
- hx_check 34894785 (day): HexaShell target split vs our eval scenes, seeds 1000-1049.
- fs_fm_sentence 34894171: flow matching at 20k with the same settings (does fm catch up with more steps).
- MeanFlow batch-size test mf_ddp4_bs256 finished 5k steps; conditioning check and simulator eval (34894664) running.
- HexaShell target split done (matches our eval on 15/15 scenes). SAM 3 service works (smoke test passed) but is stopped: YCRC cancels idle GPU jobs, so start it on demand with the HexaShell runs (`docs/sam3_service.md`).

## Next: policy track (new priorities from Adi, Nov 27 afternoon)
1. **Gate: PASSED** (abl_mse 22/50 on OpenDrawer).
2. **Format study: LAUNCHED on mse** (16:54). Queue written by `bash policy/slurm/format_study_queue.sh <obj>` writes the queue (sentence, kv_text, points_2d, points_3d_rgb, points_3d, pose_path; 3 smoke tasks, 20k steps, same seed and data, left + wrist, shipped quantiles; then 18-task sentence). Then restart the watcher with only these runs queued. One GPU per run.
3. **Batch-size test** (mf_ddp4_bs256): report whether raw_mse and the conditioning check beat the mean action.
4. **Xiatao's weights:** when they arrive, `python -m policy.research_ckpt --head ... --vision ... --norm ... [--norm-stats ...] --lang neobert|mmbert --out DIR`, then `bash policy/slurm/score_research.sh DIR/ckpt_0000000.pt`; fine-tune any format with `policy.train --init-from DIR/ckpt_0000000.pt`. Tested on random-init stand-ins.
5. Push after every commit.

## Next: agent track (William)
See `docs/agent_port_spec.md`.

## Blocked
- Nov 27, 07:10: SSH to Bouchet needs a new Duo approval (the 8-hour shared connection expired). Asked Adi to run `ssh bouchet` once. Until then: no syncs, the fixed watcher is not restarted (it was stopped at 06:30 after its resubmit bug); jobs already queued keep running.
- Part W (RPent baseline): needs a planner LLM with credentials, which costs money or needs a terms check (Adi: `docs/decisions_needed.md` item 5). RPent is installed and its simulator checks pass (`docs/rpent_notes.md`).
- HexaShell already has a RoboCasa port; whether William still ports it is Adi's call (`docs/decisions_needed.md` item 1).
- (Resolved Nov 26) No lab API key or budget. No private DEM data loader or evaluation script. See Decisions.
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
