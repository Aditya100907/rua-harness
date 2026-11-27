# Status

Keep this current. A new session reads it first and continues from "Next".
Last updated: Thu Nov 26, 2026, about 7:45 pm ET.

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
- `setup/setup_bouchet.sh` written (env, assets, data, simtest). Not yet run.

## Never run
- `scripts/render_geometry.py` (depth, mask and point map for one scene, with a self-check).
- Any training on real data. Any evaluation of a trained policy.

## Running
- Nothing.

## Next: policy track (Adi)
Each item is done only when its test passes.
1. **Bouchet first job.** Log in, build the `dem` environment there, submit a GPU job that imports torch and prints the GPU name. Record the queue wait here.
2. **Download demos.** All 18 tasks on Bouchet, the 3 smoke-test tasks on the Mac. Test: each task folder has 500 episodes and an `extras/` folder with `states.npz`.
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
- (Resolved Nov 26) No lab API key or budget. No private DEM data loader or evaluation script. See Decisions.
- Bouchet account or partition for our jobs, and GPU access for SAM 3: not asked yet.

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
