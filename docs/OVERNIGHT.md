# Overnight run: Thu Nov 26 into Fri Nov 27

Work this list top to bottom. Read `CLAUDE.md` and `docs/STATUS.md` first.

## How to work tonight
- **Each item has a test. It is done only when the test passes.** If a test fails three times, write what you tried under "Blocked" in `docs/STATUS.md` and go to the next item that does not depend on it.
- **Commit after every item** with a clear message. Work on the branch `overnight-nov26`. Push it. Do not merge to `main`.
- **Keep `docs/STATUS.md` current** after every item.
- **Never wait idle.** While a download, job or training run is going, work on the next item that can proceed.
- **Keep a log** in `docs/overnight_log.md`: time, what you ran, what happened, in one or two lines per step.
- **Keep** `docs/MORNING_REPORT.md` (format at the bottom) current. Write a first version early and update it after every item, so it exists whenever the session stops.
- **There is no finish line.** When Parts A to E are done, go on to F through K, then M through Z. When those are done or blocked, follow the loop in Part L. Part M comes early: do it right after C3, because it keeps work going if the SSH connection drops. Do not stop, summarize and wait. Do not ask whether to continue.
- **Run things in parallel.** Use subagents or background jobs for independent items (for example E6 reading, E1 geometry, I paper scaffolding) while the main thread drives the critical path A, B, C, D.
- **The gate.** Results from Parts F to H count only after B2 (replay test), D1 (wiring and zero tests) and D3 (stock DEM scores above zero on an easy task) have passed. You may write code and launch runs for F to H before the gate passes, but mark every number from them as provisional until it does.

## Limits
- No paid LLM API calls. Headless runs of a coding agent on Adi's subscription are allowed for Part O only, at most 60 episodes tonight. No changes under `agent/` (that is William's).
- At most 8 GPUs in use on Bouchet at once, and at most 12 jobs running (the cluster's own per-user cap). CPU-only work (re-rendering, labeling) can use job arrays on the `day` partition freely. Use `gpu_devel` for tests. Check `squeue -u ak3288` before every submit.
- Never start a second copy of a download or job. Check first.
- Do not delete datasets, checkpoints or results. Do not change the version pins.
- Do not message anyone.
- If `ssh bouchet` asks for Duo or fails, do not retry in a loop. Note it under "Blocked", continue with Mac-only items, and try once more every 30 minutes.
- If a number looks too good or is exactly zero where it should not be, stop and investigate before building on it.

---

## Part A: finish the cluster setup
**A1. Verify the downloads.**
- `ssh bouchet 'pgrep -u ak3288 -af robocasa.scripts.download | wc -l'` must print 0. If not, wait and do A4 meanwhile.
- Count episodes per task under `/nfs/roberts/scratch/pi_dr877/ak3288/datasets/v1.0/target/atomic/`. Expect 18 tasks, 500 episodes each. Record sizes with `du -sh`.
- If a task is missing or short, rerun only that task with `python -m robocasa.scripts.download_datasets --split target --source human --tasks <Task>` on the cluster, once.
- Test: a table of task, episode count, size in `docs/dataset_notes.md`.

**A2. Simulator test on a GPU node.**
- `bash setup/setup_bouchet.sh simtest` on the cluster. Wait for the job.
- Test: the log ends with `SIMTEST OK`. Record the seconds for 20 steps.
- If rendering fails with EGL, try `MUJOCO_GL=osmesa` and record which works.

**A3. Look at one dataset before writing code.**
- For `OpenDrawer`: folder tree, `meta/info.json`, `meta/modality.json`, `meta/tasks.jsonl`, one parquet file's columns, dtypes and shapes, the video files, and what is in `extras/` (saved simulator states, `ep_meta.json`, scene XML).
- Find: the order and meaning of the state vector and the 12-number action, the camera names, fps, how instructions map to episodes.
- Compare with what DEM expects (`dem/policy.py`: `DEMConfig`, `build_cond`, `loss`, `act`): 2 cameras at 256, 16-number state, 12-number action, 16-step chunks.
- Test: `docs/dataset_notes.md` answers each of those, with the exact keys.

**A4. Get three tasks onto the Mac** for fast debugging: rsync `OpenDrawer`, `TurnOnMicrowave`, `PickPlaceCounterToCabinet` from scratch to `~/apollo-lab/robocasa/datasets/` (or wherever the Mac's `DATASET_BASE_PATH` points).
- Test: the same episode counts as on the cluster.

## Part B: data
**B1. Data loader** in `policy/data.py`.
- Reads the LeRobot files. Yields left and right agent-view images at 256, the 16-number state, the instruction, and a 16-step action chunk starting at that frame. Pads the chunk at episode ends and returns a mask.
- Decodes video efficiently. If per-sample mp4 decoding is slow, write `policy/extract_frames.py` to pre-extract frames to scratch once and read those.
- Computes and saves normalization statistics for state and action (`norm_stats.json`), per the DEM repo's conventions.
- Test: `tests/test_data.py` checks shapes, dtypes, value ranges, camera order, that chunks do not cross episodes, and that normalize then unnormalize returns the input. Report samples per second with 8 workers.

**B2. Replay test** in `policy/replay_test.py`. This is the most important test of the night.
- For a demo: build the environment from its saved metadata, reset to its saved simulator state, play its recorded actions, check `env._check_success()`.
- Use RoboCasa's own playback code (`robocasa/scripts/dataset_scripts/playback_dataset.py`, `playback_utils.py`) as the reference for how to reset from saved states.
- Run on 10 demos for each of the 3 smoke-test tasks on the Mac, then as a `gpu_devel` job on Bouchet.
- Test: at least 9 of 10 succeed per task. If not, find out why before going on: control mode, action scaling, absolute against relative actions, control frequency.
- Write down exactly which environment arguments and action conventions made it pass. The evaluation script must use the same ones.

## Part C: training
**C1. Training script** in `policy/train.py`, built from the DEM repo's `examples/train_minimal.py` and its README recipe (AdamW, 1e-4 head, 5e-5 vision encoder, weight decay 1e-4, cosine schedule, gradient clip 1.0, frozen NeoBERT, fine-tuned DINOv3).
- Config file or flags for tasks, steps, batch size, workers, output folder.
- Language tokens computed once per distinct instruction and cached.
- Checkpoints every N steps with optimizer and step, `--resume`, and `norm_stats.json` saved beside them. Logs loss, learning rate, steps per second to a CSV.
- Mixed precision if it is stable. Multi-GPU only if it is simple.
- Test: 50 steps on the Mac (`--device mps`, `PYTORCH_ENABLE_MPS_FALLBACK=1`) with falling loss, then stop and resume from the checkpoint and confirm the step count continues.

**C2. Speed test on Bouchet.** A `gpu_devel` job: 300 steps on the 3 smoke-test tasks.
- Test: loss falls. Record steps per second, GPU memory, and whether data loading is the bottleneck (GPU utilization).
- From that, compute hours for 20k, 100k and 300k steps. Record in STATUS.

**C3. Launch training.** `policy/slurm/train.sbatch`, with checkpoints to project and resume on restart.
- Run 1: the 3 smoke-test tasks, 20k steps.
- Run 2: all 18 tasks, as many steps as fit in 36 hours by C2's estimate, capped at 300k.
- Pick the partition from what `sinfo` shows free. Prefer `gpu_h200` or `gpu_h100` for run 2 if available, otherwise `gpu`.
- Test: both jobs running, loss CSVs growing, a checkpoint written and reloadable. Record job IDs.

## Part D: evaluation
**D1. Evaluation script** in `policy/eval.py`.
- Loads a checkpoint and `norm_stats.json`. Makes the environment with `split="target"` and the same arguments the replay test used. Feeds two camera images, state and instruction. Predicts 16 actions, executes 8, repeats. Stops on success or the task's horizon from `dataset_registry.py`.
- Writes one row per episode to `results/` (columns in `results/README.md`), with the seed and code commit. Optionally saves a video of the first 2 episodes per task.
- Test 1, wiring: feed a demo's recorded actions through the evaluation loop's environment wrapper in place of the policy. It must succeed like the replay test.
- Test 2, zero test: an untrained checkpoint scores about 0% on the 3 smoke-test tasks, 10 episodes each.

**D2. Parallel evaluation.** `policy/slurm/eval_array.sbatch`: a job array, one task per array index, 50 episodes each.
- Test: run it on run 1's latest checkpoint for the 3 smoke-test tasks. A results CSV exists for each.

**D3. First real numbers.** When run 1 has a checkpoint at 5k, 10k and 20k steps, score each on the 3 smoke-test tasks.
- Test: a table of step against success per task in `docs/MORNING_REPORT.md`. Compare with the DEM paper's category numbers (drawers 79.5%, buttons 81.5%, pick-and-place 33.2%) and say plainly how far off we are and the likely reason.

## Part E: go further (do these while training runs)
**E1. Geometry.** Run `scripts/render_geometry.py` on the Mac. Fix it until its self-check passes (the lifted point is within a few cm of the simulator's true object position). Save the images. Record which image convention was right.

**E2. Hindsight labels.** `policy/label_commands.py`: for each demo, find the frames where the gripper closes and opens from the action or state, and take the gripper position there as the grasp point and the place point. Build the `Command` from `rua_harness/command.py` using the task name and `ep_meta.json` for object and fixture names.
- Test: on 20 pick-and-place demos, the grasp point is within 5 cm of the object's simulator position at that frame. Report the distribution of distances.
- Write the labels for the 3 smoke-test tasks to a file. Do not relabel all 18 yet.

**E3. Re-rendering with depth.** Prototype on 5 demos of one task: replay saved states with `camera_depths=True`, convert to metric depth, save depth and camera intrinsics and extrinsics per frame, build point maps with `rua_harness/geometry.py`.
- Test: a point map's value at the grasp pixel matches the hindsight grasp point within a few cm. Record seconds per demo and bytes per demo, then estimate the cost for all 9,000 demos.

**E4. `run_skill` tool.** `policy/run_skill.py`: a function `run_skill(env, instruction, max_seconds, checkpoint)` that runs the trained policy on a live environment until success, timeout or the step limit, and returns a status dict and the final images. Sunday's agent run calls this.
- Test: with run 1's checkpoint, it runs on a live `OpenDrawer` environment and returns without error.

**E5. Failure sorting.** From D3's evaluation, save the failed pick-and-place episodes' videos and final states. Sort each into: never reached the object, reached but missed the grasp, grasped but dropped, placed in the wrong spot, timed out.
- Test: a small table of counts in the morning report. This answers a question open since Nov 15.

**E6. Reading.** Fetch the full text of X-Harness (arXiv 2611.34778) and the GR00T 2 white paper. For each, write half a page in `docs/related_notes.md`: what they tested about the command between agent and policy, and whether either retrains a policy per command format. Mark what you read in full against what you skimmed.

**E7. Clean up.** A `Makefile` or `scripts/` entries for the common commands. A README section on how to reproduce tonight's results from scratch. Type hints and docstrings on the public functions in `policy/`.

## Part F: the format study, built for real
This is the paper's main experiment: one small policy per command format, same data and budget.

**F1. Label all 18 tasks.** Run E2's labeling over every demo as a job array. Output one file per task with, per episode: the `Command`, the grasp point and place point in the world frame, the frame indices of gripper close and open, and the object and fixture names.
- Test: per-task counts equal the episode counts. Grasp-point distance to the object's true position: report the median and the 95th percentile per task. Flag tasks where hindsight labeling does not fit (navigation, button presses) and say what you used instead.

**F2. Re-render all 18 tasks with depth.** E3's pipeline as a job array. Save depth compactly (16-bit, compressed) with intrinsics and extrinsics per episode. Keep the output on scratch and record its size.
- Test: random spot checks, 20 frames per task, that the point map at the grasp pixel agrees with the F1 grasp point.

**F3. The five input variants** in `policy/model.py`, as thin wrappers around DEM so one flag switches format:
  1. `sentence`: the original instruction, color images (stock DEM).
  2. `kv_text`: `to_policy_text(command)`, color images.
  3. `points_2d`: text plus grasp and place points drawn on the images.
  4. `points_3d`: text plus the two 3D points as two extra context tokens through a small MLP, with a point-map input in place of color. Also a `points_3d_rgb` variant that keeps color, so the effect of the points and of the image input can be separated.
  5. `pose_path`: text plus a short path of gripper poses sampled from the demo's own future trajectory (GAE style), with point-map input.
- Add noise to points and poses during training (a few cm, a few degrees), as GAE found necessary.
- Test: each variant runs 50 training steps on the Mac with falling loss, and a unit test confirms that moving the point changes the model's output.

**F4. Train all variants on the 3 smoke-test tasks.** Same steps, batch size and seed for each. Launch as soon as F1 to F3 exist for those 3 tasks, within the GPU cap.
- Test: all runs finish or are running with checkpoints. Job IDs and configs in STATUS.

**F5. Held-out split.** Design and write `docs/heldout_split.md`: which object instances or categories are held out of training for the pick-and-place tasks, using RoboCasa's object splits or `ep_meta.json`. Build the evaluation config for it.
- Test: no held-out object appears in any training episode. Counts reported.

**F6. Score the variants with perfect commands.** Extend `policy/eval.py` so the command comes from the simulator's ground truth at reset. Score each F4 checkpoint on trained objects and on held-out objects.
- Test: a table of format against success, trained and held-out, 50 episodes per cell, in the morning report. Include the "does it follow the point" check: give a deliberately wrong point (the distractor object) and report how often the policy goes to it.

**F7. Scripted baseline.** No learning: move to the grasp point with the controller, close, move to the place point, open.
- Test: its success on the 3 smoke-test tasks, 50 episodes each. If it matches the learned policy, say so plainly. That would mean the learned part is not earning its place yet.

## Part G: more baselines and repeats
**G1. Diffusion Policy** on the same data and budget for the 3 smoke-test tasks (Xiatao asked for an end-to-end baseline). Use a maintained implementation, pinned, in its own environment if its dependencies clash.
**G2. Seeds.** Two more seeds of run 1 (stock DEM, 3 tasks), so there is a noise estimate. The DEM paper trained each setup once.
**G3. Longer and bigger.** If C2 showed spare throughput, a second 18-task run with a different seed.
- Test for each: runs launched with configs recorded, scored when checkpoints exist.

## Part H: speed and cost numbers the paper needs
**H1. Policy timing.** Milliseconds per `act` call and per executed second of robot time, on the RTX 5000 Ada and on one faster GPU. Language encoding counted once per command.
**H2. Episode timing.** Wall-clock seconds per episode for the policy alone, split into simulator time and model time.
**H3. Re-render and label cost** per demo, from F1 and F2.
- Test: a timing table in the morning report, with the hardware named.

## Part I: paper scaffolding
**I1.** `paper/` with the ICML 2027 LaTeX template (fetch the official style files), section stubs, and a `Makefile`.
**I2.** `scripts/make_tables.py`: builds every results table from the CSVs in `results/`, so no number is typed by hand.
**I3.** `scripts/make_plots.py`: loss curves, success against training steps, success by format.
**I4.** A draft figure 1 (request, agent, command, policy, result) as a vector graphic.
**I5.** A related-work draft from `docs/related_notes.md`. Every claim about another paper carries its arXiv ID and a note saying whether it was read in full or from the abstract. Mark anything unverified.
- Test: `make` in `paper/` produces a PDF with real tables from tonight's CSVs.

## Part J: code quality
- Continuous checks: `pytest` for everything in `policy/` and `rua_harness/`, run before each commit.
- One config system for training and evaluation, with every run's full config saved beside its checkpoints.
- A `scripts/status.sh` that prints, in one screen: running jobs, latest checkpoint and loss per run, latest success numbers.
- Remove dead code and stale notes you created tonight.

## Part K: things to investigate when a number surprises you
- Stock DEM far below the paper's category numbers at the same step count: check normalization, camera order and resolution, action chunk execution (16 predicted, 8 executed), the control frequency, language token caching, and whether evaluation kitchens match the paper's split.
- A variant that ignores its points: check the token actually reaches the head, the noise scale, and whether the instruction alone already determines the answer for that task.
- Slow training: profile the data loader first.
Write each investigation up in `docs/overnight_log.md` with what you checked and what you found.

## Part M: make the cluster drive itself (do this right after C3)
The one thing that can stop tonight is the SSH connection from the Mac. So the cluster must keep working without it.

**M1. Chain jobs with Slurm dependencies.** Every training job is submitted together with the evaluation array that scores its checkpoints (`--dependency=afterany:<jobid>`), and long runs resubmit themselves from the last checkpoint before the 2-day limit.
**M2. A watcher job** on the `agent` partition (7 days, 1 CPU, 8 GB, 2 running jobs allowed): a plain bash loop, no LLM. Every 10 minutes it finds checkpoints with no results CSV and submits the evaluation for them, resubmits training jobs that died, rebuilds the tables with `scripts/make_tables.py`, and appends one line to `logs/watcher.log`. It respects the GPU cap.
**M3. A queue file** `runs/queue.txt` on the cluster: one experiment config per line. The watcher submits the next line whenever GPUs are free under the cap. Fill it with everything from Parts F, G and the Part L backlog, in priority order, so experiments keep launching all night with no SSH.
**M4. Find out whether an agent can run on the cluster itself.** Read the YCRC docs on the `agent` partition. If a coding agent can run there with internet access, write the exact setup steps in `docs/agent_on_cluster.md` for Adi to do once (it needs his login). Do not log in for him.
- Test: kill your own SSH session for 30 minutes on purpose (work on Mac-only items meanwhile). When you reconnect, the watcher log shows it kept submitting and scoring.

## Part N: what accuracy does the agent need? (no LLM required)
A stand-in for the agent: perfect commands with controlled errors added at test time.
- **N1. Point error.** Shift the grasp and place points by 0, 1, 2, 5, 10 cm in random directions and score each 3D-point variant. Output: success against pointing error. This says how accurate the real agent has to be.
- **N2. Wrong object.** Point at the distractor. Does the policy follow the point or the name?
- **N3. Missing fields.** Drop the point, or the name, or the source. Which fields carry the policy?
- **N4. Name changes.** Synonyms and unseen object names in the text.
- Test: one plot and one table per item, built by `scripts/make_plots.py`.

## Part O: the simplest agent-plus-policy system
A commander with exactly two tools, `look` and `run_skill`. The LLM sees the scene, writes commands, and the policy does all motion. This is a different condition from William's port (which has HexaShell's motion tools too), and it sits on the policy side.
- Build `policy/commander/`: a small server holding one episode, the two tools, and a runner that starts one headless coding-agent run per episode and logs success, seconds, turns and tokens to `results/`.
- For 3D-point variants the commander needs a pixel-to-point step: it names a pixel, the harness lifts it with depth (`rua_harness/geometry.py`).
- Run at most 60 episodes tonight: 3 smoke-test tasks, 10 episodes, stock DEM and the best F4 variant.
- Test: a results CSV with real token and time numbers. Compare the commander's points against ground truth and report the error in cm. That number plugs into N1's curve.

## Part P: a neutral simulator library both halves can use
In `simlib/`, not `agent/`: things William will also need, written once and tested.
- Camera images, metric depth, intrinsics and extrinsics for every camera. Pixel-to-world and world-to-pixel.
- Ground-truth lookups for grading only: object positions, and for each of the 18 tasks the fixture part that matters (drawer handle, knob, button, lever), found from the scene's bodies and sites.
- A parallel episode runner: one process and one log per episode, seeds, timeouts, the results CSV schema.
- Success checks and the per-task horizons.
- Test: unit tests, plus a table in `docs/simlib.md` of every task with its target part and how it is located. Tell William in `docs/for_william.md` what exists and how to call it. Do not message him.

## Part Q: longer tasks
- Chain atomic commands from ground truth to run RoboCasa's composite-seen tasks with the policy (use the per-frame subtask labels that ship with the target composite datasets to get the sequence).
- Report success per composite task and where chains break (which step, which handoff state).
- Set up, but do not launch at scale, evaluation on RoboCasa365's Target50 protocol, so later numbers are comparable with the harness papers. Write the differences between that protocol and DEM's 18-task protocol in `docs/protocols.md`.

## Part R: toward a real robot (investigation and setup only)
- Does robosuite at our pinned commit include an xArm 7, and can a RoboCasa task run with it? Try one. Record what breaks.
- A depth-noise model closer to a real stereo camera than `add_depth_noise`, with references, and the evaluation-time noise sweep from the backlog.
- What a real deployment needs: camera calibration, a depth source, the control interface. Write `docs/sim_to_real.md` with open questions for Xiatao as lab facts (which arm, which camera).

## Part S: a second benchmark
- Set up LIBERO in its own environment on the cluster: install, download its demos, run a random-action episode, and write a data loader and evaluation wrapper with the same interfaces as RoboCasa's.
- Test: the replay test passes on LIBERO demos. Launch one stock-DEM training run there only if GPUs are free under the cap.

## Part T: writing and handoff
- `paper/`: a full first draft of the method section and the experimental-setup section from what was actually built tonight. Results sections as stubs wired to the auto-built tables. A 150-word abstract draft for the Jan 14 deadline.
- `docs/weekly_update_draft.md`: a short update for Xiatao, in Adi's voice (warm, brief, signed Adi), stating only what is true as of the morning. Draft only.
- `docs/experiment_registry.md`: every run launched tonight: ID, config, purpose, status, where its results are.
- `docs/decisions_needed.md`: each decision only Adi can make, with the options and your recommendation.

## Part U: the full experiment grid (added Nov 26, 11:45 pm)
Put all of this in the cluster queue (`runs/queue.txt`, Part M3) in priority order, so it launches as GPUs free up over the coming days.
1. Stock DEM on all 18 tasks to the paper's full 300k steps, to reproduce 55.6%. Resubmit across the 2-day limit.
2. Every format from F3 on all 18 tasks, one seed.
3. The best two formats and stock DEM with 3 seeds each.
4. The Part L backlog ablations on the 3 smoke-test tasks.
- For each queued run, the scoring job is chained to it. Scoring uses 100 episodes per task for anything that may go in the paper.
- `scripts/make_tables.py` reports means with 95% confidence intervals over episodes and over seeds, and marks any comparison whose intervals overlap.

## Part V: the agent writes each format (the study's second half)
Extends Part O. Subscription runs only. Raise Part O's cap to 300 episodes for this part.
- For each format (structured text, 2D points, 3D points, pose path): the commander sees the scene and writes the command in that format. Measure how far its points or poses are from ground truth, and the policy's success with the agent's command against the perfect command.
- Start with the 3 smoke-test tasks, 10 episodes per format. Then extend to all 18 if time allows.
- Test: a table with one row per format: agent error, success with perfect commands, success with agent commands, seconds and tokens per episode. This is the paper's central table.

## Part W: a VLA-backed harness baseline
- Install RPent (Harness VLA's public code, https://github.com/RLinf/RPent) in its own environment on the cluster with the `[robocasa]` extra. Read its RoboCasa guide. Find what it needs: the VLA checkpoint, SAM 3, the planner.
- Get one RoboCasa365 episode running with a subscription-based planner if the terms of its tools allow, and record time and tokens. If a required checkpoint is gated, write the request steps in `docs/decisions_needed.md` and stop there.
- Also record which RoboCasa commit RPent expects and whether it matches ours.
- Test: either one logged episode, or a precise list of what blocks it.

## Part X: reproducibility
- A lock file with exact package versions for the Mac and for the cluster.
- A container recipe (Apptainer) for the cluster environment, built and tested with the simulator test.
- A clean-clone test: a fresh clone in a temporary folder, the setup script, the tests, and one tiny training and evaluation run, with no manual steps.
- GitHub Actions that run the unit tests on every push.

## Part Y: test sets the paper will need
- **Look-alike distractors:** pick-and-place scenes with two objects of the same category, where only the point says which one. Build the evaluation config from RoboCasa's object sampling.
- **Held-out kitchens** in addition to held-out objects.
- **Unseen names:** the same objects called by names never used in training.
- Test: each set runs with the scripted baseline and reports a number.

## Part Z: integration and records
- `run_skill` packaged in HexaShell's tool format (`tools/run_skill/tool.py` and `interface.md`), with a test that calls it the way William's harness will. Put it in `policy/hexashell_tool/`, not `agent/`.
- When William pushes a branch, read it and write review notes in `docs/review_william_<date>.md`: what works, what is risky, questions. Do not edit his code.
- A supplementary-video plan from saved evaluation videos: which clips show a success, a typical failure, and the policy following a moved point.
- Verified BibTeX for every paper cited in `paper/`, each checked against its arXiv page.
- At about 7 am, and again whenever a part completes: append dated entries to the log in `~/Desktop/APOLLO-Lab/PROJECT_CONTEXT.md` section 16 and update its sections 1 and 13.0, following that folder's `CLAUDE.md` (reread before editing, change only the lines you mean to).

## What is deliberately not on this list
- William's HexaShell port and the agent-alone baseline (his).
- Anything on a physical robot.
- Choosing the paper's claim. Record evidence for and against each option in `docs/decisions_needed.md`.
- Messaging Xiatao or William.

## Part L: the research loop (runs forever, day and night)
Start this loop as soon as stock DEM has a trusted number (the gate has passed). It runs alongside the remaining parts, not after them. It does not end.

### Each cycle
1. **Collect.** Check every job. Restart dead ones from their last checkpoint. Score every unscored checkpoint. Rebuild tables and plots.
2. **Analyze.** For each new result, write an entry in `docs/research_journal.md`: what was tested, the number with its episode count and interval, what it means, and what it rules out. Watch failure videos for anything that scored unexpectedly low and say why it failed, not just that it did.
3. **Update beliefs.** Keep `docs/findings.md` as the current best understanding: each claim, the evidence for it, the evidence against it, and how confident you are. Revise it when results disagree with it. Say so plainly when a result kills an idea.
4. **Propose.** List the next experiments that would most change `docs/findings.md`. For each: the question, the prediction, what each outcome would mean, the GPU-hours, and how long until an answer. Rank by information gained per GPU-hour.
5. **Launch.** Add the top ones to the cluster queue until the GPU cap is full. Never leave GPUs idle under the cap while the queue has ideas.
6. **Report.** Update `docs/MORNING_REPORT.md` and `docs/STATUS.md`. Then start the next cycle at once. While runs train, do the cheap work: analysis, failure sorting, code for the next experiment, writing.

### Rules that keep the results honest
- **One question per experiment.** Change one thing against a control trained in the same batch with the same seed and budget.
- **Fast proxies first.** Screen ideas on the 3 smoke-test tasks with short runs. Only what survives goes to all 18 tasks and full length.
- **Seeds before belief.** A difference counts as real only when it holds across 3 seeds or its interval clears zero. One run is a hint.
- **A locked test set.** Set aside the held-out objects and kitchens from Part Y as the final test. Tune and choose on a separate validation split. Score the final test set only for results going into the paper, and log every time you do.
- **Log everything, including failures.** Every experiment gets a row in `docs/experiment_registry.md` before it runs, with its prediction. Negative results stay in.
- **No quiet goalpost moves.** If you change a metric, a split or a protocol, write the reason and rerun the controls.
- **Check surprises twice.** A jump of more than 15 points from a small change means look for a bug first.
- **Spend by priority.** About 60% of GPU time on the main study (formats, with perfect and with agent-written commands), 25% on baselines and repeats, 15% on new ideas.

### Where the ideas come from, in order
1. Questions the latest results raise.
2. The weakest or least-tested claim in `docs/findings.md`.
3. Failure analysis: whatever failure type is most common, design an input or training change aimed at it.
4. This backlog: extend the format study from 3 tasks to all 18. Point-noise scale in training (0, 1, 3, 5 cm). Two extra tokens against a distance-to-point image channel. Depth only against a point map. Color plus point map against point map alone. Depth noise at test time. A learned "done" output. Chaining commands for composite tasks. More demos against fewer (100, 250, 500 per task). Model size (a smaller and a larger head). Chunk length. A grasp approach direction added to the command. Quantizing or compiling the policy.
5. Ideas from `docs/related_notes.md` that the other papers did not test.

### What stays with Adi
- Which claim the paper makes. Keep `docs/decisions_needed.md` current with the evidence for each option, and keep working on all live options until he chooses.
- Anything that needs a person: Duo, access requests, messages.

---

## Morning report format (`docs/MORNING_REPORT.md`)
1. **Bottom line** in three sentences.
2. **Numbers:** any success rates, with episode counts and the checkpoint step. Steps per second and hours to 100k and 300k steps.
3. **Done:** each item with its test result.
4. **Running:** job IDs, what they are, when they finish, how to check them.
5. **Blocked or failed:** what, what was tried, what you think the cause is.
6. **Needs Adi:** decisions and anything only he can do.
7. **Things to review by hand:** `policy/eval.py` and the replay-test conventions first.
8. **Surprises:** anything that contradicts `CLAUDE.md`, the DEM paper, or the plan.
