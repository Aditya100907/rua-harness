# Research journal

One entry per result: what was tested, the number with its episode count and interval, what it means, what it rules out. Newest last. Started Nov 27, 00:16.

## Nov 27, 00:16. Pre-gate checks
- **Tested:** replay of recorded demo actions (30 demos), the same through the evaluation loop (30), an untrained head (28 episodes).
- **Numbers:** replay 30/30; wiring 30/30; untrained 0/28 (95% interval 0 to 12%).
- **Means:** the action convention (LeRobot to robosuite reorder, gripper and base mode snapped to +-1), the state vector, the camera choice and flip, and the env construction all match the data. A trained checkpoint that scores 0 cannot be blamed on these.
- **Rules out:** absolute vs relative action mix-ups, wrong action order, wrong image orientation.

## Nov 27, 00:16. Training speed
- **Tested:** 300 steps, batch 64, 3 tasks, H100 (job 34841772).
- **Numbers:** 3.27 steps/s, 83% GPU busy, 0% data wait. L40S: 1.16 steps/s.
- **Means:** the paper's 300k steps take about 25 hours on one H100. The data loader is not the bottleneck. A 5k-step screening run on 3 tasks costs about 25 minutes of H100 time.

## Nov 27, 00:18. Commander smoke test (1 episode, a 50-step checkpoint, so success is not expected)
- **Tested:** the commander loop end to end on OpenDrawer seed 1000 (Claude Opus 5.5 through the CLI on the subscription).
- **Numbers:** 62 s wall, 11 turns, 283k input tokens (256k of them cache reads), 1.2k output tokens, $0.29 at list price. Two run_skill calls.
- **What it shows:** the agent first pointed at the drawer handle in the left camera; the lifted point was 81.5 cm from the handle, because the arm hides the handle in that view and the pixel's depth lands on the arm. Its second command used the right camera: 3.1 cm. The same occlusion shows in the depth re-render check (left camera fails the grasp-pixel test in 8 of 10 demos for the same reason).
- **Means:** pointing error depends heavily on the camera the agent picks. Worth telling the agent which camera sees the target better, or lifting from both cameras and taking the nearer consistent point. This goes into Part V's design.

## Nov 27, 06:30. First trained-policy numbers are near zero
- **Tested:** watcher evals (50 episodes per task, 100 at 20k) of run1_smoke (stock DEM, left+right) at 5k and 10k, run3 at 5k, f4_kv_text at 5k to 20k.
- **Numbers:** run1 5k: 0/50, 0/50, 0/50; run1 10k: 0/50 on all three. f4_kv_text 20k: PickPlace 0/100, OpenDrawer 0/100, TurnOnMicrowave 7/100 (7%, 95% CI 3 to 14%). Paper category numbers (18 tasks, 300k steps): drawers 79.5%, buttons 81.5%, pick-place 33.2%. Scripted oracle on the same seeds: 46%, 82%, 100%.
- **Renders checked clean** (img_rowdiff0 4 to 6). Video of an OpenDrawer failure: the arm moves down and right but never reaches the handle (min distance 41 to 79 cm).

## Nov 27, 06:45. Part K investigation: the model has not learned the actions
- **Open-loop on its own training frames (run1 20k, 150 frames per task):** MAE on the 8 executed arm actions: OpenDrawer 0.191, Microwave 0.191, PickPlace 0.233, versus 0.159 to 0.160 for always predicting the mean action and 0.05 to 0.06 for repeating the current action. Gripper sign agreement: 83%, 91%, 53% (chance on pick-place).
- **Conditioning check (OpenDrawer, 32 frames):** with fixed noise, predictions vary 0.088 across frames; on a fixed frame they vary 0.183 across noise draws. The noise dominates the output. With z0 = 0 the MAE is 0.129 (vs 0.155 for the mean action). Zeroing the images: 0.135 (images barely used). Zeroing the state: 0.245 (state used). Run2 at 70k (18 tasks): the same picture (0.093 vs 0.182).
- **Training log:** total loss falls from 0.74 to -0.6, but raw_mse (the flow-matching error) stays around 1 (0.6 to 2.0) for the whole run; the fall is the dispersive term (weight 0.25), which can be negative.
- **Normalization:** heavy tails in near-constant dims (PickPlace base action dim 2 has a q01..q99 range of 0.02, so rare base moves map to z-scores of 10+; base height in the state has ranges of 0.0001 to 0.007).
- **Rules out:** eval-time input mismatch (the model fails on training frames too); render corruption.
- **Live hypotheses, tested now:** (a) undertraining of the flow model, (b) dispersive term dominating, (c) batch 64 too small, (d) outliers from the quantile normalization, (e) head learning rate. Ablations abl_nodisp, abl_bs128, abl_clip3, abl_lrhead3e4 (5k steps, 3 tasks) against run1 5k; eval-only tests of run1 20k with z0 = 0 and with 4 sampling steps; a one-demo overfitting test on the Mac.

## Nov 27, 07:05. One-demo overfitting test (Mac, MPS)
- **Tested:** train on OpenDrawer demo 0 only (334 frames), batch 8, 400 steps (about 10 passes), recipe defaults (head lr 1e-4, dispersive 0.25); open-loop on that same demo.
- **Numbers:** raw_mse 1.20 to 0.69. Open-loop arm MAE 0.182 vs 0.157 for the mean action and 0.062 for repeating the current action. Gripper sign 66%.
- **Means:** the model cannot memorize one demo in 400 steps. Not decisive on its own: DEM's own toy test needs 1,200 steps at lr 3e-4 and batch 256 to contract onto a single 2-D point, so MeanFlow converges slowly. The DEM paper trained on 8 H100s for 300k steps, so its effective batch was likely 8x ours or more.
- **Next:** the same test with lr 3e-4 and no dispersive term (Mac, running); a 3,000-step, batch-64 version on Bouchet once SSH is back.

## Nov 27, 16:00. Sampling changes do not rescue the policy; more training alone does not either
- **Tested:** run1 20k evaluated with zero noise (deterministic) and with 4 MeanFlow steps; run2 (18 tasks) at 70k.
- **Numbers:** z0 = 0: PickPlace 0/30, Drawer 0/30, Microwave 0/30. nfe = 4: 0/30, 2/30 (7%), 2/30 (7%). run2 70k: 0/24, 0/24, 0/24.
- **Means:** the failure is not the one-step sampler (4 steps helps at most a few points) and not only the 3-task budget (run 2 at 70k steps on 18 tasks also fails). The training signal itself is the suspect.
- **raw_mse is spiky:** run 2's mean raw_mse over steps 50k to 55k is 35.8 (vs about 1 elsewhere). Rare samples with huge normalized targets, consistent with the near-constant dims whose q01..q99 range is tiny (PickPlace base action dim 2: range 0.02). These spikes may be what keeps the head from learning; the clip ablation tests part of this.

## Nov 27, 16:40. Objective tests (open-loop and conditioning, OpenDrawer training frames; arm MAE on the 8 executed steps)
| Checkpoint | What changed | Arm MAE | Mean-action MAE | Notes |
|---|---|---|---|---|
| overfit1_mse | regression, 1 demo, 2k steps, batch 64 | **0.009** | 0.157 | gripper 100%; memorized |
| overfit1_paper | MeanFlow, 1 demo, 2k steps, batch 64, left + wrist, shipped quantiles, head lr 1e-4 | 0.095 (random noise), 0.124 (zero noise) | 0.157 | gripper 92%; first MeanFlow model to beat the mean action |
| overfit1_meanflow | MeanFlow, 1 demo, 2k steps, batch 64, left + right, head lr 3e-4 | 0.210 | 0.157 | gripper 61% |
| abl_mse (3 tasks, 5k) | regression | 0.128 (zero noise) | 0.155 | input spread n/a (deterministic) |
| abl_fm (3 tasks, 5k) | plain flow matching, 10 steps | 0.145 (zero noise) | 0.155 | input spread 0.0125 vs noise 0.22 |
- **Means:** the data, vision and head pipeline can fit (regression memorizes a demo). MeanFlow can fit one demo with the paper's cameras and learning rate, but overfit1_paper and overfit1_meanflow differ in two things (cameras, head lr), so this does not yet say which one mattered. On 3 tasks at 5k steps, both new objectives are only slightly better than the mean action; whether that is enough to act in the simulator is what the gate evals decide.
- **mmBERT path:** the research-checkpoint converter and --init-from fine-tune run with mmBERT-small (384-d) tokens (job 34893756).

## Nov 27, 16:52. THE GATE PASSES with regression (abl_mse, 5k steps)
- **Tested:** abl_mse (objective mse = deterministic regression through the same DEM head; cameras left + right; recomputed quantiles; 3 smoke tasks; 5k steps; batch 64; seed 0), evaluated with zero noise, 50 episodes per task, seeds 1000 to 1049 (job 34893757).
- **Numbers:** OpenDrawer **22/50 (44%, 95% CI 31 to 58%)**; PickPlaceCounterToCabinet 0/50; TurnOnMicrowave 0/50.
- **Checked before believing it:** 50 rows, zero noise recorded, renders clean (img_rowdiff0 4.3 to 4.6), successes end at steps 210 to 290 (demo lengths 200 to 430), the success video shows the arm reaching the lower cabinet and the drawer pulled out.
- **Means:** with the same data, vision, head and eval code, replacing MeanFlow by regression turns 0% into 44% on the drawer after only 5k steps, where MeanFlow had 0% at 20k (run1) and at 70k (run2). The pipeline was never the problem; MeanFlow's optimization in our setup is. The gate rule (beat 20% on OpenDrawer or TurnOnMicrowave) is met.
- **Still to see:** abl_fm (plain flow matching, 10 steps) gate numbers; then choose the objective (fm if close, per Adi).

## Nov 27, 16:54. Objective chosen: mse (regression)
- **abl_fm (plain flow matching, 5k steps, 10 sampling steps), 50 episodes per task:** PickPlace 0/50, OpenDrawer 0/50, TurnOnMicrowave 0/50. abl_mse at the same steps: 0/50, 22/50, 0/50. Not close, so the format study runs on mse (Adi's rule: prefer fm only if close). fs_fm_sentence keeps training to 20k with the format-study settings, to see whether fm catches up with more steps.
- **Format study launched** (watcher queue, one GPU each, 3 smoke tasks, 20k steps, seed 0, held-out-50 split, left + wrist, shipped quantiles, objective mse): sentence (fs_mse_sentence, already running), kv_text, points_2d, points_3d_rgb, points_3d, pose_path; then the 18-task sentence run (300k steps). No format result counts until it is scored with the same protocol; each is scored every 5k steps, 50 episodes per task (100 at 20k).

## Nov 27, 17:01. MeanFlow batch-size test (priority 4): bigger batch does not fix MeanFlow at 5k
- **Tested:** mf_ddp4_bs256: MeanFlow, 4 GPUs x 64 = global batch 256, 3 smoke tasks, 5k steps, seed 0, left + right cameras, recomputed quantiles. Only change vs run1_smoke at 5k: the global batch (64 -> 256).
- **raw_mse over steps 4k to 5k:** run1 mean 1.148 / median 0.894; batch 256 mean 1.185 / median 0.804 (51 logs each). Barely different.
- **Conditioning check (OpenDrawer, 32 frames):** MAE with zero noise 0.152 vs 0.155 for the mean action (run1 5k: 0.151). Input spread with fixed noise 0.094 (run1 5k: 0.060); noise spread 0.183 (run1 5k: 0.230). Zeroing the images raises MAE to 0.171 (images used a little more than before).
- **Answer:** no. With a 4x larger batch, neither raw_mse nor the conditioning check beats the mean action at 5k steps. The output depends on the inputs somewhat more, but is still dominated by the sampling noise. For contrast, regression at 5k with batch 64 scores 22/50 on OpenDrawer. Simulator eval of this checkpoint (50 episodes per task) is running (34894664) for the record. Per Adi's rule, no other MeanFlow run is launched.

## Nov 27, 17:26. Determinism fix does not change the gate score
- **Tested:** abl_mse 5k on OpenDrawer, seeds 1000 to 1049, 50 episodes each: the original run (no fix), a second no-fix run, and a run with HexaShell's counter-region fix (policy/determinism.py).
- **Numbers:** original 22/50 (44%); no-fix rerun 22/50 (44%); with fix 21/50 (42%, 95% CI 29 to 56%). Per-seed agreement with the original: no-fix rerun 48/50, with fix 47/50. Without the fix, 1 of 50 OpenDrawer scenes changed between runs (a different instruction).
- **Scene test (separate processes, seeds 1000 to 1004):** without the fix, PickPlaceCounterToCabinet rebuilt differently for 3 of 5 seeds (seed 1003: different layout and target object), OpenDrawer for 1 of 5; with the fix, all identical.
- **Means:** the fix matters for reproducibility (especially pick-place) but does not move the OpenDrawer number beyond noise; the gate still passes with it. Every eval after about 17:25 uses it (on by default); results before that ran without it.

## Nov 27, 17:28. Format study, first score: sentence arm at 5k
- **Tested:** fs_mse_sentence (regression; cameras left + wrist; shipped quantiles; held-out-50 split; seed 0; 5k of 20k steps), 50 episodes per task, seeds 1000 to 1049, with the determinism fix (watcher eval 34895234).
- **Numbers:** OpenDrawer 8/50 (16%, 95% CI 8 to 29%); PickPlaceCounterToCabinet 5/50 (10%, 4 to 21%); TurnOnMicrowave 4/50 (8%, 3 to 19%). First pick-place and microwave successes of any learned policy here.
- **Compared with abl_mse at 5k** (left + right, recomputed quantiles, all demos, no determinism fix): 22/50, 0/50, 0/50. Not a clean comparison: cameras, quantile source, training split and the determinism fix all differ (quantiles were shown identical; the fix moved OpenDrawer by 1 episode). The camera change is the likely cause of the shift, but this is not tested as one change.
- **Counts toward the format study** (gate passed; same protocol for every format). 10k scoring running (34896150).

## Nov 27, 18:00. Format study, early scores (one seed; 50 episodes per task; determinism fix on)
| Format, step | OpenDrawer | PickPlace | Microwave |
|---|---|---|---|
| sentence 5k | 8/50 (16%) | 5/50 (10%) | 4/50 (8%) |
| sentence 10k | 10/50 (20%) | 8/50 (16%) | 10/50 (20%) |
| sentence 15k | 25/50 (50%) | 14/50 (28%) | 26/50 (52%) |
| kv_text 5k | 20/50 (40%) | 4/50 (8%) | 5/50 (10%) |
| points_2d 5k | 28/50 (56%) | 6/50 (12%) | 1/50 (2%) |
- 95% CIs at 5k on OpenDrawer: sentence 8 to 29%, kv_text 27 to 54%, points_2d 42 to 69%: the drawer intervals for kv_text and points_2d clear sentence's at 5k. One seed and an early checkpoint, so a hint, not a finding. On the microwave and pick-place, the 5k differences are inside the noise.
- sentence improves fast between 10k and 15k (drawer 20% to 50%, microwave 20% to 52%); 20k scoring is running.
- Seeds before belief: queued fs_mse_sentence_seed1 (same settings, seed 1) on the free GPU.

## Nov 27, 18:35. Sentence arm final (20k, 100 episodes per task) and 10k scores for kv_text and points_2d
| Format, step | OpenDrawer | PickPlace | Microwave |
|---|---|---|---|
| sentence 20k (final) | 46/100 (46%, 36 to 56%) | 26/100 (26%, 18 to 35%) | 43/100 (43%, 34 to 53%) |
| kv_text 10k | 33/50 (66%, 52 to 78%) | running | 18/50 (36%, 24 to 50%) |
| points_2d 10k | 25/50 (50%, 37 to 63%) | running | 15/50 (30%, 19 to 44%) |
- Stock-sentence regression policy, one seed, 3 tasks, 20k steps, left + wrist: 46%, 26%, 43%. DEM paper category numbers (18 tasks, 300k steps, MeanFlow): drawers 79.5%, pick-place 33.2%, buttons 81.5%. Scripted oracle on the same seeds: 46%, 82%, 100% (pick, drawer, microwave). The learned policy now beats the scripted oracle on nothing except tying pick-place; it is still below it on drawer and microwave.
- sentence 20k vs 15k: drawer 46% vs 50%, microwave 43% vs 52%, inside the noise (100 vs 50 episodes).
- kv_text at 10k already scores 66% on the drawer (vs sentence 46% at 20k; intervals 52 to 78% vs 36 to 56% just overlap). Hint only: one seed, different step counts.

## Nov 27, 18:55. More early scores
| Format, step | OpenDrawer | PickPlace | Microwave |
|---|---|---|---|
| kv_text 10k | 33/50 | 9/50 | 18/50 |
| points_2d 10k | 25/50 | 6/50 | 15/50 |
| points_3d_rgb 5k | 10/50 | 0/50 | 0/50 |
| pose_path 5k (held-out demo scenes, demo's own future path) | 5/50 | 1/50 | 4/50 |
- The two formats that add learned point or path tokens start slower than the text and drawn-point formats (5k, one seed). Their new MLP starts from scratch while everything else is shared, so check again at 20k before reading anything into it.
- pose_path is scored on a different protocol (held-out demo scenes with the demo's future path), so its numbers are not directly comparable with the others' fresh-seed scenes.

## Nov 28, 13:32. FORMAT STUDY RESULTS (3 smoke tasks, regression objective, 20k steps, 100 episodes per seed per task, determinism fix on)
Settings shared by every run: left + wrist cameras, shipped quantiles, held-out-50 split, batch 64, seed 0 unless noted. Brackets: pooled Wilson 95% CI.
| Format | Seeds | OpenDrawer | PickPlace | Microwave |
|---|---|---|---|---|
| sentence | 3 | 53% (48 to 59); per seed 47, 67, 46 | 22% (17 to 27); 11, 28, 26 | 44% (39 to 50); 45, 45, 43 |
| kv_text | 3 | 55% (49 to 60); 52, 51, 61 | 23% (19 to 28); 24, 31, 15 | 48% (42 to 53); 56, 41, 46 |
| points_2d | 3 | **65% (59 to 70)**; 64, 69, 62 | 23% (19 to 28); 22, 28, 20 | **56% (51 to 62)**; 65, 60, 44 |
| pose_path (held-out demo scenes, demo's own path) | 1 | 56% | 16% | 38% |
| points_3d (point map + point tokens) | 1 | 41% | 12% | 18% |
| points_3d_rgb (color + point tokens) | 1 | 33% | 7% | 25% |
- **points_2d beats sentence** on the microwave (intervals clear) and on the drawer (intervals just touch at 0.59); same on pick-place. kv_text is not different from sentence anywhere.
- **Learned point tokens (points_3d, points_3d_rgb) are worse** than sentence at 20k (one seed): the new token MLP may need more steps, or the point carries little the image does not.

## Nov 28, 13:32. Part N sweeps (points formats, final checkpoints, 50 episodes per mode per task)
- **N1 point error (0, 1, 2, 5, 10 cm):** almost flat. points_2d seed 0 drawer 37, 37, 37, 36, 36 of 50; microwave 32, 31, 29, 31, 26; pick-place 15, 17, 19, 11, 11. Same flatness for the other seeds and for points_3d / points_3d_rgb.
- **N2 distractor point (pick-place):** with the grasp point placed on a distractor object, the gripper came within 5 cm of the distractor in 5, 1, 4 of 50 episodes (points_2d seeds 0, 1, 2), 1 (points_3d), 2 (points_3d_rgb), and within 5 cm of the true target in 18, 27, 10, 11, 17. With the correct point, within 5 cm of the target in 31, 26, 30, 11, 19. **The policies do not follow the point to a different object; they follow the name and the image.** The wrong point does lower reaching of the true object (points_2d seed 2: 30 to 10).
- **N3 dropped fields:** dropping the object name or renaming it to "thing" collapses OpenDrawer for points_2d seeds 0 and 2 (37 to 2, 36 to 0) and for points_3d / points_3d_rgb (to 0), but not seed 1 (28 to 24). Dropping the point: drawer 37 to 17 (seed 0), 36 to 5 (seed 2), 28 to 29 (seed 1). Dropping the source name: no change.
- **Means:** in this training setup the policy uses the text far more than the point. Points help success on the drawer and microwave (seen in the format table), but they do not steer the policy to a different object. Point accuracy beyond about 10 cm is not the bottleneck, so an agent's pointing error (commander smoke test: 3 cm with a clear view) is not what would limit it. Caveat: training points come from the demo's own target, so the point and the name never disagree in training; the policy has no reason to prefer the point. A training set with look-alike distractors (Part Y) is the test of whether points can carry the choice.

## Nov 28, 13:32. 18-task stock run (sentence, regression, left + wrist, 300k steps, one seed)
- Mean success over 18 tasks: 50k 37.0% (333/900), 100k 40.8%, 150k 47.8%, 200k 47.9%, 250k 49.8%, 300k **51.2% (921/1800)**. DEM paper (MeanFlow): 55.6%.
- Per task at 300k (100 episodes each): OpenStandMixerHead 88, PickPlaceToasterToCounter 81, OpenCabinet 78, TurnOnElectricKettle 78, TurnOnSinkFaucet 75, OpenDrawer 73, SlideDishwasherRack 64, CloseToasterOvenDoor 61, CloseFridge 60, PickPlaceSinkToCounter 51, TurnOnMicrowave 51, CoffeeSetupMug 48, PickPlaceCounterToCabinet 41, PickPlaceDrawerToCounter 25, TurnOffStove 21, PickPlaceCounterToStove 16, CloseBlenderLid 10, NavigateKitchen 0.
- **Means:** with regression in place of MeanFlow, DEM on our pipeline comes within 4.4 points of the paper's 18-task number. Training on all 18 tasks also helps the smoke tasks a lot (OpenDrawer 73% vs 46 to 67% for 3-task runs; pick-place 41% vs 11 to 28%).

## Nov 28, 14:30. Relabel to test-time points: how far the old training points were
- Grasp-point shift, hindsight (gripper-close) label vs test-time perfect command, all demos: PickPlaceCounterToCabinet median 2.3 to 2.8 cm per chunk (p95 up to 8.6, max 15.9); OpenDrawer median 2.4 to 3.1 cm (max 6.6); **TurnOnMicrowave median 18.8 to 22.1 cm (p95 32 to 38, max 44)**.
- **Means:** for the microwave, the old point formats were trained with points about 20 cm from the start button, the point they got at test. The old microwave scores of point formats (points_2d 56%, points_3d 18%, points_3d_rgb 25%) were measured under that mismatch and do not count. The relabelled runs (P0) are the clean comparison.

## Nov 28, 14:42. First agent-written commands (open model, pilot)
- **Tested:** Qwen3-VL-8B-Instruct (vLLM on Bouchet, free) as the commander: it sees the left, right and wrist images and writes one command (names plus a pixel per object), lifted to 3D with depth; up to 6 calls per episode. Checkpoints: fs_mse_sentence and fs_mse_points_2d (seed 0, 20k, **old labels**). 10 episodes per smoke task, seeds 1000 to 1009. Perfect-command success on the same checkpoints and seeds for comparison.
| Checkpoint | Drawer | PickPlace | Microwave | Total | Perfect commands |
|---|---|---|---|---|---|
| sentence | 2/10 | 2/10 | 1/10 | 5/30 | 10/30 |
| points_2d | 0/10 | 0/10 | 6/10 | 6/30 | 17/30 |
- Commander pointing error (grasp point vs ground truth, median): drawer 15.2 to 16.5 cm, pick-place 15.3 cm, microwave 9.8 cm. Median 1 to 2 model calls and 1.3k to 3.5k input tokens per episode; 2 to 3.5 minutes per episode on CPU nodes.
- **Means (pilot, one seed, n=10 per cell):** with perfect commands points_2d leads sentence 17 to 10; with the agent writing the commands they tie (6 vs 5). points_2d holds on the microwave, where the agent's error is about 10 cm, and collapses on the drawer and pick-place, where it is about 15 cm, consistent with the capture radius seen in the tolerance sweep (drawer seed 0: 37/50 at up to 10 cm, 28/50 at 20 cm, 13/50 at 40 cm). This is the pattern the research predicted (coordinate formats lose more under a real writer). Sentence also drops (10 to 5), likely because the agent's wording differs from the training instructions.
- **Next:** repeat on the relabelled checkpoints (P0) and with the P-2 drawing; add the decode-first snap row (P8) on the logged commands; more episodes.
