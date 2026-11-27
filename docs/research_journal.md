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
