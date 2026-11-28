# Diffusion Policy baseline

The end-to-end baseline for the paper: Diffusion Policy (DP, Chi et al. 2023) trained on the same demos,
the same splits and the same normalization as the DEM runs, and scored by the same loop (`policy/eval.py`).

## Implementation and version
- Model: `DiffusionModel` from LeRobot 0.4.4 (`lerobot.policies.diffusion.modeling_diffusion`), the CNN
  (1D conditional U-Net) variant of DP. Used unchanged. Our wrapper is `policy/diffusion_policy.py`.
- Why LeRobot: maintained, pinned on PyPI, pure PyTorch. The original real-stanford/diffusion_policy repo
  needs robomimic for its image encoder.
- Install: LeRobot's pins clash with envs/dem (torch<2.11, huggingface_hub<0.36), so it lives in an
  overlay folder, `pip install --no-deps --target` (`setup/setup_dp_overlay.sh`). Bouchet:
  `/nfs/roberts/project/pi_dr877/ak3288/envs/dp_overlay`. Mac: `~/apollo-lab/envs/dp_overlay`. The wrapper
  appends that folder to `sys.path`, so envs/dem's own packages always win. envs/dem is unchanged.
- `lerobot/policies/__init__.py` imports every policy (GR00T fails under transformers 5.x), so the wrapper
  registers `lerobot.policies` as a bare namespace and imports only the diffusion modules.

## Settings
| | DP baseline | DEM format study |
|---|---|---|
| Samples | `RoboCasaDataset`, every frame of every training demo | same |
| Cameras | left agent view + wrist, 256 px, resized to 128 then 90% crop (random in training, center at eval) | left + wrist, 256 px |
| Observation frames | 1 (`n_obs_steps=1`) | 1 |
| State | 16 numbers, stage-1 normalized (per-task q01..q99 to [-1, 1], `--quantiles dataset`), clipped to +-1 | same stats, z-scored after stage 1, no clip |
| Actions | 16 predicted, 8 executed, stage-1 normalized, clipped to +-1 | same, z-scored |
| Task input | one-hot task id appended to the state (DP has no language input) | instruction through frozen NeoBERT |
| Vision | one ResNet-18 per camera, group norm, from scratch, spatial softmax (32 keypoints) | DINOv3 ConvNeXt-B, fine-tuned |
| Head | 1D conditional U-Net, down dims 512/1024/2048, 274M parameters in total | MeanFlow / MSE head |
| Noise schedule | DDIM, 100 training steps, 10 inference steps, epsilon prediction, squared-cosine betas | n/a |
| Optimizer | AdamW lr 1e-4, betas 0.95/0.999, weight decay 1e-6, warmup 500, cosine, grad clip 10, fp32 (TF32) | AdamW 1e-4 / 5e-5 |
| Weight averaging | EMA, power 0.75, max 0.9999 (original DP; LeRobot has none). Checkpoints hold the EMA weights | none |
| Budget | 20k steps at batch 64 (3 tasks), 300k at batch 64 (18 tasks) | same |

Notes on the choices:
- Normalization: stage 1 is already DP's own scheme (map a robust range to [-1, 1]). Clipping at +-1 makes
  every training target fit DP's sample clipping (`clip_sample_range=1`). It clamps about 1% of action
  values to the q01 or q99 value. Unnormalization is the same code as DEM (`Quantile.unnormalize`).
- Padding: as DEM, the chunk past the end of a demo repeats the last action, and the loss covers it (DP's
  default, `do_mask_loss_for_padding=False`). LeRobot's own loader also drops the last 7 frames of each
  demo; we do not, so the sample set is exactly DEM's.
- Budget: equal steps and batch to the format study, so equal samples seen. This is much shorter than
  DP's native schedule (thousands of epochs), so DP may be undertrained relative to its paper. A longer run
  is a later item if DP looks budget-limited (rising success at the last checkpoints).
- DDIM 10 at inference: the DP paper uses DDIM with 100 training and 10 inference steps for its real-robot
  runs. On a CPU scoring node one chunk takes 0.3 s with 10 steps and 3 s with 100 (DDPM).
- One-hot task id: the smoke tasks each have a fixed goal type, but the one-hot does not say which drawer
  or which object. DEM's sentence does. This is part of the baseline's handicap and is stated with every
  number. A frozen text embedding of the instruction is the other option and is not built.

## How it plugs in
- Training: `python -m policy.train --format diffusion_policy <args>` hands over to `policy/dp_train.py`.
  Same folder layout as DEM runs: `config.json` (with `tasks`, `steps`, `format`), `norm_stats.json`,
  `ckpt_XXXXXXX.pt` (rua_format `diffusion_policy`), `train_state.pt`, `train_log.csv`, and the
  `TRAIN DONE` / `TRAIN STOPPED` lines. So `policy/slurm/train.sbatch` (time-limit resubmit), the watcher's
  resubmit of dead runs and its scoring all work unchanged.
- Scoring: `policy/model.py load_policy` loads a DP checkpoint, and `PolicyRunner` has a `diffusion_policy`
  branch: the same images (`sim.images`), the same 16-number state and stage-1 stats, then DP's sampler,
  then the same unnormalization. Everything else is the shared loop: target split envs, seeds 1000 + i,
  the task horizon, 16 predicted / 8 executed, the same results CSV (format `diffusion_policy`,
  objective `diffusion`, command_mode `task_onehot`).

      python -m policy.eval --ckpt $PROJ/checkpoints/dp_smoke/ckpt_0020000.pt --tasks OpenDrawer --episodes 50
      CKPT=... TASKS="PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave" EPISODES=100 \
        sbatch --array=0-2 policy/slurm/eval_array.sbatch           # what the watcher does

## Tests
- `tests/test_diffusion_policy.py`: loss and prediction shapes, clipping, one-hot, checkpoint round trip
  through `load_policy`, EMA, argument dispatch. Full suite passes (104 passed, 1 skipped, Mac).
- Mac, real OpenDrawer frames: 30-step train, resume to 40, one episode scored through `policy.eval`
  (CSV row written, format `diffusion_policy`). Small U-Net for speed.
- Bouchet: `policy/slurm/dp_devel_test.sbatch` (full model, 3 smoke tasks, batch 64, train 300, resume
  to 400, one episode per task on GPU and one on CPU). Status in the run table below.
- The zero test for DP: the step-0 untrained checkpoint should score about 0%. The 2500-step checkpoint
  of dp_smoke is the first one scored; an untrained DP checkpoint is not scored separately (to do if any
  early number looks off).

## Runs
| Run | Tasks | Split | Steps x batch | Seeds | Where |
|---|---|---|---|---|---|
| dp_smoke, dp_smoke_seed1, dp_smoke_seed2 | 3 smoke tasks | runs/smoke_train_minus50.json (last 50 demos held out) | 20k x 64 | 0, 1, 2 | queue.txt pool B |
| dp_all18 | 18 atomic-seen | all demos (as fs_mse_sentence_all18) | 300k x 64 | 0 | queue.txt pool B |

The watcher scores them as any other run: every 5k steps on 50 episodes for the 3-task runs, every 50k for
the 18-task run, and 100 episodes for each final checkpoint, on CPU nodes.

## Results
None yet.
