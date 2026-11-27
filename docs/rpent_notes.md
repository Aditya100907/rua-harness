# RPent (Harness VLA) as a baseline: notes

Part W. Written Nov 27 2026. Sources: RPent at commit `6b7255b` (Nov 27 2026), its README, `docs/source-en/rst_source/simulators/robocasa.rst`, `docs/source-en/rst_source/guides/configure_planner.rst`, `pyproject.toml`, `robots/robocasa/`, `robots/robocasa/eval/target50.json`, the Hugging Face model pages, and the GitHub compare API. I read the code; I did not run any of its scripts on the Mac.

## What it is
RPent ("Recursive Physical Agent", https://github.com/RLinf/RPent, paper arXiv 2607.08448, the "Harness VLA" system) is an LLM agent harness for robots. On RoboCasa the planner LLM gets tools (`view_env_state`, `back_project_batch`, `query_world_map`, motion primitives, `finish`) plus one learned skill, `rldx_skill`, which runs the RLDX-1 vision-language-action model (VLA: a policy that maps camera images, robot state and a sentence to actions) for up to 40 chunks of 8 actions. The planner talks to the VLA only through a sentence prompt. That makes it the "sentence" arm of our format question, with a much bigger policy than DEM.

Its RoboCasa benchmark, "Target50", is 50 tasks on the target split. Its 18 atomic tasks are exactly our 18 (`TARGET_TASKS["atomic_seen"]`, checked by name against `target50.json`). Atomic protocol: seeds 1 to 10, 1800 s per run, max 100 tool calls, success from `env._check_success()` (`state.success`). Reference planner: Codex, GPT-5.5, reasoning effort xhigh.

Note for a fair comparison: the default evaluation reads task memory from the `RLinf/RPent-memory` dataset. That memory includes per-task notes and a "recipe" (the winning tool calls) from an exploration run on seed 0 of the same task. It can be turned off or replaced with `--memory-profile local --memory-dir <dir>`.

## What one RoboCasa episode needs

| Item | Where | Gated? | Size | Status on Bouchet |
|---|---|---|---|---|
| RPent code | GitHub, Apache-2.0 | no | small | cloned, commit `6b7255b` |
| `rpent[robocasa]` extra | pulls `rpent[rlinf]` (RLinf `release/v0.4`), `rpent-robocasa365` (RLinf/robocasa `rpent`), `rlinf-rldx` (RLinf/RLDX-1 `rpent`), robosuite (RLinf/robosuite `rpent`) | no | several GB of wheels (torch, ray, deepspeed, transformers 4.57) | installed, `envs/rpent` |
| VLA checkpoint `RLWRLD/RLDX-1-FT-RC365` at revision `587e9ec` | Hugging Face | **not gated**, public. License "RLWRLD Model License v1.0": non-commercial (research is fine), attribution, share-alike, no military or surveillance use, indemnity clause | 13.8 GB (3 safetensors shards, bf16, about 7B parameters) | downloaded to `$SCR/models/rldx-1-ft-rc365` |
| RLDX support files `RLWRLD/RLDX-1-VLM` at `4b9f870` | Hugging Face | not gated, same license | 15 config files, about 16 MB (weights not needed) | downloaded to the HF cache in `$SCR/cache/hf` |
| Kitchen assets | `robocasa-download-assets` (about 10 GB) | no | | not downloaded. The fork reads `ROBOCASA_ASSETS_PATH`, so I point it at our existing 23 GB asset folder `$PROJ/apollo-lab/robocasa/robocasa/models/assets` (read only) |
| Task memory `RLinf/RPent-memory` | Hugging Face dataset | not gated | small | not downloaded (fetched at run time in `hf` mode) |
| SAM 3 (`facebook/sam3`) | Hugging Face | **gated (manual approval)**, 6.9 GB | | **not needed for RoboCasa.** Only the LIBERO, LIBERO-PRO, LIBERO-plus and RoboDojo extras use SAM 3. `robots/robocasa/` never imports it; the `[robocasa]` extra does not install it |
| Planner LLM | see next section | | | **missing; this is the blocker** |
| GPU | Linux, NVIDIA, CUDA and EGL, Python 3.10 (RLDX pins `==3.10.*`). Reference run: Torch 2.7.0, torchvision 0.22.0, CUDA 12.6. No VRAM figure is stated. The model is about 7B parameters in bf16, so plan on 24 GB or more; one GPU can hold both the simulator (`--env-cuda-device`) and the VLA (`--am-cuda-device`). L40S, A40, RTX 5000 Ada (32 GB) and larger should fit | | | |

### Planner options
`--planner` takes `api`, `claude_code`, `codex`, `gemini_cli`, `opencode` (and `flash`, which replays stored plans but exists only for LIBERO).
- `api`: direct model API with a key: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `OPENAI_API_KEY` (Responses or Chat Completions). Paid per token.
- `claude_code`: runs Claude Code through the Claude Agent SDK. Accepts a Claude login or `ANTHROPIC_API_KEY`. Default budget cap `--claude-code-max-budget-usd` 10 dollars. Terms: the Agent SDK docs say "Unless previously approved, Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK. Use the API key authentication methods..." and the SDK is under Anthropic's Commercial Terms. Using Adi's Pro/Max login inside RPent is therefore not clearly allowed; an API key is the documented path.
- `codex`: runs the Codex CLI through the OpenAI Codex SDK. Reuses an existing Codex login (ChatGPT plan sign-in) or `CODEX_API_KEY`/`CODEX_BASE_URL`. The RPent guide shows `codex login --device-auth`, which works on a headless node. This is the reference planner. Whether running Codex unattended inside a harness for hundreds of episodes fits the ChatGPT plan's usage terms and rate limits is for Adi to check; I did not.
- `gemini_cli`, `opencode`: reuse those CLIs' own logins and providers.
- Free option: point `codex` or `claude_code` at a local model server (`CODEX_BASE_URL=http://127.0.0.1:8000`, e.g. vLLM serving Qwen3.6-27B). No paid API, but it needs another GPU and its own environment, and it would not match the published GPT-5.5 numbers.

Also unchecked: whether Bouchet compute nodes can reach the planner's cloud endpoint. Login nodes can reach GitHub and Hugging Face.

## Commit comparison

| Component | Ours | RPent expects | Difference |
|---|---|---|---|
| RoboCasa | v1.0.1 `425684d` (upstream, Nov 12) | `RLinf/robocasa@rpent` = `2692d8f`, packaged as `rpent-robocasa365` | Forked from upstream `b4684e6` (Jul 8). The fork adds 10 commits: asset and macro paths from environment variables, packaging, an asset installer. Our pin is 16 commits past `b4684e6`: README changes, the subtask-labeling pipeline, `arrange_drinkware.py` (a composite task) and a duplicate function removed from `models/fixtures/sink.py`. None of these touch the 18 atomic tasks, so scenes and success checks should match. Not tested |
| robosuite | `5ce6643` | `RLinf/robosuite@rpent` = `97cfbde` | Exactly our commit plus one: adds a navigation camera `mobilebase0_navview` to `omron_mobile_base.xml`. RPent's navigation tools need it |
| MuJoCo | 3.3.1 | `>=3.3,<3.10` (both forks) | ours fits. The resolver picked 3.9.0; I set the rpent env to 3.3.1 to match ours |
| numpy | 2.2.5 | `>=1.24` (fork), RLDX `>=1.26` | compatible (rpent env has 2.2.6) |
| Python | 3.11 | 3.10 (RLDX pins it) | separate environment required |
| torch | DEM env's | `>=2.7`, reference 2.7.0 + CUDA 12.6 | separate environment |

Because RPent needs Python 3.10, its own forks and transformers 4.57, it cannot share `envs/dem`. It lives in its own env, and our pins are untouched.

## What I installed and where (Bouchet)
- Code: `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/RPent` at `6b7255bf86dfdd7fc2feece5ff5605b949441a89`.
- Environment: `/nfs/roberts/project/pi_dr877/ak3288/envs/rpent` (conda, Python 3.10, uv inside it), `torch==2.7.0 torchvision==0.22.0` (cu126), then `uv pip install -e ".[robocasa,test]"` (run as a CPU batch job, see below). `envs/dem` was not touched.
- Caches: `/nfs/roberts/scratch/pi_dr877/ak3288/cache/{conda_pkgs,pip,uv,hf,tmp}`.
- Install script and log: `$PROJ/apollo-lab/logs/rpent_install.sh`, `rpent_install.log`; frozen package list `rpent_installed_requirements.txt`.
- Weights: `/nfs/roberts/scratch/pi_dr877/ak3288/models/rldx-1-ft-rc365` (RLDX-1-FT-RC365 at `587e9ec`); support files in the HF cache under `$SCR/cache/hf`. Script and log: `$PROJ/apollo-lab/logs/rpent_weights.sh`, `rpent_weights.log`. Scratch is purged after 30 days; copy to project space if we keep this baseline.

### Install result
- Login node: conda env and torch went in. The big `uv pip install` was killed on the login node (first "Cannot allocate memory", then "Killed"). It finished in a CPU batch job instead (`day` partition, 4 CPUs, 32 GB, job 34845626, 8.5 min). Compute nodes reach GitHub, PyPI and Hugging Face.
- `uv pip check`: all 272 packages compatible.
- Resolved versions: torch 2.7.0+cu126, torchvision 0.22.0+cu126, transformers 4.57.6, numpy 2.2.6, rlinf `2348ece`, rlinf-rldx `ebcfd13`, rpent-robocasa365 `2692d8f`, robosuite `97cfbde`.
- MuJoCo: the resolver picked 3.9.0. I replaced it with **3.3.1** in the rpent env so the physics matches our DEM runs. This is a change from what RPent's resolver gives, so record it with any result. To undo: `uv pip install mujoco==3.9.0` in the rpent env.
- Weights: CPU batch job 34846496. All three shards match the sizes on Hugging Face (13 GB on disk). The 15 support files are in `$SCR/cache/hf/hub/models--RLWRLD--RLDX-1-VLM/snapshots/4b9f870...`.

### Test job (gpu_devel, job 34846509, RTX 5000 Ada 32 GB, 55 s, no LLM)
Script: `$PROJ/apollo-lab/logs/rpent_devtest.sbatch`, log `rpent_devtest_34846509.log`.
- Simulator check `tests/e2e_tests/robocasa/test_environment_smoke.py`: **4 of 4 passed**: environment contract for OpenDrawer, NavigateKitchen and PickPlaceCounterToCabinet (target split, seed 1), and the navigation camera check. This ran against our existing asset folder through `ROBOCASA_ASSETS_PATH`, with MuJoCo 3.3.1 and EGL.
- VLA check `test_components.py::test_rldx_component` and scripted chain `test_policy_chain.py` (one `rldx_skill` chunk, no planner): **both errored in test setup, before loading the model**: `AttributeError: 'Namespace' object has no attribute 'vla_model_path'` at `tests/e2e_tests/robocasa/scenario.py:180` and `:216`. This is a bug in RPent's test file at `6b7255b`: the CLI argument is now `--am-checkpoint` (read with `select_am(args, ...)`), but the test still reads the old name. So RLDX loading and inference are **not yet verified**, and neither is GPU memory use. I used my one gpu_devel job and did not resubmit.

## What blocks one logged episode
Present now: code, environment, VLA weights and support files, assets, working simulator on a Bouchet GPU. Missing, in order:
1. **A planner LLM with credentials on Bouchet (needs Adi).** Every online planner calls a model. The choices: (a) an API key (Anthropic, OpenAI, or Gemini), which is paid and needs Adi's approval. An atomic episode allows up to 100 tool calls with images, so one episode can cost a few dollars at xhigh effort, and the 180-run atomic matrix costs far more. (b) Adi's Codex (ChatGPT plan) login via `codex login --device-auth` with `CODEX_HOME` in project space. This matches the reference planner, but Adi must check the plan terms and rate limits. (c) Claude Code with Adi's claude.ai login, which the Agent SDK docs say is not allowed without approval, so do not use it. (d) A local open model behind vLLM: free, but needs a second environment and GPU, and the result is not comparable to the published numbers. Recorded as decision 5 in `docs/decisions_needed.md`.
2. **VLA check not run.** Fix the stale attribute in RPent's test (`args.vla_model_path` to `args.am_checkpoint`, two lines in `tests/e2e_tests/robocasa/scenario.py`, or run the component through `rpent` directly), then run `test_rldx_component` and `test_policy_chain` in one gpu_devel job. This needs no LLM and can run before decision 5. It confirms RLDX loads on 32 GB and that `rldx_skill` moves the robot.
3. **Memory setting.** Decide whether the baseline uses RPent's published seed-0 task memory (its reference protocol) or none (closer to a fresh agent). It changes the numbers, and both are defensible. Default: run their protocol as published, and also with no task memory if the budget allows.
4. **Results plumbing.** A small wrapper that turns `result.json` (success, time, tool calls; tokens come from the transcript) into a row in `results/` with our columns and the commit. Not written.
5. **Network from compute nodes to the planner endpoint.** Compute nodes reach GitHub and Hugging Face, so OpenAI and Anthropic endpoints are likely reachable. Not tested.

Not blockers: SAM 3 (not used on RoboCasa), the 10 GB asset download (we reuse ours), Duo (none needed so far).

## Run command, once a planner is chosen
```bash
module load miniconda; conda activate /nfs/roberts/project/pi_dr877/ak3288/envs/rpent
export MUJOCO_GL=egl HF_HOME=/nfs/roberts/scratch/pi_dr877/ak3288/cache/hf HF_HUB_CACHE=$HF_HOME/hub
export ROBOCASA_ASSETS_PATH=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/robocasa/robocasa/models/assets
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8; unset RLDX_RESET_SEED
cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/RPent
rpent-check-llm --planner codex --model gpt-5.5
rpent --robot robocasa --task-name OpenDrawer --split target --seed 1 \
  --am-checkpoint /nfs/roberts/scratch/pi_dr877/ak3288/models/rldx-1-ft-rc365 \
  --env-cuda-device 0 --am-cuda-device 0 \
  --planner codex --model gpt-5.5 --reasoning-effort xhigh \
  --max-turns 100 --planner-timeout-s 1800 \
  --output-dir /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/runs/rpent/OpenDrawer_s1
```
Results land in `result.json`, `transcript_*.json` and `run.log` in the output directory. A wrapper would still need to turn `result.json` into a CSV row in `results/` with our columns.
