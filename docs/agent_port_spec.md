# Agent port: HexaShell on RoboCasa

Owner: William. Due: agent-alone results for all 18 tasks by Sunday morning, Nov 29.

## Goal
HexaShell's real tools running on RoboCasa, driven by an LLM, with success, time and tokens measured per episode.

## Setup
1. Run `bash setup/setup_dem.sh --assets`. It installs DEM, robosuite and RoboCasa into `~/apollo-lab` in a Python 3.11 environment called `dem`, and downloads the kitchen assets.
2. Pin versions:
   `git -C ~/apollo-lab/robocasa checkout 425684d`
   `git -C ~/apollo-lab/robosuite checkout 5ce6643`
3. Check it runs (from `~/apollo-lab/runs`, with `conda activate dem`):
   ```python
   import gymnasium as gym, robocasa
   env = gym.make("robocasa/PickPlaceCounterToCabinet", split="target", seed=0)
   obs, info = env.reset()
   for _ in range(20):
       obs, r, term, trunc, info = env.step(env.action_space.sample())
   env.close()
   ```
4. Request access to SAM 3 (HexaShell's segmentation model). It needs a GPU.

## Build, in order
1. Read HexaShell: https://github.com/HexaFuture/HexaShell and its `interface/EpisodeAPI.md`. The repo has the 14 tools but not the harness that runs them. It targets RoboDojo in Isaac Sim with a two-arm robot.
2. Implement `EpisodeAPI` on top of RoboCasa: camera images, depth, camera calibration, robot state, motion commands.
3. Get the 14 tools running against it. Perception uses SAM 3 plus geometry. Simulator ground truth is for grading only, never for acting.
4. Test each tool by hand, with no LLM. A commanded gripper move should end within 1 cm of its target. A located object should be within about 2 cm of the simulator's true position.
5. Script one pick-and-place by hand with the tools. It must succeed before the LLM is added.
6. Agent loop as in the HexaShell paper: the LLM writes one Python cell per turn that calls the tools. Turn limit per episode. Prompt caching on.
7. Episode runner: one process and one log file per episode, so episodes run in parallel.
8. Log per episode to `results/` (columns in `results/README.md`).

## Runs
1. Smoke test: 1 episode each of `PickPlaceCounterToCabinet`, `OpenDrawer`, `TurnOnMicrowave`. Report cost and time per episode before going further.
2. Then 10 episodes of each of those three.
3. Then all 18 tasks, 10 episodes each, in parallel, Saturday afternoon into the night. Use `split="target"`.

The 18 tasks: CloseBlenderLid, CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug, NavigateKitchen, OpenCabinet, OpenDrawer, OpenStandMixerHead, PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter, PickPlaceToasterToCounter, SlideDishwasherRack, TurnOffStove, TurnOnElectricKettle, TurnOnMicrowave, TurnOnSinkFaucet.

## Rules
- Keep `docs/hexashell_changes.md`: every change from the original HexaShell and why (their robot has two arms, ours has one arm on a mobile base).
- No API keys in the repo. Use a `.env` file, which is ignored by git.
- LLM spend: report the smoke-test cost first. The budget for the full run is set after that.
- Blocked for more than an hour: message Adi.

## Sunday
Adi adds a 15th tool, `run_skill`, that runs the trained DEM. The same 18-task run is repeated with it available.
