# Research journal

One entry per result: what was tested, the number with its episode count and interval, what it means, what it rules out. Newest last. Started Nov 27, 00:20.

## Nov 27, 00:20. Pre-gate checks
- **Tested:** replay of recorded demo actions (30 demos), the same through the evaluation loop (30), an untrained head (28 episodes).
- **Numbers:** replay 30/30; wiring 30/30; untrained 0/28 (95% interval 0 to 12%).
- **Means:** the action convention (LeRobot to robosuite reorder, gripper and base mode snapped to +-1), the state vector, the camera choice and flip, and the env construction all match the data. A trained checkpoint that scores 0 cannot be blamed on these.
- **Rules out:** absolute vs relative action mix-ups, wrong action order, wrong image orientation.

## Nov 27, 00:20. Training speed
- **Tested:** 300 steps, batch 64, 3 tasks, H100 (job 34841772).
- **Numbers:** 3.27 steps/s, 83% GPU busy, 0% data wait. L40S: 1.16 steps/s.
- **Means:** the paper's 300k steps take about 25 hours on one H100. The data loader is not the bottleneck. A 5k-step screening run on 3 tasks costs about 25 minutes of H100 time.

## Nov 27, 00:35. Commander smoke test (1 episode, a 50-step checkpoint, so success is not expected)
- **Tested:** the commander loop end to end on OpenDrawer seed 1000 (Claude Opus 5.5 through the CLI on the subscription).
- **Numbers:** 62 s wall, 11 turns, 283k input tokens (256k of them cache reads), 1.2k output tokens, $0.29 at list price. Two run_skill calls.
- **What it shows:** the agent first pointed at the drawer handle in the left camera; the lifted point was 81.5 cm from the handle, because the arm hides the handle in that view and the pixel's depth lands on the arm. Its second command used the right camera: 3.1 cm. The same occlusion shows in the depth re-render check (left camera fails the grasp-pixel test in 8 of 10 demos for the same reason).
- **Means:** pointing error depends heavily on the camera the agent picks. Worth telling the agent which camera sees the target better, or lifting from both cameras and taking the nearer consistent point. This goes into Part V's design.
