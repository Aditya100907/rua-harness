# Hindsight command labels (E2, F1)

Code: `policy/label_commands.py`. Job: 34842819 (18-task array, day partition). Output: `$SCR/labels/<task>.jsonl` (Bouchet), `~/apollo-lab/runs/labels/` (Mac, 3 smoke tasks).

Per episode: the `Command` (skill, object and target names, points in the robot base frame at frame 0), `kv_text`, the grasp and place points in the world frame, the frames of gripper close and open, the object category and fixture refs from `ep_meta.json`, and the method used.

## Counts and methods
Every task's label count equals its episode count.

| Task | Episodes | Method |
|---|---|---|
| CloseBlenderLid | 502 | gripper close 502 |
| CloseFridge | 513 | success frame 509, gripper close 4 |
| CloseToasterOvenDoor | 505 | success frame 334, gripper close 171 |
| CoffeeSetupMug | 502 | close/open 502 |
| NavigateKitchen | 500 | final base position 500 |
| OpenCabinet | 500 | gripper close 500 |
| OpenDrawer | 514 | gripper close 514 |
| OpenStandMixerHead | 502 | success frame 502 |
| PickPlaceCounterToCabinet | 502 | close/open 502 |
| PickPlaceCounterToStove | 501 | close/open 501 |
| PickPlaceDrawerToCounter | 501 | close/open 501 |
| PickPlaceSinkToCounter | 501 | close/open 501 |
| PickPlaceToasterToCounter | 512 | close/open 512 |
| SlideDishwasherRack | 502 | success frame 376, gripper close 126 |
| TurnOffStove | 500 | gripper close 500 |
| TurnOnElectricKettle | 520 | gripper close 383, success frame 137 |
| TurnOnMicrowave | 543 | gripper close 492, success frame 51 |
| TurnOnSinkFaucet | 506 | success frame 382, gripper close 124 |

"Success frame" means the demo never closed the gripper, so the point is the gripper position 17 frames before the end (demos end 17 frames after first success, seen in the replay test). Pushing and pressing tasks fit this. It is not yet checked against the true fixture part (`simlib.truth.target_part`); that check is the next step for these tasks.

## Check against the simulator (20 demos per task)
The gripper position computed from the state vector equals the simulator's `robot0_eef_pos` at that frame in every checked demo of every task (0.0 cm).

Grasp point to the object's body origin, pick-and-place style tasks:

| Task | Median cm | p95 cm | Max cm | Within 5 cm | Inside the object's bounding sphere |
|---|---|---|---|---|---|
| PickPlaceCounterToCabinet | 2.1 | 8.3 | 8.6 | 14/20 | 20/20 |
| PickPlaceCounterToStove | 2.4 | 5.1 | 6.8 | 19/20 | 20/20 |
| PickPlaceDrawerToCounter | 3.5 | 10.0 | 11.3 | 13/20 | 20/20 |
| PickPlaceSinkToCounter | 1.6 | 3.5 | 4.1 | 20/20 | 20/20 |
| PickPlaceToasterToCounter | 5.6 | 8.1 | 8.9 | 5/20 | 20/20 |
| CoffeeSetupMug | 5.0 | 6.3 | 6.7 | 10/20 | 20/20 |

The 5 cm test against the body origin fails for many demos, but every grasp point lies within the object's extent (bounding sphere of its geoms, a loose bound). The gap is the distance between where the hand grasps and the body origin (toast and mugs are grasped at an edge or handle). For F6 "perfect commands" at test time, the simulator gives the body origin, not the grasp point; this mismatch between training labels and test-time commands is up to about 10 cm and is a known issue. Training adds 2 cm noise to points.

TurnOnMicrowave also has an object named `obj` (a distractor, not the target), so its 50 cm distance is not meaningful.
