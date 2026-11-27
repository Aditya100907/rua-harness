# Test sets and the scripted baseline

Written Nov 27, 2026. Code: `policy/testsets.py` (test sets), `policy/scripted.py` (baseline).
Tests: `tests/test_testsets.py` (no simulator needed).

## Scripted baseline (`policy/scripted.py`)

A hand-written controller with no learning. It reads the simulator's ground truth: the object
position, the cabinet interior, the drawer handle and slide joint, the microwave start button.
So it is the **"perfect command" oracle**: what a simple controller does when the target is
known exactly. It is not an agent result and it never sees a camera. This is the one place
outside grading where ground truth drives actions, and every CSV row says `format = scripted`.

Control is proportional on the end-effector pose in the robot base frame. The arm controller
is OSC_POSE with delta inputs in the base frame; input [-1, 1] means 5 cm and 0.5 rad per step.
The mobile base and torso never move (2 to 5 of every 60 demos move them). The plan is made
once, at the first step, as a list of phases (move to a pose, open or close, wait).

| task | plan |
|---|---|
| PickPlaceCounterToCabinet | gripper points toward the cabinet, tilted 45 degrees down, fingers closing sideways; descend onto the object from 10 cm above, close, lift 5 cm, back away from the counter, level the gripper while rising to shelf height, slide 15 cm into the lowest shelf, open, back out 35 cm |
| OpenDrawer | gripper points at the drawer face with fingers closing vertically (as in the demos); go in front of the handle, move in 1 cm past its centre, close, pull along the slide joint by 0.55 x drawer depth + 4 cm |
| TurnOnMicrowave | close the gripper, point it at the button along the button's face normal, push to 2 cm past the face, stop when the microwave turns on, back off 22 cm (success needs 15 cm) |

Same seeds and env construction as `policy/eval.py`: `make_env` from the dataset's env kwargs,
then `env.rng = np.random.default_rng(seed0 + i); env.reset()`, success from
`env._check_success()` each step, horizon from the dataset registry. One difference: the env is
built without cameras unless `--video` is set. Checked: seeds 1000 to 1002 give the same
layout, style, object and object position with and without cameras.

    python -m policy.scripted --tasks OpenDrawer --episodes 50 --seed0 1000 --note smoke
    python -m policy.scripted --testset lookalike --tasks PickPlaceCounterToCabinet --episodes 20

Output: `results/<date>_scripted[_<testset>]_<note>.csv`, the `policy/eval.py` columns plus
`testset`, `went_for` (which object the plan went for) and `phases` (each phase, its step
count, and `ok` or `timeout`).

RESULTS_PLACEHOLDER

## The test sets (`policy/testsets.py`)

`testsets.make_env(name, task)` builds the env; `testsets.instruction(name, task, env, seed)`
gives the instruction. Everything else (controller, object instance split, clutter) is the
dataset's own env kwargs, as in `policy.sim.make_env`.

### lookalike: look-alike distractors

How RoboCasa picks objects: each task's `_get_obj_cfgs()` returns a list of object configs
(`name`, `obj_groups`, `placement`). `Kitchen._create_objects()` samples each one in order
with `EnvUtils.create_obj`, which calls `sample_kitchen_object(groups, split=...)` with the
env rng. For PickPlaceCounterToCabinet the configs are `obj` (groups "all", graspable, a
0.60 x 0.30 m region of the counter in front of the cabinet), `distr_counter` (any object,
further back on the counter) and `distr_cab` (any object, back of the cabinet).

The set adds a subclass, `LookalikePickPlaceCounterToCabinet` (robosuite registers it by
class name). Its `_get_obj_cfgs()` first draws the target's category with the env rng and the
same rules as the target (graspable, target instance split), sets `obj` to that category, and
inserts a `lookalike` config with the same category and the same placement region. The two
objects may be the same instance or two instances of the category. The instruction is the
normal one ("Pick the onion from the counter ..."), so it fits both objects; only a point (or
a pose) on the target says which one. Success is still about `obj`.

Checked on seeds 1000 to 1003: both objects share a category every time (cucumber, onion,
bread, turmeric), 0.08 to 0.39 m apart.

Two scripted numbers: going for the true target (should match the plain split), and
`--guess-lookalike`, which goes for the target or the twin at random (seeded), which is what a
command carrying only the category name allows. Episodes that go for the twin can only
succeed by accident.

### kitchens_recombined and kitchens_pretrain: held-out kitchens

RoboCasa has 60 layouts and 60 styles (`robocasa/models/scenes/scene_registry.py`). Layouts and
styles 1 to 10 are the "test" group, which the target split uses, and only as the 10 pairs
(1, 1) ... (10, 10) (`layout_and_style_ids` in every target dataset's env kwargs). Layouts and
styles 11 to 60 are the "train" group, used by the pretrain split. To evaluate elsewhere, pass
a different `layout_and_style_ids` to the env; nothing else changes.

- `kitchens_recombined`: the 90 pairs (l, s), l != s, l and s in 1..10. Known rooms and known
  textures, never in that combination.
- `kitchens_pretrain`: the 25 pairs RoboCasa uses for its own 5x5 evaluation
  (`EnvUtils.KITCHEN_SCENES_5X5`: layouts 11, 15, 18, 40, 50 x styles 14, 28, 34, 46, 58).
  Never in the target split. Objects stay on the target instance split.

Note: the pretrain split's demos use these kitchens, so a policy trained only on target demos
has never seen them; one fine-tuned from a pretrain checkpoint has.

### unseen_names: unseen object names and phrasings

The scene is the normal target scene; only the instruction changes. For
PickPlaceCounterToCabinet the category name in "Pick the X from the counter ..." is swapped for
a synonym from `UNSEEN_NAMES` (one per category, all 105 categories seen in the 500 training
instructions). For OpenDrawer and TurnOnMicrowave, which have only 2 and 1 training sentences,
the whole sentence is swapped (`UNSEEN_PHRASES`). The choice is seeded per episode.
`tests/test_testsets.py` checks against `meta/tasks.jsonl` that no synonym phrase occurs in any
training instruction and that no synonym is itself a category name. Single words inside a
synonym can be seen words ("espresso cup" contains "cup"); the phrase as a whole is new.

| task | training sentence | unseen versions |
|---|---|---|
| OpenDrawer | Open the right drawer. | Pull out the drawer on your right. / Slide the right-hand drawer open. |
| OpenDrawer | Open the left drawer. | Pull out the drawer on your left. / Slide the left-hand drawer open. |
| TurnOnMicrowave | Press the start button on the microwave. | Start the microwave oven. / Hit the begin key on the microwave oven. |

Object categories (PickPlaceCounterToCabinet), a sample; the full table is
`testsets.UNSEEN_NAMES`:

| category | unseen name | category | unseen name |
|---|---|---|---|
| apple | red fruit | eggplant | aubergine |
| avocado | alligator pear | hot dog | frankfurter |
| baguette | french loaf | ketchup | catsup |
| bell pepper | capsicum | mug | beaker |
| bread | loaf | onion | shallot |
| can | tin | potato | spud |
| coffee cup | espresso cup | shrimp | prawn |
| colander | pasta drainer | strainer | sieve |
| cucumber | gherkin | sweet potato | yam |
| cup | tumbler | tangerine | clementine |

The scripted baseline ignores language, so its unseen_names number only checks that the set
runs; it should match the plain split on the same seeds.
