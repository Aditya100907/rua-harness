"""Held-out test sets for the command-format comparison. See docs/testsets.md.

Each set is a way to build the environment (and, for one set, the instruction) for a task.
Nothing here changes the training data; these sets are for evaluation only.

  lookalike            PickPlaceCounterToCabinet with a second object of the SAME category
                       placed in the same counter region as the target. The sentence names a
                       category, so it cannot tell the two apart; a point on the target can.
  kitchens_recombined  target layouts 1..10 with a different target style (l, s), s != l.
                       The target split only ever pairs layout i with style i.
  kitchens_pretrain    the 25 pretrain kitchens RoboCasa itself uses (KITCHEN_SCENES_5X5:
                       layouts 11, 15, 18, 40, 50 x styles 14, 28, 34, 46, 58). Never in the
                       target split. Objects stay on the target instance split.
  unseen_names         the normal target scenes, with the instruction rewritten to use words
                       that never appear in the training instructions (UNSEEN_NAMES).

    from policy import testsets
    env = testsets.make_env("lookalike", "PickPlaceCounterToCabinet")
    env.rng = np.random.default_rng(seed); obs = env.reset()
    instr = testsets.instruction("unseen_names", "OpenDrawer", env, seed)
"""
from __future__ import annotations

import copy

import numpy as np

from policy import paths, sim

SETS = ["lookalike", "kitchens_recombined", "kitchens_pretrain", "unseen_names"]

KITCHENS_RECOMBINED = [(l, s) for l in range(1, 11) for s in range(1, 11) if l != s]   # 90 pairs
KITCHENS_PRETRAIN = [(l, s) for l in [11, 15, 18, 40, 50] for s in [14, 28, 34, 46, 58]]  # 25 pairs


# ------------------------------------------------------------------ look-alike distractors

def _lookalike_class():
    """Defined lazily so importing this module does not import robocasa. Subclassing a
    robosuite env registers it by class name, so robosuite.make(env_name=...) finds it."""
    global LookalikePickPlaceCounterToCabinet
    if "LookalikePickPlaceCounterToCabinet" in globals():
        return LookalikePickPlaceCounterToCabinet
    import robocasa  # noqa: F401
    from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToCabinet

    class LookalikePickPlaceCounterToCabinet(PickPlaceCounterToCabinet):
        """PickPlaceCounterToCabinet plus `lookalike`: same category as `obj`, same counter
        region. The category is drawn first (with the env rng, same rules as the target:
        graspable, target instance split), then both objects are sampled from it."""

        def _get_obj_cfgs(self):
            cfgs = super()._get_obj_cfgs()
            obj = next(c for c in cfgs if c["name"] == "obj")
            _, info = self.sample_object(obj["obj_groups"], exclude_groups=obj.get("exclude_obj_groups"),
                                         graspable=True)
            cat = info["cat"]
            obj["obj_groups"] = cat
            obj["exclude_obj_groups"] = None
            twin = dict(name="lookalike", obj_groups=cat, graspable=True,
                        placement=copy.deepcopy(obj["placement"]))
            cfgs.insert(1, twin)
            self.lookalike_cat = cat
            return cfgs

    globals()["LookalikePickPlaceCounterToCabinet"] = LookalikePickPlaceCounterToCabinet
    return LookalikePickPlaceCounterToCabinet


# ------------------------------------------------------------------ unseen names

# Words for each object category in the smoke tasks that never occur in the 500 training
# instructions per task (checked by tests/test_testsets.py against meta/tasks.jsonl).
# Category names are the RoboCasa category with "_" -> " ", as get_obj_lang prints them.
UNSEEN_NAMES: dict[str, list[str]] = {
    "apple": ["red fruit"], "avocado": ["alligator pear"], "bagel": ["ring roll"],
    "baguette": ["french loaf", "long loaf"], "banana": ["yellow fruit"], "bar": ["snack stick"],
    "bar soap": ["soap block"], "beer": ["lager"], "bell pepper": ["capsicum"], "bowl": ["basin dish"],
    "boxed drink": ["juice box"], "boxed food": ["food carton"], "bread": ["loaf"],
    "broccoli": ["green florets"], "cake": ["gateau"], "can": ["tin"], "candle": ["wax light"],
    "canned food": ["tinned food"], "carrot": ["orange root"], "cereal": ["breakfast box"],
    "cheese": ["cheddar block"], "cheese grater": ["shredder"], "chicken drumstick": ["poultry leg"],
    "cinnamon": ["spice stick"], "coffee cup": ["espresso cup"], "colander": ["pasta drainer"],
    "condiment bottle": ["sauce bottle"], "cookie dough ball": ["dough lump"], "corn": ["maize cob"],
    "cream cheese stick": ["dairy stick"], "croissant": ["crescent pastry"], "cucumber": ["gherkin"],
    "cup": ["tumbler"], "cupcake": ["fairy cake"], "dish brush": ["scrub brush"], "donut": ["doughnut"],
    "egg": ["hen's egg"], "eggplant": ["aubergine"], "fish": ["fillet"], "garlic": ["clove bulb"],
    "glass cup": ["drinking glass"], "honey bottle": ["nectar squeeze bottle"], "hot dog": ["frankfurter"],
    "hotdog bun": ["sausage roll bun"], "ice cube": ["frozen cube"], "jam": ["preserves"],
    "jar": ["mason pot"], "jug": ["carafe"], "jug wide opening": ["wide carafe"], "juice": ["fruit drink"],
    "kebab skewer": ["brochette"], "kettle": ["boiler"], "ketchup": ["catsup"], "kiwi": ["kiwifruit"],
    "ladle": ["soup scoop"], "lemon": ["citron"], "lemon wedge": ["citrus slice"], "lime": ["green citrus"],
    "liquor": ["spirits"], "mango": ["stone fruit"], "marshmallow": ["puffed sweet"],
    "mayonnaise": ["aioli"], "measuring cup": ["graduated beaker"], "milk": ["dairy carton"],
    "mug": ["beaker"], "mushroom": ["fungus"], "mustard": ["dijon"], "oil and vinegar bottle": ["cruet"],
    "onion": ["shallot"], "orange": ["navel fruit"], "paprika": ["chili powder"], "peach": ["nectarine"],
    "pear": ["bartlett"], "peeler": ["paring tool"], "pitcher": ["ewer"], "pizza cutter": ["pizza wheel"],
    "potato": ["spud"], "reamer": ["citrus press"], "rolling pin": ["dough roller"],
    "salt and pepper shaker": ["seasoning set"], "shaker": ["sprinkler"], "shrimp": ["prawn"],
    "soap dispenser": ["soap pump"], "sponge": ["scourer"], "spray": ["spritzer"],
    "squash": ["gourd"], "steak": ["beef cut"], "strainer": ["sieve"], "straw": ["sipper"],
    "sugar cube": ["sweet cube"], "sweet potato": ["yam"], "syrup bottle": ["maple flask"],
    "tangerine": ["clementine"], "teapot": ["brew pot"], "tomato": ["love apple"], "tongs": ["grippers"],
    "tupperware": ["food container"], "turkey slice": ["deli meat"], "turmeric": ["yellow spice"],
    "water bottle": ["flask"], "bottled water": ["mineral water"], "bottled drink": ["soda bottle"],
    "whisk": ["egg beater"], "wine": ["vino"], "wooden spoon": ["stirring stick"], "yogurt": ["curd cup"],
}

# Whole-instruction rewrites for the fixture tasks (no category noun to swap).
UNSEEN_PHRASES: dict[str, dict[str, list[str]]] = {
    "OpenDrawer": {
        "Open the right drawer.": ["Pull out the drawer on your right.", "Slide the right-hand drawer open."],
        "Open the left drawer.": ["Pull out the drawer on your left.", "Slide the left-hand drawer open."],
    },
    "TurnOnMicrowave": {
        "Press the start button on the microwave.": ["Start the microwave oven.",
                                                     "Hit the begin key on the microwave oven."],
    },
}


def instruction(name: str, task: str, env, seed: int) -> str:
    """The instruction an episode of test set `name` uses (the env's own unless unseen_names)."""
    lang = env.get_ep_meta().get("lang", "")
    if name != "unseen_names":
        return lang
    rng = np.random.default_rng(seed + 7919)
    if task in UNSEEN_PHRASES:
        opts = UNSEEN_PHRASES[task].get(lang)
        return str(rng.choice(opts)) if opts else lang
    cat = env.get_obj_lang("obj") if hasattr(env, "get_obj_lang") else ""
    if cat in UNSEEN_NAMES and cat in lang:
        return lang.replace(cat, str(rng.choice(UNSEEN_NAMES[cat])), 1)
    return lang


def make_env(name: str, task: str, cams=None, size: int = 256):
    """robosuite env for test set `name`, otherwise identical to policy.sim.make_env (dataset
    env kwargs: controller, target object split, clutter)."""
    import robosuite
    if name not in SETS:
        raise KeyError(f"unknown test set {name!r}; choose from {SETS}")
    meta = sim.dataset_env_meta(paths.task_dir(task))
    kw = copy.deepcopy(meta["env_kwargs"])
    kw["env_name"] = meta["env_name"]
    if name == "lookalike":
        if task != "PickPlaceCounterToCabinet":
            raise ValueError("lookalike is built for PickPlaceCounterToCabinet only")
        kw["env_name"] = _lookalike_class().__name__
    elif name == "kitchens_recombined":
        kw["layout_and_style_ids"] = KITCHENS_RECOMBINED
    elif name == "kitchens_pretrain":
        kw["layout_and_style_ids"] = KITCHENS_PRETRAIN
    kw.update(has_renderer=False, ignore_done=True, use_object_obs=True, seed=None)
    if cams:
        kw.update(has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=[sim.CAMERAS.get(c, c) for c in cams],
                  camera_widths=size, camera_heights=size, camera_depths=False)
    else:
        kw.update(has_offscreen_renderer=False, use_camera_obs=False)
    return robosuite.make(**kw)
