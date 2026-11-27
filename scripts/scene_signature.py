"""Print a signature of the scenes eval builds (layout, style, objects, positions) for given seeds.
Run it twice in separate processes and diff, with RUA_DETERMINISM_FIX=1 or 0.
    python scripts/scene_signature.py OpenDrawer 1000 1001 1002
"""
import json
import sys

import numpy as np

from policy import determinism, paths, sim

task, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
env = sim.make_env(paths.task_dir(task), cams=None)
out = {"fix": determinism.enabled()}
for s in seeds:
    env.rng = np.random.default_rng(s)
    obs = env.reset()
    m = env.get_ep_meta()
    objs = {n: np.round(env.sim.data.body_xpos[i], 4).tolist() for n, i in sorted(env.obj_body_id.items())}
    out[str(s)] = {"layout": str(m.get("layout_id")), "style": str(m.get("style_id")), "lang": m.get("lang"),
              "base": np.round(obs["robot0_base_pos"], 4).tolist(), "objs": objs,
              "fixture_refs": str(m.get("fixture_refs"))}
print("SIG", json.dumps(out, sort_keys=True))
