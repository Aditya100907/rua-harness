"""HexaShell's RoboCasa determinism fix, applied at runtime (Nov 27, Adi's request).

HexaShell (THIRD_PARTY_VERSIONS.md, `[roboshell-determinism]`) changes one line of
robocasa/models/fixtures/counter.py, in `Counter.get_reset_regions`:
    valid_geoms = list(set(valid_geoms))   ->   valid_geoms = list(dict.fromkeys(valid_geoms))
A set of XML elements iterates in memory-address order, so the sampled counter regions, and with
them the scene, can differ between two resets with the same seed.

We do not edit the pinned RoboCasa checkout. `apply()` recompiles that one method with the line
swapped and installs it on the class. It is on by default (policy.sim.make_env calls it); set
RUA_DETERMINISM_FIX=0 or pass `--no-determinism-fix` to policy/eval.py to run without it.
"""
from __future__ import annotations

import inspect
import os
import textwrap

OLD = "valid_geoms = list(set(valid_geoms))"
NEW = "valid_geoms = list(dict.fromkeys(valid_geoms))"
_applied = False


def enabled() -> bool:
    return os.environ.get("RUA_DETERMINISM_FIX", "1") != "0"


def apply() -> bool:
    """Install the fix once per process. Returns True if the fix is active."""
    global _applied
    if _applied or not enabled():
        return _applied
    import robocasa.models.fixtures.counter as cmod
    fn = cmod.Counter.get_reset_regions
    src = textwrap.dedent(inspect.getsource(fn))
    if src.count(OLD) != 1:
        raise RuntimeError(f"determinism fix: expected the line once in Counter.get_reset_regions, found {src.count(OLD)}")
    ns: dict = {}
    exec(compile(src.replace(OLD, NEW), inspect.getsourcefile(fn), "exec"), cmod.__dict__, ns)
    cmod.Counter.get_reset_regions = ns["get_reset_regions"]
    _applied = True
    return True
