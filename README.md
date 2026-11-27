# rua-harness

LLM agent plus a small fast robot policy, on RoboCasa. APOLLO Lab, Yale. Adi Kothari and William Gu, advised by Xiatao Sun.

**Goal:** find the best way for an LLM agent to hand a physical job to a small policy, and show the pairing keeps the agent's success while cutting time and cost.

**This weekend (Nov 27 to 29):** three conditions on RoboCasa's 18 atomic-seen tasks: policy alone, agent alone, agent plus policy.

| Folder | What | Owner |
|---|---|---|
| `agent/` | HexaShell's tools on RoboCasa, LLM loop, episode runner | William |
| `policy/` | Demo data loader, DEM training, evaluation | Adi |
| `rua_harness/` | The command format between agent and policy, geometry helpers | Both |
| `CLAUDE.md` | Instructions for AI agents working in this repo | |
| `docs/` | `STATUS.md` (what is done, next, blocked) and `agent_port_spec.md` | |
| `results/` | One CSV per run | Both |
| `setup/` | `setup_dem.sh`, one-command install | |
| `tests/`, `scripts/` | Tests and utilities for `rua_harness/` | |

**Versions:** RoboCasa v1.0.1 at commit `425684d`, robosuite `5ce6643`, MuJoCo 3.3.1, numpy 2.2.5, Python 3.11.

**Tests:** `PYTHONPATH=. python -m pytest -q tests`
