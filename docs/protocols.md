# Evaluation protocols: ours, DEM's, RoboCasa365's

Sources: RoboCasa docs in our pinned checkout (`robocasa/docs/benchmarking/*.md`, read Nov 27), `robocasa/utils/dataset_registry.py`, HexaShell's README (read Nov 27, not run), `docs/related_notes.md`. DEM's protocol details marked "assumed" are not confirmed from the DEM paper text.

| | Ours (tonight) | DEM paper (18 tasks) | RoboCasa365 Target50 (RoboCasa docs) | HexaShell's RoboCasa365 runs |
|---|---|---|---|---|
| Tasks | 18 atomic-seen; 3 smoke tasks first | 18 atomic-seen | 50: 18 atomic-seen, 16 composite-seen, 16 composite-unseen | 50 |
| Split | `target`: 10 kitchen scenes (layout, style 1..10), target objects | target (assumed) | `target` for target-task benchmarks | `pretrain` (layouts 11 to 60) for their official-protocol runs; their headline 85.6% is "pretrain50, test seeds 20 to 49" |
| Episodes | 50 per task (100 for final checkpoints), seeds 1000 + i | 50 per task (assumed, RoboCasa default) | 50 sampled scenarios per task | 50 per task, seeds 1000 to 1049 (official); 30 per task in the headline |
| Horizon | `ATOMIC_TASK_DATASETS[task]["horizon"]` at RoboCasa 425684d (v1.0.1, horizons 1.5x the v1.0 values) | unknown which version | v1.0.1 horizons | task's official horizon, plus a cap of 40 model replies (atomic) |
| Success | `env._check_success()` after every step; episode ends at first success | same (assumed) | binary success per rollout | same |
| Training data | target human demos, 500+ per task (100%) | same | target data, 10/30/100% variants | none (agent) |
| Cameras | left + right agent view at 256 (run 5 tests left + wrist) | 2 cameras at 256 | 3 cameras available | left, right, wrist RGB-D at 512 |
| Reference numbers | none yet | DEM 55.6% (18 tasks) | GR00T N1.5 target-only (100% data): atomic-seen 50.6%, composite-seen 22.7%, composite-unseen 27.5%; with pretraining: 68.5 / 40.6 / 42.1 | 59.3% without skills, 85.6% with skills (50 tasks) |

## Differences that matter
- **Split:** HexaShell's headline numbers use the pretrain split; ours use target. Not comparable without rerunning.
- **Horizons:** RoboCasa v1.0.1 lengthened every horizon by 1.5x. If DEM was evaluated with v1.0 horizons, our numbers get extra time. Check which version the DEM paper used.
- **Composite tasks:** our policy is trained only on atomic tasks; composite tasks are evaluated by chaining atomic commands (`policy/composite.py`, Part Q), not by a policy trained on composite data. Not comparable with GR00T's composite numbers.

## Target50 evaluation (set up, not launched at scale)
- Atomic-seen: `policy/eval.py --tasks <18 tasks> --episodes 50` with the run-2 checkpoint (the watcher does this every 50k steps).
- Composite-seen and composite-unseen: `policy/composite.py --tasks <task> --episodes 50` once the target composite datasets are on Bouchet (download pending). The chain needs each demo's subtask labels, so composite evaluation replays demo scenes rather than fresh seeds; this differs from the official protocol, which samples fresh scenes.
