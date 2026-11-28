# Open vision-language model as the commander (backlog 4)

An open vision-language model (VLM), served by vLLM on Bouchet at no cost, plays the agent: it looks at the
robot's camera images, writes one command, the trained policy runs it, and the model looks again. No Claude
CLI and no paid API. Written Sat Nov 28 2026.

## Pieces

| piece | where |
|---|---|
| env (Python 3.12, vLLM 0.11.0 with its own CUDA 12.8 torch) | `$PROJ/envs/vllm`, built by `setup/setup_vllm_bouchet.sh` as a `day` CPU job |
| weights | Hugging Face cache on scratch, `$SCR/cache/hf` (Qwen3-VL-8B-Instruct, Qwen2.5-VL-7B-Instruct) |
| server job | `policy/slurm/vlm_server.sbatch` (OpenAI-compatible API on `0.0.0.0:18400`, address file, idle guard) |
| agent loop | `policy/commander/open_model.py` |
| client job array | `policy/slurm/vlm_eval_array.sbatch` (one task per array index, one episode per process, `day` CPU nodes) |
| table | `scripts/vlm_summary.py` |
| unit tests | `tests/test_open_model.py` |

## Model

`Qwen/Qwen3-VL-8B-Instruct` (bf16, about 17 GB, fits one gpu_devel GPU). It is trained for pointing and
answers points in relative coordinates 0 to 1000, which do not depend on the image size. Fallback:
`Qwen/Qwen2.5-VL-7B-Instruct` (absolute pixels of the image it gets; run the client with `--coords pixel
--img-size 448`). Reason in `docs/decisions_log.md`. Sampling is greedy (temperature 0, seed 0).

## The loop (`open_model.py`)

1. Build the episode with `policy.commander.server.Episode` (same env, seed, policy load order and
   grading as the Claude commander). Seeds 1000 + i, as `policy/eval.py`.
2. Send the system prompt, the job (RoboCasa's instruction), the robot time, and the left, right and wrist
   images (256 x 256, upscaled to 512 before sending). vLLM's structured output forces one JSON object:
   `{"thought", "action": "run_skill" | "done", "command": {skill, object {name, camera, uv}, target, source,
   direction, max_seconds}}`.
3. Validate: unknown skill, missing required field or a point outside the image goes back to the model as
   "Command refused: ..." and costs a call. Fields the skill does not take are dropped.
4. Lift each point to 3D with that camera's depth (`Episode._lift`), run `run_skill` (the harness converts
   the command to the checkpoint's format: `to_sentence`, `to_policy_text`, drawn points or 3D points).
   `max_seconds` defaults to all the remaining robot time.
5. Send the new images (only the last 2 image sets stay in the context). Stop when the model says done,
   robot time runs out, or after 6 model calls.

The model always gives names and pixels, whatever the checkpoint's format, so the commander is the same for
every format and its pointing error is measured for every format (the sentence and kv_text policies ignore
the points). As in the Claude commander, a skill stops early when the simulator's success check fires; the
model is never told why it stopped.

Logged: every model call (`calls.jsonl`: prompt and completion tokens, seconds, reply) and `final.json`
(commands, sent text, grading-only point errors) in `runs/vlm_commander/<note>/<task>_<seed>/`. CSV columns:
`results/README.md` plus format, checkpoint, robot_seconds, n_skill, grasp_err_cm and place_err_cm (first
command; every command in `*_all`), llm_seconds, agent_status, model.

Formats: sentence, kv_text, points_2d, points_3d_rgb, points_3d. **pose_path is skipped**: it needs a timed
path of future gripper poses, which the model cannot produce from images.

## How to run

On Bouchet, from `$PROJ/apollo-lab/rua-harness`, after `squeue -u ak3288` shows gpu_devel free:

```
S=$(sbatch --parsable -p gpu_devel --time=01:30:00 policy/slurm/vlm_server.sbatch)
for f in sentence points_2d; do
  CKPT=$PROJ/checkpoints/fs_mse_$f/ckpt_0020000.pt TASKS="PickPlaceCounterToCabinet OpenDrawer TurnOnMicrowave" \
    EPISODES=10 sbatch --array=0-2 --dependency=after:$S policy/slurm/vlm_eval_array.sbatch
done
python scripts/vlm_summary.py --formats sentence points_2d --episodes 10     # after syncing results
```

The server stops itself 20 minutes after its last request (40 minutes if none ever arrives), so it never sits
idle for YCRC's 1-hour limit. Clients wait up to 60 minutes for `runs/vlm_server.txt`.

## Results
