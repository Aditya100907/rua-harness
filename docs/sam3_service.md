# SAM 3 service on Bouchet

HexaShell's `robot.find` and `robot.measure` need SAM 3 (Segment Anything Model 3) running as an HTTP
service. Without it, those two calls refuse and the rest of the robot API still works. Written Nov 27 2026.

## How HexaShell runs it (read from HexaShell `914d482`)

- Server: `services/sam3_server.py`. One model on one GPU, shared by all episodes on the machine. HTTP
  with JSON on **127.0.0.1 only** (default port 18300): `POST /image` (PNG, JPEG or raw RGB) returns an
  `image_id`; `POST /segment` with `text`, `points` or `box` returns masks (score, box, area, 1-bit PNG);
  `GET /health`. All GPU work runs one request at a time. Weights stored in bf16 (same masks as fp32).
- Launcher: `scripts/sam3_service.sh start GPU PORT` (`setsid nohup`, pid file under `logs/`,
  `HF_HUB_OFFLINE=1`). Settings: `SAM3_PYTHON`, `SAM3_CHECKPOINT_PATH` (default `checkpoints/sam3.pt`).
- Clients find it through `sam3_url` in `config/benches.yml` (default `http://127.0.0.1:18300`). So the
  service and the episodes must run on the same node.
- Packages (README "Setup" step 1 and `requirements.txt`): torch 2.7.0 and torchvision 0.22.0 with CUDA,
  `timm`, `sam3` 0.1.0 from https://github.com/facebookresearch/sam3, plus `einops`, `pycocotools`,
  `psutil` and `setuptools<81` (sam3 imports `pkg_resources` and does not declare the others).
- Weights: `sam3.pt` (3.45 GB) from the gated Hugging Face repo `facebook/sam3` (manual approval).

## What we have ready

| piece | where |
|---|---|
| env (Python 3.11, torch 2.7.0 cu128, torchvision 0.22.0, timm, sam3 at `31cb7e6`, einops, pycocotools, psutil, setuptools<81, numpy<2) | `/nfs/roberts/project/pi_dr877/ak3288/envs/sam3`, built by `setup/setup_hexashell_bouchet.sh sam3` (CPU job) |
| HexaShell checkout | `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/HexaShell` (`914d482`) |
| service job | `policy/slurm/sam3_service.sbatch`: downloads `sam3.pt` to `$SCR/models/sam3/` once, starts the server, then runs the smoke test (`MODE=smoke`, default, gpu_devel) or keeps serving (`MODE=serve`, writes node and port to `logs/sam3_service.addr`) |
| smoke test | `policy/sam3_smoke.py`: frame 0 of a task's left camera from our frame store (`$SCR/frames`), text prompts (`microwave`, `robot gripper`; `cabinet`, `counter`), prints mask count, scores, boxes, areas; saves PNGs under `logs/sam3_smoke_<job>/` |

## Status

**Blocked on access.** On Nov 27 17:00 the Hugging Face login on Bouchet (`~/.cache/huggingface/token`,
user `adikothar1`) could read the repo's metadata, but downloading any file gave `403 GatedRepoError:
Your request to access model facebook/sam3 is awaiting a review from the repo authors.` Nothing to do
on our side but wait. We did not request or accept anything.

## Steps once access lands

1. Check: `ssh bouchet "/nfs/roberts/project/pi_dr877/ak3288/envs/dem/bin/python -c \"from huggingface_hub import hf_hub_download as d; print(d('facebook/sam3','config.json'))\""` prints a path instead of a 403.
2. Make sure gpu_devel is free: `ssh bouchet "squeue -u ak3288 -p gpu_devel"` (1 job per user).
3. `ssh bouchet "cd /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness && sbatch policy/slurm/sam3_service.sbatch"`.
   The first run downloads 3.45 GB to scratch, then loads the model and runs the smoke test.
4. Read `logs/sam3_<job>.out`: expect `server up`, then for `microwave` at least one mask with a score
   above about 0.5 and a box on the microwave. Masks in `logs/sam3_smoke_<job>/`.
5. For episodes: run the service inside the same job as the HexaShell episodes (it listens on localhost
   only), e.g. start it in the background at the top of the episode sbatch as `sam3_service.sbatch` does.
   Scratch is purged after 30 days: copy `sam3.pt` to project space if we keep using it.
