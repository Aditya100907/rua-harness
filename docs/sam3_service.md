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

## What runs now (Nov 27 2026, 17:05)

**Service up.** Job **34895235** on `gpu_devel`, node **`a1125u42n01`** (NVIDIA A40), URL
**`http://a1125u42n01:18300`**. Started 17:04:49, ends at its time limit **23:04:49** (6 h, the
gpu_devel maximum). The current address is always in
`/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/runs/sam3_service.txt` (line 1 `host:port`, then
`url`, `job`, `ends`); the job deletes the file when it stops, so a missing file means no service.

**How it is reached.** HexaShell's server listens on `127.0.0.1:18299` only. The job runs
`policy/tcp_forward.py` next to it on `0.0.0.0:18300`, which passes each connection through unchanged.
Checked: `curl http://a1125u42n01:18300/health` answers from the login node and from another compute
node (`c1102u05n01`, a `day` job); the inner port 18299 does not answer from outside. There is no
authentication: anyone on the cluster network can send images. HexaShell runs point at it with
`sam3_url: http://<host>:18300` (read the host from `runs/sam3_service.txt` at episode start).

**Smoke test (passed, in the same job).** Frame 0 of the left camera, 256 x 256, from our frame store.
Model load 32 s, GPU memory 2.9 GB, about 57 ms per text prompt.

| task / prompt | masks | best score | best box [x0, y0, x1, y1] | best area |
|---|---|---|---|---|
| TurnOnMicrowave / `microwave` | 8 | 0.824 | [80, 0, 255, 84] | 19.7% |
| TurnOnMicrowave / `robot gripper` | 8 | 0.004 | (nothing real; the gripper may be out of view) | |
| PickPlaceCounterToCabinet / `cabinet` | 8 | 0.789 | [75, 186, 228, 255] | 7.0% |
| PickPlaceCounterToCabinet / `counter` | 8 | 0.750 | [71, 112, 255, 207] | 11.4% |

Mask PNGs for the microwave frame: `logs/sam3_smoke_34895235/`. Server log:
`logs/sam3_server_34895235.log`; job log `logs/sam3_34895235.out`. I did not look at the mask images by eye.

**Restart / longer runs.** Check `squeue -u ak3288` first (gpu_devel: 1 job per user), then from the
repo on Bouchet:
- 6 h on gpu_devel: `sbatch -p gpu_devel --time=06:00:00 policy/slurm/sam3_service.sbatch`
- longer on `gpu` (default in the script, 6 h; up to 2 days): `sbatch --time=1-00:00:00 policy/slurm/sam3_service.sbatch`
- smoke test only: `sbatch -p gpu_devel --time=01:00:00 --export=ALL,MODE=smoke policy/slurm/sam3_service.sbatch`
- stop: `scancel <job>`. The node changes on each start; clients must reread `runs/sam3_service.txt`.

## Pieces

| piece | where |
|---|---|
| env (Python 3.11, torch 2.7.0+cu128, torchvision 0.22.0, timm 1.0.31, sam3 at `31cb7e6`, einops, pycocotools, psutil, setuptools<81, numpy<2) | `/nfs/roberts/project/pi_dr877/ak3288/envs/sam3`, built by `setup/setup_hexashell_bouchet.sh sam3` (CPU job 34894784) |
| weights | `sam3.pt` (3.45 GB) at `/nfs/roberts/scratch/pi_dr877/ak3288/models/sam3/` (scratch: purged after 30 days unused; copy to project space if we keep it) |
| HexaShell checkout | `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/HexaShell` (`914d482`), server file `services/sam3_server.py` unchanged |
| service job | `policy/slurm/sam3_service.sbatch` (downloads weights once, server, forwarder, smoke test, address file) |
| forwarder | `policy/tcp_forward.py` (stdlib asyncio) |
| smoke test | `policy/sam3_smoke.py` |

## Access history

Nov 27 ~16:55: the Hugging Face login on Bouchet (user `adikothar1`) got `403 GatedRepoError: awaiting a
review`. Adi reported access granted shortly after; at 17:00 a test download of `config.json` worked.
We did not request access or accept terms ourselves.

## Run it on demand, not idle (Nov 27, 17:50)
YCRC cancels GPU jobs that show 0% GPU use (on gpu_devel after about 1 hour), and warned on job 34895235 at 17:45 because nothing was calling the service yet. So the service was stopped at 17:50 and must not be left running idle (and must not be kept "busy" with fake requests; that works around the cluster's policy).

How to use it: start the service in the same batch as the HexaShell runs that need it, and stop it when they finish.
1. `sbatch -p gpu_devel --time=06:00:00 policy/slurm/sam3_service.sbatch` (or on `gpu` for a longer batch), from the repo on Bouchet, right before submitting the HexaShell episodes.
2. Wait until `/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/runs/sam3_service.txt` exists (about 1 minute; the model loads in about 32 s), then point HexaShell's `sam3_url` at the URL in it.
3. Submit the HexaShell runs with `--dependency=after:<service job id>` so they start once the service is up, and `scancel` the service when the last one ends (or give the service job a `--time` that matches the batch).
