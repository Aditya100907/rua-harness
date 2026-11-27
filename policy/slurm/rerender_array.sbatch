#!/bin/bash
# CPU job array: re-render demos with depth (policy/rerender.py), F2.
# One array index = one chunk of CHUNK episodes of one task. 18 tasks x 12 chunks of 50 = 216
# indices (the biggest task has under 600 demos; empty chunks exit at once).
#   sbatch --array=0-215%60 policy/slurm/rerender_array.sbatch
#   TASKS="OpenDrawer" sbatch --array=0-11 policy/slurm/rerender_array.sbatch    # one task
# Output: $SCR/depth/<task>/episode_NNNNNN.npz plus rerender_log.jsonl. Episodes already on disk
# are skipped, so resubmitting the same array resumes. Afterwards run depth_spotcheck.sbatch.
#
# Rendering: day nodes have no GPU. MUJOCO_GL=egl (set in common.sh) still works there through
# Mesa's EGL (llvmpipe software renderer): checked on the login node, which also has no GPU
# (GL_RENDERER "llvmpipe (LLVM 19.1.7)", Mesa 24.2.8). OSMesa is NOT available (no libOSMesa in
# the env or the system), so do not switch to MUJOCO_GL=osmesa. The probe below logs the renderer.
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --job-name=rerender
#SBATCH --output=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/logs/rerender_%A_%a.out
source /nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/policy/slurm/common.sh
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RUA_DEPTH=$SCR/depth
CHUNK=${CHUNK:-50}
NCHUNK=${NCHUNK:-12}
CAMS=${CAMS:-"left right"}
T=(${TASKS:-$ALL_TASKS})
i=$SLURM_ARRAY_TASK_ID
TASK=${T[$((i / NCHUNK))]}
C=$((i % NCHUNK))
if [ -z "$TASK" ]; then echo "index $i: no task"; exit 0; fi
echo "index $i task $TASK chunk $C (episodes $((C * CHUNK))..$((C * CHUNK + CHUNK - 1))) cams $CAMS on $(hostname)"
python - <<'PY'
import time, mujoco
from OpenGL import GL
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="box" size="1 1 1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 256, 256); r.render()
print("GL renderer:", GL.glGetString(GL.GL_RENDERER).decode(), flush=True)
PY
# the first episode of every chunk is compared with the dataset video (RGB mean abs error)
python -m policy.rerender --task $TASK --chunk $C --chunk-size $CHUNK --cams $CAMS \
  --check-rgb-every $CHUNK --out $RUA_DEPTH
