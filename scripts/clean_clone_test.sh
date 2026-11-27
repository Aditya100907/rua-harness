#!/bin/bash
# Clean-clone test: does a fresh copy of the repo work with the existing 'dem' env?
#   bash scripts/clean_clone_test.sh               # branch overnight-nov26 of this checkout
#   BRANCH=main KEEP=1 bash scripts/clean_clone_test.sh
# Steps:
#   1. git clone this checkout (committed files only) into a temp folder
#   2. unit tests that need no simulator: pytest -m "not sim"
#   3. tiny training: 5 steps, batch 2, CPU, 2 OpenDrawer demos read straight from the mp4 files
#      (--video), so no frame store is needed. Uses the pretrained NeoBERT and DINOv3 weights from
#      the Hugging Face cache. Not --offline: an --offline checkpoint has no tokenizer, so eval.py
#      cannot load it ("no tokenizer available"). OFFLINE=1 trains with --offline and skips step 4.
#   4. tiny evaluation of that checkpoint: 1 episode, 16 steps, in the real simulator
# Needs: the dem env, OpenDrawer demos under RUA_DATA (or ~/apollo-lab/robocasa/datasets),
# RoboCasa kitchen assets, and NeoBERT plus the gated DINOv3 weights in the Hugging Face cache
# (or a Hugging Face login with DINOv3 access, so they can download).
# Writes nothing into this checkout. Prints CLEAN CLONE OK at the end if every step passed.
set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
BRANCH="${BRANCH:-overnight-nov26}"
PY="${PY:-$HOME/miniconda3/envs/dem/bin/python}"
DEVICE="${DEVICE:-cpu}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/rua_clean_clone.XXXXXX")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
[ "$(uname)" = Linux ] && export MUJOCO_GL="${MUJOCO_GL:-egl}"
FAIL=0
step() { printf '\n== %s\n' "$1"; }
check() { if [ "$1" -eq 0 ]; then echo "PASS $2"; else echo "FAIL $2 (exit $1)"; FAIL=1; fi; }

step "clone $SRC ($BRANCH) into $TMP"
git clone -q --branch "$BRANCH" "$SRC" "$TMP/rua-harness"; check $? "clone"
cd "$TMP/rua-harness" || exit 1
echo "commit $(git rev-parse --short HEAD)"
"$PY" -c "import sys, numpy, torch, mujoco; print(sys.version.split()[0], 'numpy', numpy.__version__, 'torch', torch.__version__, 'mujoco', mujoco.__version__)"

step "unit tests (not sim)"
t=$(date +%s)
PYTHONPATH=. "$PY" -m pytest -q -m "not sim" tests 2>&1 | tail -3
check "${PIPESTATUS[0]}" "pytest -m 'not sim' ($(( $(date +%s) - t )) s)"

step "tiny training ($DEVICE, 5 steps)"
echo '{"OpenDrawer": [0, 1]}' > "$TMP/episodes.json"
t=$(date +%s)
PYTHONPATH=. "$PY" -m policy.train --tasks OpenDrawer --steps 5 --batch 2 --workers 0 --device "$DEVICE" \
  ${OFFLINE:+--offline} --video --episodes-json "$TMP/episodes.json" --log-every 1 --save-every 5 --out "$TMP/train" \
  > "$TMP/train.log" 2>&1
rc=$?; tail -4 "$TMP/train.log"
check $rc "train 5 steps ($(( $(date +%s) - t )) s)"
CK="$TMP/train/ckpt_0000005.pt"
[ -f "$CK" ]; check $? "checkpoint written ($CK)"

if [ -n "${OFFLINE:-}" ]; then echo "OFFLINE=1: skipping evaluation"; else
step "tiny evaluation (1 episode, 16 steps)"
t=$(date +%s)
PYTHONPATH=. "$PY" -m policy.eval --ckpt "$CK" --tasks OpenDrawer --episodes 1 --horizon 16 --video 0 \
  --device "$DEVICE" --out "$TMP/eval.csv" --outdir "$TMP/eval_media" > "$TMP/eval.log" 2>&1
rc=$?; grep -v WARN "$TMP/eval.log" | tail -3
check $rc "eval 1 episode ($(( $(date +%s) - t )) s)"
[ "$(wc -l < "$TMP/eval.csv" 2>/dev/null || echo 0)" -ge 2 ]; check $? "eval CSV has a row"
head -2 "$TMP/eval.csv" 2>/dev/null
fi

if [ -z "${KEEP:-}" ]; then rm -rf "$TMP"; else echo "kept $TMP"; fi
[ $FAIL -eq 0 ] && echo "CLEAN CLONE OK" || { echo "CLEAN CLONE FAILED"; exit 1; }
