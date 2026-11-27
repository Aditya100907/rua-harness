#!/bin/bash
# Send this repo to Bouchet (code only; no .git). Writes COMMIT so runs record the commit.
set -e
cd "$(dirname "$0")/.."
git rev-parse --short HEAD > COMMIT
[ -z "$(git status --porcelain --untracked-files=no)" ] || sed -i '' 's/$/-dirty/' COMMIT
rsync -a --exclude .git --exclude __pycache__ --exclude paper/build ./ bouchet:/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness/
echo "synced $(cat COMMIT)"
