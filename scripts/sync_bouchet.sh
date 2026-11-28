#!/bin/bash
# Send this repo's code to the Bouchet copy that jobs run from (no .git). Writes COMMIT so runs record the commit.
# From the Mac: rsync over ssh. On Bouchet (agent session in the git clone): local rsync into the same folder.
# SYNC_DRYRUN=1 lists what would change and copies nothing.
# If the jobs' folder is itself the git clone, there is nothing to copy: runs read the commit from git.
set -e
cd "$(dirname "$0")/.."
source scripts/on_bouchet.sh
DEST=/nfs/roberts/project/pi_dr877/ak3288/apollo-lab/rua-harness
if [ "$ON_BOUCHET" = 1 ] && [ "$(cd "$DEST" && pwd -P)" = "$(pwd -P)" ]; then
  echo "this checkout is the jobs' folder ($DEST); nothing to sync (commit $(git rev-parse --short HEAD))"
  exit 0
fi
git rev-parse --short HEAD > COMMIT
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "$(cat COMMIT)-dirty" > COMMIT; fi
if [ "$ON_BOUCHET" = 1 ]; then TARGET="$DEST/"; else TARGET="bouchet:$DEST/"; fi
rsync -a ${SYNC_DRYRUN:+-n -i} --exclude .git --exclude __pycache__ --exclude paper/build ./ "$TARGET"
echo "${SYNC_DRYRUN:+dry run, nothing copied: }synced $(cat COMMIT) to $TARGET"
