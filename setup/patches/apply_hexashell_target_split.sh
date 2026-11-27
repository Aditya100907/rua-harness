#!/bin/bash
# Apply rua-harness's `target` split to a HexaShell checkout at commit 914d482.
#
#   bash setup/patches/apply_hexashell_target_split.sh [HEXASHELL_DIR]   (default ~/apollo-lab/HexaShell)
#
# Idempotent: does nothing if the patch is already in. What the patch changes: docs/hexashell_changes.md.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH="$HERE/hexashell_target_split.patch"
HX=${1:-$HOME/apollo-lab/HexaShell}
cd "$HX"
base=$(git rev-parse --short=7 HEAD)
[ "$base" = "914d482" ] || echo "warning: HexaShell is at $base, the patch was made against 914d482" >&2
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "already applied in $HX"
    exit 0
fi
git apply --check "$PATCH"
git apply "$PATCH"
echo "applied to $HX ($base): $(git diff --stat -- hexashell | tail -1)"
