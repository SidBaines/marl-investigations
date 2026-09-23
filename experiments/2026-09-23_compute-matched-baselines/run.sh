#!/usr/bin/env bash
# Compute-matched baselines (AIME 2025, Qwen3.5-4B on Tinker). Usage:
#   ./run.sh smoke    # 3 tasks x 7 cells, max_usd=1
#   ./run.sh pilot    # 30 tasks x 7 cells, max_usd=15
# Paid: Sid confirmed smoke ($1) and pilot ($15) on 2026-09-23.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"
STAGE="${1:?usage: run.sh smoke|pilot}"
set -a; source ~/.env; set +a   # TINKER_API_KEY (never printed)

TASKS=$(uv run marli data build source=aime_2025 --out "$OUT/tasks" | jq -r .manifest)
case "$STAGE" in
  smoke) EXTRA=(common.max_tasks=3); BUDGET=1 ;;
  pilot) EXTRA=(); BUDGET=15 ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
uv run marli eval grid "$HERE/configs/grid.yaml" common.tasks="$TASKS" "${EXTRA[@]}" \
  max_usd=$BUDGET parallel_cells=2 --out "$OUT/$STAGE"
