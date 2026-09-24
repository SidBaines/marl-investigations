#!/usr/bin/env bash
# Tinker RL smoke (M3): 4 runs x 3 steps, each max_usd=1.25 (total <= $5).
# Paid: Sid approved the M3 Tinker smoke on 2026-09-23.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"
set -a; source ~/.env; set +a
TASKS=$(uv run marli data build source=polaris_53k max_n=32 shuffle=true seed=0 --out "$OUT/tasks" | jq -r .manifest)
for run in coord_shared coord_perrole ms_all ms_last; do
  uv run marli train rl "$C/base.yaml" "$C/$run.yaml" tasks="$TASKS" max_usd=1.25 --out "$OUT/$run"
done
