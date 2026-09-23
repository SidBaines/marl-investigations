#!/usr/bin/env bash
set -euo pipefail

# Works wherever this study dir lives (including nested sub-studies).
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"
CONFIGS="$HERE/configs"

# Example chain (the verbs land in later milestones). Every paid step needs an
# explicit max_usd budget and Sid's confirmation before launch. Each verb
# prints exactly one JSON line; chain on its "manifest" field:
#
# TASKS=$(uv run marli data build aime_2025 --out "$OUT/tasks" | jq -r .manifest)
# EPISODES=$(uv run marli eval rollout "$CONFIGS/rollout.yaml" --tasks "$TASKS" \
#     protocol=swarm seating.by_role.peer=<policy-ref> max_usd=1 --out "$OUT/episodes" | jq -r .manifest)
# uv run marli eval report --episodes "$EPISODES" --out "$OUT/report"
