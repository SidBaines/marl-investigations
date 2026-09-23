#!/usr/bin/env bash
set -euo pipefail

STUDY="$(cd "$(dirname "$0")" && basename "$PWD")"
cd "$(dirname "$0")/../.."
OUT="experiments/$STUDY/out"

# These verbs land in later milestones; this is a commented example only.
# Fill in model/protocol/seating configs before use. Every paid step needs
# an explicit max_usd budget and Sid's confirmation before launch.
# Chain verbs via the manifest in the single JSON line each prints:
# TASKS=$(uv run marli data build gsm8k split=test max_n=100 --out "$OUT/tasks" | jq -r .manifest)
# EPISODES=$(uv run marli eval rollout --tasks "$TASKS" max_usd=1 --out "$OUT/episodes" | jq -r .manifest)
