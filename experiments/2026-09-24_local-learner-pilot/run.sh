#!/usr/bin/env bash
# M4 local-learner smoke + pilot on one H100 pod (pod pre-approved by Sid, 2026-09-23).
# Runs ON THE POD from the checkout root; results are rsynced back before teardown.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"
# --no-sync: a plain `uv run` would re-sync .venv to uv.lock and drop the pod-train torch pins.
export HF_HOME=/workspace/hf TRITON_CACHE_DIR=/workspace/.triton/cache
# vLLM (own venv), co-located on GPU 0 at 0.5 utilisation.
SERVER=$(uv run --no-sync marli serve vllm model=qwen3_5_4b python=/workspace/venv-vllm/bin/python detach=true \
  max_model_len=32768 gpu_memory_utilization=0.5 max_lora_rank=32 max_loras=4 \
  learner_ranks='[32,32]' --out "$OUT/serve" | jq -r .manifest)
TASKS="$OUT/tasks/taskset.json"   # `marli data build source=polaris_53k max_n=256 shuffle=true seed=0`, built off-pod and copied
for run in smoke pilot; do
  uv run --no-sync marli train rl "$C/base.yaml" "$C/$run.yaml" tasks="$TASKS" max_usd=1 \
    local_server_json="$(dirname "$SERVER")/server.json" --out "$OUT/$run"
done
