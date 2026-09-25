#!/usr/bin/env bash
# Sacrifice relay (signs of life) on one 2xH200 pod: vLLM on GPU 0, the learner on GPU 1.
# Runs ON THE POD from the checkout root. Phases (run one at a time; see README.md):
#   ./run.sh serve [overrides] | filter [K] | repos | gate | train <solo|relay_individual|relay_team>
# `filter K` samples only the first K candidate problems and pauses; a later `filter` (same out dir,
# any pod) resumes the rest without re-sampling them, then runs `data filter`.
# `out/candidates` (800 DeepCoder train problems, `marli data build source=deepcoder max_n=800
# shuffle=true seed=0`) is built off-pod and copied in. Results are rsynced back before teardown.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"
# --no-sync: a plain `uv run` would re-sync .venv to uv.lock and drop the pod-train torch pins.
M="uv run --no-sync marli"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
SERVER="$OUT/serve/server.json"
POLICY="policies.q.ref=vllm:@$SERVER#qwen3_8_27b"
EVAL() { $M eval rollout "$C/eval_common.yaml" "$C/$1.yaml" "${@:2}" "$POLICY" max_usd=1; }

case "${1:?phase}" in
serve)  # vLLM on GPU 0 only, most of its memory for KV cache. Extra args override (see bench/).
  HF_HUB_OFFLINE=0 $M serve vllm model=qwen3_8_27b python=/opt/vllm/bin/python detach=true \
    cuda_visible_devices=0 gpu_memory_utilization=0.9 max_model_len=32768 \
    max_lora_rank=32 max_loras=4 'learner_ranks=[32]' "${@:2}" --out "$OUT/serve" ;;
filter)  # per-problem pass@4 of the untrained model (single agent, no house rules)
  if [ -n "${2:-}" ]; then  # pause after the first K problems; `filter` alone resumes
    EVAL eval_filter tasks="$OUT/candidates/taskset.json" stop_after_tasks="$2" "${@:3}" \
      --out "$OUT/filter_rollout"
    exit 0
  fi
  EVAL eval_filter tasks="$OUT/candidates/taskset.json" --out "$OUT/filter_rollout"
  $M data filter tasks="$OUT/candidates/taskset.json" episodes="$OUT/filter_rollout" \
    metric=pass_all lo=0.25 hi=0.75 inclusive=true --out "$OUT/pool" ;;
repos)  # the same filtered problems, as solo (1 per repo) and relay (4 per repo) repos
  $M data repos tasks="$OUT/pool/taskset.json" n_per_repo=1 seed=0 --out "$OUT/repos_n1"
  $M data repos tasks="$OUT/pool/taskset.json" n_per_repo=4 seed=0 --out "$OUT/repos_n4" ;;
gate)  # base rates of the untrained model (the pre-training gate in README.md)
  EVAL eval_gate_n4 tasks="$OUT/repos_n4/taskset.json" --out "$OUT/gate_n4"
  EVAL eval_gate_n1 tasks="$OUT/repos_n1/taskset.json" --out "$OUT/gate_n1"
  python3 "$HERE/analyze.py" "$OUT/gate_n4"; python3 "$HERE/analyze.py" "$OUT/gate_n1" ;;
train)  # one training arm; the learner on GPU 1
  run="${2:?solo|relay_individual|relay_team}"
  tasks="$OUT/repos_n4/taskset.json"; [ "$run" = solo ] && tasks="$OUT/repos_n1/taskset.json"
  CUDA_VISIBLE_DEVICES=1 TRITON_CACHE_DIR=/workspace/.triton/cache \
    $M train rl "$C/base.yaml" "$C/$run.yaml" tasks="$tasks" max_usd=1 \
    local_server_json="$SERVER" concurrency=64 --out "$OUT/$run"
  python3 "$HERE/analyze.py" "$OUT/$run" --every 10 ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
