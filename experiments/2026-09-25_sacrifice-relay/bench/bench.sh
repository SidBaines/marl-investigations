#!/usr/bin/env bash
# Inference + learner speed benchmark for the sacrifice relay (see EXPERIMENT_1.md). Runs ON THE POD.
#   ./bench.sh serve <variant> [overrides]   # serve/<variant>.yaml -> out/bench/serve_<variant>, + metrics watcher
#   ./bench.sh stop <variant>
#   ./bench.sh load <variant> <concurrency> <n_tasks>   # throwaway single-agent code rollouts
#   ./bench.sh window <variant> <run_dir>    # vLLM/GPU metrics over a run's wall-clock window
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"; STUDY="$(dirname "$HERE")"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$STUDY/out/bench"; C="$STUDY/configs"; CAND="$STUDY/out/candidates/taskset.json"
M="uv run --no-sync marli"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
mkdir -p "$OUT"
v="${2:?variant}"; SERVE="${SERVE_DIR:-$OUT/serve_$v}"  # SERVE_DIR=../out/serve for the kept filter run
case "$1" in
serve)
  t0=$(date +%s)
  $M serve vllm "$HERE/serve/$v.yaml" detach=true "${@:3}" --out "$SERVE"
  echo "{\"variant\": \"$v\", \"startup_s\": $(( $(date +%s) - t0 ))}" > "$SERVE/startup.json"
  url=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_url'].removesuffix('/v1'))" "$SERVE/server.json")
  nohup python3 "$HERE/metrics.py" watch "$url" "$OUT/metrics_$v.jsonl" --every 5 \
    > /dev/null 2>&1 & echo $! > "$OUT/watch_$v.pid"
  grep -E "KV cache|Maximum concurrency|model weights took|CUDA graph|speculative|prefix" "$SERVE/server.log" | tail -n 12 || true ;;
stop)
  $M serve stop server_json="$SERVE/server.json" --out "$SERVE-stop" --force || true
  [ -f "$OUT/watch_$v.pid" ] && kill "$(cat "$OUT/watch_$v.pid")" 2>/dev/null || true ;;
load)
  conc="${3:?concurrency}"; n="${4:?n_tasks}"; run="$OUT/load_${v}_c${conc}_n${n}"
  t0=$(date +%s.%N)
  $M eval rollout "$C/eval_common.yaml" "$C/eval_filter.yaml" tasks="$CAND" max_tasks="$n" \
    episodes_per_task=1 concurrency="$conc" "policies.q.ref=vllm:@$SERVE/server.json#qwen3_8_27b" \
    max_usd=1 --out "$run"
  echo "{\"start\": $t0, \"end\": $(date +%s.%N)}" > "$run/bench_window.json"
  "$0" window "$v" "$run" ;;
window)
  run="${3:?run_dir}"
  read -r s e < <(python3 -c "import json,sys; w=json.load(open(sys.argv[1])); print(w['start'], w['end'])" "$run/bench_window.json")
  python3 "$HERE/metrics.py" summary "$OUT/metrics_$v.jsonl" --start "$s" --end "$e" | tee "$run/bench_metrics.json"
  python3 "$HERE/episodes.py" "$run" | tee "$run/bench_episodes.json" ;;
*) echo "unknown command $1" >&2; exit 2 ;;
esac
