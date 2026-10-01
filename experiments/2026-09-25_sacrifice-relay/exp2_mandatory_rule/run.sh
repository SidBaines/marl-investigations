#!/usr/bin/env bash
# Experiment 2 (mandatory house rule, no position) on one 2xH200 pod, two-GPU layout ("option 2"):
# one vLLM server split over both GPUs that sleeps while the learner (one copy per GPU) trains.
# Runs ON THE POD from the checkout root (install: docs/runbooks/pod.md). Phases, one at a time:
#   ./run.sh serve          # vLLM, TP2 + MTP + sleep mode
#   ./run.sh dashboard      # live dashboard in the background (port 8888 = the pod's HTTPS proxy)
#   ./run.sh train          # team-reward training; resume = rerun (same --out)
#   ./run.sh check [run]    # abort-rule numbers (default: the training run)
#   ./run.sh gate           # optional: base rates of the untrained model (51 repos x 4), GO/STOP
#   ./run.sh stop           # stop vLLM and the dashboard
# Inputs from experiment 1 (copied in from the dev box): ../out/repos_n4 (51 relay repos).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"; STUDY="$(dirname "$HERE")"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"; SC="$STUDY/configs"
REPOS="$STUDY/out/repos_n4/taskset.json"
# --no-sync: a plain `uv run` would re-sync .venv to uv.lock and drop the pod-train torch pins.
M="uv run --no-sync marli"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
SERVER="$OUT/serve/server.json"

case "${1:?phase}" in
serve)
  HF_HUB_OFFLINE=0 $M serve vllm "$STUDY/bench/serve/tp2_mtp2_sleep.yaml" detach=true "${@:2}" \
    --out "$OUT/serve" ;;
gate)
  $M eval rollout "$SC/eval_common.yaml" "$SC/eval_gate_n4.yaml" "$C/env.yaml" "$C/gate.yaml" \
    tasks="$REPOS" "policies.q.ref=vllm:@$SERVER#qwen3_8_27b" max_usd=1 "${@:2}" --out "$OUT/gate"
  python3 "$HERE/check.py" "$OUT/gate" | tee "$OUT/gate_check.txt" ;;
train)
  # No CUDA_VISIBLE_DEVICES: the server and both learner ranks see both GPUs.
  TRITON_CACHE_DIR=/workspace/.triton/cache \
    $M train rl "$SC/base.yaml" "$C/env.yaml" "$C/train_team.yaml" tasks="$REPOS" max_usd=1 \
    local_server_json="$SERVER" 'local_devices=[cuda:0,cuda:1]' local_sleep_sampler=true \
    concurrency=64 "${@:2}" --out "$OUT/train_team" ;;
dashboard)
  # Off loopback the page needs its key: the URL with ?key= is in out/dashboard/serve.json.
  # Open https://<pod-id>-8888.proxy.runpod.net/?key=<key> (stop Jupyter first if it holds 8888).
  mkdir -p "$OUT"
  nohup $M dashboard "$HERE/dashboard.yaml" "roots=[$OUT]" serve_host=0.0.0.0 \
    serve_port="${2:-8888}" "${@:3}" --out "$OUT/dashboard" > "$OUT/dashboard.log" 2>&1 &
  echo $! > "$OUT/dashboard.pid"
  echo "dashboard pid $!; URL in $OUT/dashboard/serve.json" ;;
check)
  python3 "$HERE/check.py" "${2:-$OUT/train_team}" "${@:3}" ;;
stop)
  $M serve stop server_json="$SERVER" --out "$OUT/serve-stop" --force
  # One SIGINT to uv, which passes it on once; the dashboard then records its manifest and exits.
  [ -f "$OUT/dashboard.pid" ] && kill -INT "$(cat "$OUT/dashboard.pid")" 2>/dev/null || true ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
