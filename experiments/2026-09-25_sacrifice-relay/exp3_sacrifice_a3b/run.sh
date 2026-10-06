#!/usr/bin/env bash
# Experiment 3 (experiment 1's 0/1/3 scoring with experiment 2's prompt changes, on Qwen3.6-35B-A3B): one 2xH200
# pod per arm, in experiment 2's A3B layout: one vLLM server split over both GPUs (TP2 + MTP, sleep mode) that
# sleeps while the learner (one copy per GPU) trains. Runs ON THE POD from the checkout root
# (install: docs/runbooks/pod.md). Phases, one at a time:
#   ./run.sh serve                  # vLLM: ../exp2_mandatory_rule/configs/serve_a3b.yaml
#   ./run.sh dashboard [port]       # live dashboard in the background (default 8888 = the pod's HTTPS proxy)
#   ARM=<team|individual> ./run.sh train    # -> out/train_<arm>; resume = rerun (same --out)
#   ./run.sh check [run dir]        # abort-rule numbers (exp2's check.py), then experiment 1's sacrifice metrics
#   ./run.sh stop                   # stop vLLM and the dashboard
# Inputs from experiment 1 (copied in from the dev box): ../out/repos_n4 (51 relay repos).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"; STUDY="$(dirname "$HERE")"; EXP2="$STUDY/exp2_mandatory_rule"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"; SC="$STUDY/configs"
REPOS="$STUDY/out/repos_n4/taskset.json"
# --no-sync: a plain `uv run` would re-sync .venv to uv.lock and drop the pod-train torch pins.
M="uv run --no-sync marli"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
# Some H200 hosts have broken NVLink SHARP (NCCL "unhandled cuda error" at TP2 start); it is not needed on 2 GPUs.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
ARM=${ARM:-team}
case "$ARM" in team|individual) ;; *) echo "ARM must be team or individual" >&2; exit 2 ;; esac
SERVE_DIR="$OUT/serve"; SERVER="$SERVE_DIR/server.json"

case "${1:?phase}" in
serve)
  HF_HUB_OFFLINE=0 $M serve vllm "$EXP2/configs/serve_a3b.yaml" detach=true \
    cuda_visible_devices=0,1 port=8000 max_loras=4 "${@:2}" --out "$SERVE_DIR" ;;
train)
  # No CUDA_VISIBLE_DEVICES: the server and both learner ranks see both GPUs. local_adapter_check_tol: the A3B's
  # fixed-probe drift is ~0.18 nats from MoE routing near-ties (experiment 2's preflight); sampled-token agreement
  # (kl_sample_train, abort > 5e-3) is the real guard.
  TRITON_CACHE_DIR=/workspace/.triton/cache \
    $M train rl "$SC/base.yaml" "$C/env.yaml" "$C/train_$ARM.yaml" tasks="$REPOS" \
    max_usd=1 local_server_json="$SERVER" 'local_devices=[cuda:0,cuda:1]' local_sleep_sampler=true \
    concurrency=64 local_adapter_check_tol=0.5 "${@:2}" --out "$OUT/train_$ARM" ;;
dashboard)
  # Off loopback the page needs its key: the URL with ?key= is in out/dashboard/serve.json.
  mkdir -p "$OUT"
  nohup $M dashboard "$HERE/dashboard.yaml" "roots=[$OUT]" serve_host=0.0.0.0 \
    serve_port="${2:-8888}" "${@:3}" --out "$OUT/dashboard" > "$OUT/dashboard.log" 2>&1 &
  echo $! > "$OUT/dashboard.pid"
  echo "dashboard pid $!; URL in $OUT/dashboard/serve.json" ;;
check)
  RUN_DIR="${2:-$OUT/train_$ARM}"
  python3 "$EXP2/check.py" "$RUN_DIR" "${@:3}"
  python3 "$STUDY/analyze.py" "$RUN_DIR" --every 10 --bonus 3 ;;
stop)
  $M serve stop server_json="$SERVER" --out "$SERVE_DIR-stop" --force
  # One SIGINT to uv, which passes it on once; the dashboard then records its manifest and exits.
  [ -f "$OUT/dashboard.pid" ] && kill -INT "$(cat "$OUT/dashboard.pid")" 2>/dev/null || true ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
