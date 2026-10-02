#!/usr/bin/env bash
# Experiment 2 (mandatory house rule, no position) on one 2xH200 pod, two-GPU layout ("option 2"):
# one vLLM server split over both GPUs that sleeps while the learner (one copy per GPU) trains.
# Runs ON THE POD from the checkout root (install: docs/runbooks/pod.md). Phases, one at a time:
#   ./run.sh serve          # vLLM, TP2 + MTP + sleep mode
#   ./run.sh dashboard      # live dashboard in the background (port 8888 = the pod's HTTPS proxy)
#   ./run.sh train          # team-reward training; resume = rerun (same --out)
#   ./run.sh check [run]    # abort-rule numbers (default: the training run)
#   ./run.sh gate           # optional: base rates of the untrained model (51 repos x 4), GO/STOP
#   ./run.sh serve_eval     # vLLM for evals only: one engine per GPU (DP2) + MTP, full memory
#   ./run.sh variants       # untrained model under each prompt variant (configs/variants/), GO/STOP each
#   ./run.sh stop           # stop vLLM and the dashboard
#   MODEL=a3b ./run.sh preflight  # A3B only, after serve: memory, logprob agreement, hot-load (~10 min)
# VARIANT=<name> ./run.sh train adds configs/variants/<name>.yaml and trains into out/train_team_<name>.
# ARM=<team|individual|opener> picks configs/train_<arm>.yaml (default team); out dir train_<arm>[_<variant>].
# SLOT=<a|b> picks a GPU pair on a 4-GPU pod: a = GPUs 0,1, port 8000, out/serve (default);
#   b = GPUs 2,3, port 8002, out/serve_b. `serve`, `train` and `stop` follow SLOT.
#   ARM=opener ./run.sh opener loads experiment 2's step-29 adapter into this slot's server as exp2-team-s29.
# MODEL=a3b runs the arm on Qwen3.6-35B-A3B instead of Qwen3.8-27B (phases serve, preflight, train, check,
#   dashboard, stop): configs/serve_a3b.yaml and configs/train_<arm>_a3b.yaml, and every out dir it writes gains an
#   _a3b suffix (out/serve_a3b, out/train_team_checks_a3b), so the 27B runs are never touched.
# Inputs from experiment 1 (copied in from the dev box): ../out/repos_n4 (51 relay repos).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"; STUDY="$(dirname "$HERE")"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"; SC="$STUDY/configs"
REPOS="$STUDY/out/repos_n4/taskset.json"
# --no-sync: a plain `uv run` would re-sync .venv to uv.lock and drop the pod-train torch pins.
M="uv run --no-sync marli"
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1
# Some H200 hosts have broken NVLink SHARP (NCCL "unhandled cuda error" at TP2 start); it is not needed on 2 GPUs.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
SLOT=${SLOT:-a}; ARM=${ARM:-team}; MODEL=${MODEL:-27b}
case "$MODEL" in
  27b) SFX=""; SERVE_YAML="$STUDY/bench/serve/tp2_mtp2_sleep.yaml" ;;
  a3b) SFX="_a3b"; SERVE_YAML="$C/serve_a3b.yaml" ;;
  *) echo "MODEL must be 27b or a3b" >&2; exit 2 ;;
esac
case "${1:-}" in
  gate|variants|serve_eval|opener) [ "$MODEL" = 27b ] || { echo "phase $1 is 27B-only" >&2; exit 2; } ;;
  preflight) [ "$MODEL" = a3b ] || { echo "phase preflight is A3B-only (MODEL=a3b)" >&2; exit 2; } ;;
esac
case "$SLOT" in
  a) GPUS=0,1; PORT=8000; SERVE_DIR="$OUT/serve$SFX"; DEVICES='local_devices=[cuda:0,cuda:1]'; LORAS=4 ;;
  b) GPUS=2,3; PORT=8002; SERVE_DIR="$OUT/serve_b$SFX"; DEVICES='local_devices=[cuda:2,cuda:3]'; LORAS=6 ;;
  *) echo "SLOT must be a or b" >&2; exit 2 ;;
esac
[ "$ARM" = opener ] && LORAS=6  # the frozen opener adapter needs a slot beside the learner's
SERVER="$SERVE_DIR/server.json"
EVAL_SERVER="$OUT/serve_eval/server.json"
VARIANT_ARGS=(); RUN="train_$ARM$SFX"; TRAIN_CFG="$C/train_$ARM$SFX.yaml"
if [ -n "${VARIANT:-}" ]; then VARIANT_ARGS=("$C/variants/$VARIANT.yaml"); RUN="train_${ARM}_$VARIANT$SFX"; fi
OPENER_DIR=/workspace/opener/sacrifice-relay-exp2-team-policy-s29-d81d6d

case "${1:?phase}" in
serve)
  HF_HUB_OFFLINE=0 $M serve vllm "$SERVE_YAML" detach=true \
    cuda_visible_devices="$GPUS" port="$PORT" max_loras="$LORAS" "${@:2}" --out "$SERVE_DIR" ;;
opener)
  # Experiment 2's team-trained step-29 adapter (HF sidbaines/amber-baton) as a frozen seat on this slot.
  [ -f "$OPENER_DIR/adapter_config.json" ] || HF_HUB_OFFLINE=0 uv run --no-sync hf download \
    sidbaines/amber-baton --include "exp2/train_team_checks/adapters/$(basename "$OPENER_DIR")/*" \
    --local-dir /workspace/opener-hf
  [ -f "$OPENER_DIR/adapter_config.json" ] || { mkdir -p /workspace/opener; cp -r \
    "/workspace/opener-hf/exp2/train_team_checks/adapters/$(basename "$OPENER_DIR")" /workspace/opener/; }
  curl -sf -X POST "http://127.0.0.1:$PORT/v1/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"exp2-team-s29\", \"lora_path\": \"$OPENER_DIR\"}" && echo
  # Record it in the server manifest: policies only use models the manifest lists. It is not an
  # `adapters` entry, so the learner's slot bookkeeping and stale-adapter sweep leave it alone.
  uv run --no-sync python -c "import sys; from dataclasses import replace; from marli.serve.vllm import Server
s = Server.load(sys.argv[1]); s.models.count('exp2-team-s29') or replace(s, models=[*s.models, 'exp2-team-s29']).save()
print(Server.load(sys.argv[1]).models)" "$SERVER" ;;
gate)
  $M eval rollout "$SC/eval_common.yaml" "$SC/eval_gate_n4.yaml" "$C/env.yaml" "$C/gate.yaml" \
    tasks="$REPOS" "policies.q.ref=vllm:@$SERVER#qwen3_8_27b" max_usd=1 "${@:2}" --out "$OUT/gate"
  python3 "$HERE/check.py" "$OUT/gate" | tee "$OUT/gate_check.txt" ;;
preflight)
  TRITON_CACHE_DIR=/workspace/.triton/cache uv run --no-sync python "$HERE/a3b_preflight.py" "$SERVER" \
    2>&1 | tee "$OUT/preflight$SFX.log" ;;
train)
  # No CUDA_VISIBLE_DEVICES: the server and both learner ranks see both GPUs.
  TRITON_CACHE_DIR=/workspace/.triton/cache \
    $M train rl "$SC/base.yaml" "$C/env.yaml" "${VARIANT_ARGS[@]}" "$TRAIN_CFG" tasks="$REPOS" \
    max_usd=1 local_server_json="$SERVER" "$DEVICES" local_sleep_sampler=true \
    concurrency=64 "${@:2}" --out "$OUT/$RUN" ;;
serve_eval)
  HF_HUB_OFFLINE=0 $M serve vllm "$STUDY/bench/serve/dp2_mtp2.yaml" detach=true "${@:2}" \
    --out "$OUT/serve_eval" ;;
variants)
  # The current prompt on the first 16 repos; each variant on the first 8 (a subset), 4 each.
  pids=()
  for v in baseline tools others checks all; do
    n=8; [ "$v" = baseline ] && n=16
    $M eval rollout "$SC/eval_common.yaml" "$SC/eval_gate_n4.yaml" "$C/env.yaml" "$C/gate.yaml" \
      "$C/variants/$v.yaml" tasks="$REPOS" max_tasks=$n concurrency=$((n * 4)) \
      "policies.q.ref=vllm:@$EVAL_SERVER#qwen3_8_27b" max_usd=1 --out "$OUT/variants/$v" \
      > "$OUT/variants_$v.log" 2>&1 & pids+=($!)
  done
  wait "${pids[@]}" || true
  for v in baseline tools others checks all; do
    echo "== $v"; python3 "$HERE/check.py" "$OUT/variants/$v"
  done | tee "$OUT/variants_summary.txt" ;;
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
  $M serve stop server_json="$SERVER" --out "$SERVE_DIR-stop" --force
  # One SIGINT to uv, which passes it on once; the dashboard then records its manifest and exits.
  [ -f "$OUT/dashboard.pid" ] && kill -INT "$(cat "$OUT/dashboard.pid")" 2>/dev/null || true ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
