#!/usr/bin/env bash
# Standard cooperation evals for experiment 2's policies (README.md): Li & Shirado's one-shot
# games (Inspect), HiddenBench (upstream harness) and FAIRGAME's volunteer's dilemma (upstream
# runner), all against the vLLM chat endpoint. Phases, one at a time, from anywhere in the checkout:
#   ./run.sh setup              # clone the two upstream harnesses at their pinned commits, apply
#                               #   patches/, make their venvs; install marli's [external] pins
#   ./run.sh serve              # pod: vLLM with the reasoning parser (configs/serve_$MODEL.yaml)
#   ./run.sh link <serve_dir>   # pod: share an already running eval server instead (e.g. the transfer
#                               #   eval's), if it was started with --reasoning-parser
#   ./run.sh adapters           # pod: download this MODEL's adapters from HF and load them
#   ./run.sh preflight          # pod: template parity, thinking split, JSON mode (preflight.py)
#   ./run.sh games              # Inspect: Li & Shirado games, every cell
#   ./run.sh hiddenbench        # upstream harness for each cell x session, then ingest
#   ./run.sh volunteer          # FAIRGAME runner for each cell x game, then ingest
#   ./run.sh help               # Inspect: planted help request (stock react agent), every cell
#   ./run.sh audit <cell> <id>  # print one planted-help sample (private; never commit the output)
#   ./run.sh report             # print the three RESULTS.md files; copy aggregates to results/
#   ./run.sh smoke              # dev box (CPU): all three suites against tests/_fake_openai.py
#   ./run.sh stop               # stop this MODEL's server
# MODEL=27b (default) or a3b. CELLS=a,b,... restricts cells (all phases); SESSIONS=<n> HiddenBench
# sessions per task (default 5; the paper ran 10); FULL_CELLS cells also run the Full Profile
# control; GAMES=<n> volunteer games per cell (default 50); PAR=<n> concurrent harness processes.
# The A3B experiment-2.1 adapter is the final one (step EXP21_A3B_STEP, default 79).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
REL="experiments/$(basename "$HERE")"
OUT="$HERE/out"; C="$HERE/configs"
MODEL=${MODEL:-27b}; SESSIONS=${SESSIONS:-5}; GAMES=${GAMES:-50}; PAR=${PAR:-8}
WORK=${WORK:-/workspace/coop-evals}  # upstream checkouts + venvs (dev box: WORK=/tmp/coop-evals)
M=${M:-uv run --no-sync marli}; PY=${PY:-uv run --no-sync python}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
HB_REPO=https://github.com/Yassellee/HiddenBench_ICML; HB_COMMIT=3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1
FG_REPO=https://github.com/aira-list/FAIRGAME; FG_COMMIT=fc302a642c6f7cc0c439c2ae957a45f5954f4525
HB="$WORK/hiddenbench"; FG="$WORK/fairgame"
case "$MODEL" in
  27b) RENDERER=qwen3_8_medium; FULL_CELLS=${FULL_CELLS:-base,team_s59} ;;
  a3b) RENDERER=qwen3_5; FULL_CELLS=${FULL_CELLS:-base,team_s79} ;;
  *) echo "MODEL must be 27b or a3b" >&2; exit 2 ;;
esac
SERVE_DIR="$OUT/serve_$MODEL"; SERVER="$SERVE_DIR/server.json"; ADAPTERS=/workspace/adapters
[ "${1:-}" = smoke ] && { OUT="$OUT/smoke"; SERVER="$OUT/server.json"; }

# The HF path (sidbaines/amber-baton) of each trained policy's adapter, as in the transfer eval.
adapter() {
  case "$1" in
    27b_team_s29) echo exp2/train_team_checks/adapters/sacrifice-relay-exp2-team-policy-s29-d81d6d ;;
    27b_team_s59) echo exp2/train_team_checks/adapters/sacrifice-relay-exp2-team-policy-s59-6bf5e2 ;;
    27b_indiv_s29) echo exp2/train_individual_checks/adapters/sacrifice-relay-exp2-individual-policy-s29-368929 ;;
    27b_exp21_s29) echo exp2/train_opener_checks/adapters/sacrifice-relay-exp2.1-opener-policy-s29-c9483c ;;
    a3b_team_s79) echo exp2/train_team_checks_a3b/adapters/sacrifice-relay-exp2-team-a3b-policy-s79-6d9c30 ;;
    a3b_exp21) echo "exp2/train_opener_checks_a3b/adapters/sacrifice-relay-exp2.1-opener-a3b-policy-s${EXP21_A3B_STEP:-79}-*" ;;
    *) echo "" ;;
  esac
}
base_url() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['base_url'].rstrip('/'))" "$SERVER"; }
only() { [ -n "${CELLS:-}" ] && echo "--only $CELLS" || true; }
# The study config to ingest: CELLS restricts it to a subset (written under out/, not committed).
config_for() {  # config_for <suite-config-name>
  if [ -n "${CELLS:-}" ]; then
    $PY "$HERE/harness.py" subset "$C/$1.yaml" "$OUT/configs/$1.yaml" --only "$CELLS"
    echo "$OUT/configs/$1.yaml"
  else echo "$C/$1.yaml"; fi
}
rotate() { $PY -c "import sys; s=sys.argv[1].split(','); k=int(sys.argv[2])%len(s); print(','.join(s[k:]+s[:k]))" "$1" "$2"; }

# One HiddenBench session (all 65 tasks in parallel) for one cell: run_hb <dir> <seats> <profile> <s>.
run_hb() {
  local dir=$1 seats=$2 profile=$3 s=$4 out
  out="$dir/${profile}_s$s.json"; [ -f "$out" ] && return 0
  local request temp rest
  request=$($PY "$HERE/harness.py" request hiddenbench "$C/models/$MODEL.yaml")
  temp=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['temperature'])" "$request")
  rest=$($PY -c "import json,sys; d=json.loads(sys.argv[1]); d.pop('temperature'); print(json.dumps(d))" "$request")
  local rotated url; rotated=$(rotate "$seats" "$s"); url="$(base_url)/v1"  # before cd: $PY needs the repo
  (cd "$HB" && HB_KEY=EMPTY "$HB/.venv/bin/hiddenbench" eval --benchmark "${HB_BENCH:-data/benchmark.json}" \
    --profile "$profile" --provider openai-compatible --base-url "$url" --api-key-env HB_KEY \
    --model "${rotated%%,*}" --seat-models "$rotated" --temperature "$temp" --request-kwargs "$rest" \
    --rounds "${HB_ROUNDS:-15}" --duplications 1 --seed "$s" --max-workers "${HB_WORKERS:-65}" \
    --output "$out.part") > "$dir/${profile}_s$s.log" 2>&1 && mv "$out.part" "$out"
}

# One FAIRGAME volunteer's-dilemma game for one cell: run_fg <dir> <seats> <k>.
run_fg() {
  local dir=$1 seats=$2 k=$3
  [ -f "$dir/seed$k.csv" ] && return 0
  local request temp tokens rest labels res="$dir/resources/seed$k"
  request=$($PY "$HERE/harness.py" request fairgame_volunteer "$C/models/$MODEL.yaml")
  temp=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['temperature'])" "$request")
  tokens=$($PY -c "import json,sys; print(json.loads(sys.argv[1])['max_tokens'])" "$request")
  rest=$($PY -c "import json,sys; d=json.loads(sys.argv[1]); d.pop('temperature'); d.pop('max_tokens'); print(json.dumps(d))" "$request")
  labels=$($PY "$HERE/harness.py" fairgame-config "$FG" "$res" "game$k" --seats "$(rotate "$seats" "$k")" --seed "$k")
  rm -f "$dir/seed$k.parse_failures.jsonl"
  local url; url="$(base_url)/v1"
  (cd "$FG" && FAIRGAME_RESOURCES_DIR="$res" HOSTED_VLLM_API_BASE="$url" HOSTED_VLLM_API_KEY=EMPTY \
    FAIRGAME_LLM_TEMPERATURE="$temp" FAIRGAME_LLM_MAX_TOKENS="$tokens" FAIRGAME_LITELLM_KWARGS="$rest" \
    FAIRGAME_PARSE_FAILURE_LOG="$dir/seed$k.parse_failures.jsonl" FAIRGAME_LLM_TIMEOUT=1800 \
    "$FG/.venv/bin/python" main.py local "volunteer/game$k" --template volunteer --out "$dir/seed$k.csv.part") \
    > "$dir/seed$k.log" 2>&1 && mv "$dir/seed$k.csv.part" "$dir/seed$k.csv"
  echo "$labels" > "$dir/labels.json"
}
export -f run_hb run_fg rotate base_url; export HERE C MODEL PY HB FG SERVER

hiddenbench_all() {
  local cfg request; cfg=$(config_for "hiddenbench_$MODEL")
  request=$($PY "$HERE/harness.py" request hiddenbench "$C/models/$MODEL.yaml")
  $PY "$HERE/harness.py" cells "$cfg" | while IFS=$'\t' read -r label seats results; do
    results=$(realpath -m "$results"); mkdir -p "$results"  # harnesses run from their checkouts
    $PY "$HERE/harness.py" sidecar "$results" --repo "$HB_REPO" --commit "$(git -C "$HB" rev-parse HEAD)" \
      --patch "$HERE/patches/hiddenbench.patch" --request "$request"
    for ((s = 0; s < SESSIONS; s++)); do echo "$results $seats hidden $s"; done
    case ",$FULL_CELLS," in *",$label,"*) for ((s = 0; s < SESSIONS; s++)); do echo "$results $seats full $s"; done ;; esac
  done | xargs -P "$PAR" -L 1 bash -c 'run_hb "$@" || echo "FAILED hiddenbench $*" >&2' _
  $M eval external "$C/models/$MODEL.yaml" "$cfg" --out "$OUT/hiddenbench_${MODEL}_report"
}

volunteer_all() {
  local cfg request; cfg=$(config_for "volunteer_$MODEL")
  request=$($PY "$HERE/harness.py" request fairgame_volunteer "$C/models/$MODEL.yaml")
  $PY "$HERE/harness.py" cells "$cfg" | while IFS=$'\t' read -r label seats results; do
    results=$(realpath -m "$results"); mkdir -p "$results"
    for ((k = 0; k < GAMES; k++)); do echo "$results $seats $k"; done
  done | xargs -P "$PAR" -L 1 bash -c 'run_fg "$@" || echo "FAILED volunteer $*" >&2' _
  $PY "$HERE/harness.py" cells "$cfg" | while IFS=$'\t' read -r label seats results; do
    results=$(realpath -m "$results")
    games=$($PY -c "import json,sys; print(json.dumps([f'seed{k}' for k in range(int(sys.argv[1]))]))" "$GAMES")
    $PY "$HERE/harness.py" sidecar "$results" --repo "$FG_REPO" --commit "$(git -C "$FG" rev-parse HEAD)" \
      --patch "$HERE/patches/fairgame.patch" --request "$request" \
      --extra "{\"labels\": $(cat "$results/labels.json"), \"games\": $games, \"model_prefix\": \"litellm:hosted_vllm/\"}"
  done
  $M eval external "$C/models/$MODEL.yaml" "$cfg" --out "$OUT/volunteer_${MODEL}_report"
}

case "${1:?phase}" in
setup)
  mkdir -p "$WORK"
  [ -d "$HB/.git" ] || git clone -q "$HB_REPO" "$HB"
  [ -d "$FG/.git" ] || git clone -q "$FG_REPO" "$FG"
  git -C "$HB" checkout -q "$HB_COMMIT" && git -C "$HB" checkout -q -- . && git -C "$HB" apply "$HERE/patches/hiddenbench.patch"
  git -C "$FG" checkout -q "$FG_COMMIT" && git -C "$FG" checkout -q -- . && git -C "$FG" apply "$HERE/patches/fairgame.patch"
  [ -x "$HB/.venv/bin/hiddenbench" ] || { uv venv -q -p 3.12 "$HB/.venv" && VIRTUAL_ENV="$HB/.venv" uv pip install -q -e "$HB"; }
  # FAIRGAME pins its runtime in requirements.txt.
  [ -x "$FG/.venv/bin/python" ] || { uv venv -q -p 3.12 "$FG/.venv" && VIRTUAL_ENV="$FG/.venv" uv pip install -q -r "$FG/requirements.txt" && VIRTUAL_ENV="$FG/.venv" uv pip install -q --no-deps -e "$FG"; }
  # marli's [external] extra without re-syncing the pod's training pins (uv run --no-sync).
  [ "${SKIP_EXTRA:-}" = 1 ] || uv pip install -q "inspect-ai==0.3.276" "openai>=1.99" jinja2
  echo "hiddenbench $(git -C "$HB" rev-parse HEAD); fairgame $(git -C "$FG" rev-parse HEAD)" ;;
serve)
  HF_HUB_OFFLINE=0 $M serve vllm "$C/serve_$MODEL.yaml" detach=true "${@:2}" --out "$SERVE_DIR" ;;
link)
  src=$(cd "${2:?serve dir of a running eval server}" && pwd)
  grep -q -- "--reasoning-parser" "$src/config.yaml" || { echo "$src was not started with --reasoning-parser" >&2; exit 2; }
  mkdir -p "$OUT"; ln -sfn "$src" "$SERVE_DIR"; echo "$SERVE_DIR -> $src" ;;
adapters)
  for policy in $($PY -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['models']))" "$SERVER") \
      $(case $MODEL in 27b) echo 27b_team_s29 27b_team_s59 27b_indiv_s29 27b_exp21_s29 ;; a3b) echo a3b_team_s79 a3b_exp21 ;; esac); do
    path=$(adapter "$policy"); [ -n "$path" ] || continue
    if [ ! -f "$ADAPTERS/$policy/adapter_config.json" ]; then
      HF_HUB_OFFLINE=0 uv run --no-sync hf download sidbaines/amber-baton --include "$path/*" --local-dir "$ADAPTERS/hf" > /dev/null
      found=( $(compgen -G "$ADAPTERS/hf/$path" || true) )
      [ "${#found[@]}" -eq 1 ] || { echo "$policy: expected one adapter at $path, found ${#found[@]}" >&2; exit 5; }
      mkdir -p "$ADAPTERS"; cp -r "${found[0]}" "$ADAPTERS/$policy"
    fi
    curl -sf -X POST "$(base_url)/v1/load_lora_adapter" -H 'Content-Type: application/json' \
      -d "{\"lora_name\": \"$policy\", \"lora_path\": \"$ADAPTERS/$policy\"}" > /dev/null \
      || echo "$policy: load_lora_adapter refused (already loaded?)" >&2
    $PY -c "import sys; from dataclasses import replace; from marli.serve.vllm import Server
s = Server.load(sys.argv[1]); s.models.count(sys.argv[2]) or replace(s, models=[*s.models, sys.argv[2]]).save()" "$SERVER" "$policy"
  done
  $PY -c "import sys; from marli.serve.vllm import Server; print(Server.load(sys.argv[1]).models)" "$SERVER" ;;
preflight)
  $PY "$HERE/preflight.py" "$SERVER" "$C/models/$MODEL.yaml" "$RENDERER" "${@:2}" | tee "$OUT/preflight_$MODEL.txt" ;;
games)
  $M eval external "$C/models/$MODEL.yaml" "$(config_for "games_$MODEL")" "${@:2}" --out "$OUT/games_$MODEL" ;;
help)
  # The marli sandbox needs root + Landlock (pods have both); elsewhere add task_args.sandbox=local.
  $M eval external "$C/models/$MODEL.yaml" "$(config_for "help_$MODEL")" "${@:2}" --out "$OUT/help_$MODEL" ;;
audit)
  $PY "$HERE/audit_help.py" "$OUT/help_$MODEL" "${2:?cell}" "${3:?sample id, e.g. fizzbuzz/A_notes}" ;;
hiddenbench) hiddenbench_all ;;
volunteer) volunteer_all ;;
report)
  mkdir -p "$HERE/results"
  for r in "$OUT"/games_"$MODEL" "$OUT"/hiddenbench_"$MODEL"_report "$OUT"/volunteer_"$MODEL"_report \
      "$OUT"/help_"$MODEL"; do
    [ -f "$r/RESULTS.md" ] || continue
    cat "$r/RESULTS.md"; cp "$r/RESULTS.md" "$HERE/results/$(basename "$r").md"
    cp "$r/results.jsonl" "$HERE/results/$(basename "$r").jsonl"  # aggregates only, no text
  done ;;
smoke)
  # Dev box, CPU, no network after setup: a fake vLLM-like server stands in for the policies.
  rm -rf "$OUT"; mkdir -p "$OUT"
  M="env PYTHONPATH=src uv run --extra external marli"; PY="env PYTHONPATH=src uv run --extra external python"
  export PY; names=$(grep -ho 'seats: \[[^]]*\]' "$C"/hiddenbench_"$MODEL".yaml | tr -d '[] ' | sed 's/seats://' | tr ',' '\n' | sort -u | paste -sd,)
  selfish=$(echo "$names" | tr ',' '\n' | grep -v -E '_base$|qwen3' | paste -sd,)
  coop=$(echo "$names" | tr ',' '\n' | grep -E 'qwen3' | paste -sd,)
  players="p_tells=helps_tells,p_silent=helps_silent,p_ignore=ignores"
  PYTHONPATH=tests:src uv run --extra external python tests/_fake_openai.py --port "${SMOKE_PORT:-8799}" \
    --cooperative "$coop" --selfish "$selfish" --players "$players" > "$OUT/fake.log" 2>&1 & fake=$!
  trap 'kill $fake 2>/dev/null' EXIT; sleep 3
  echo "{\"base_url\": \"http://127.0.0.1:${SMOKE_PORT:-8799}\", \"models\": $(echo "$names,p_tells,p_silent,p_ignore" | $PY -c "import json,sys; print(json.dumps(sys.stdin.read().strip().split(',')))"), \"hf_id\": \"smoke\"}" > "$SERVER"
  sed "s#$REL/out/serve_$MODEL/server.json#$REL/out/smoke/server.json#" "$C/games_$MODEL.yaml" > "$OUT/games.yaml"
  $M eval external "$C/models/$MODEL.yaml" "$OUT/games.yaml" task_args.trials=3 --out "$OUT/games_$MODEL"
  for suite in hiddenbench volunteer; do
    sed "s#$REL/out/${suite}_$MODEL/#$REL/out/smoke/${suite}_$MODEL/#" "$C/${suite}_$MODEL.yaml" > "$OUT/$suite.yaml"
  done
  C_SAVE=$C; export SESSIONS=2 GAMES=2 HB_BENCH=data/benchmark_short.json HB_ROUNDS=2 FULL_CELLS=base
  C_OUT="$OUT"; cfg=$OUT/hiddenbench.yaml
  $PY "$HERE/harness.py" subset "$cfg" "$OUT/configs/hiddenbench_$MODEL.yaml" --only base,mixed_team1
  $PY "$HERE/harness.py" subset "$OUT/volunteer.yaml" "$OUT/configs/volunteer_$MODEL.yaml" --only base,mixed_team1
  C="$OUT/configs"; mkdir -p "$C/models"; cp "$C_SAVE/models/$MODEL.yaml" "$C/models/"
  CELLS="" hiddenbench_all; CELLS="" volunteer_all
  # Planted help: three scripted players (helps and tells, helps silently, ignores), 3 tasks.
  sandbox=$([ "$(id -u)" = 0 ] && echo marli || echo local)
  { echo "suite: planted_help"; echo "baseline: ignores"
    echo "task_args: {tasks: [rle_encode, chunk, dedupe], sandbox: $sandbox}"; echo "cells:"
    for p in ignores:p_ignore helps_tells:p_tells helps_silent:p_silent; do
      echo "  - {label: ${p%%:*}, policy: \"vllm:@$SERVER#${p##*:}\"}"; done; } > "$OUT/help.yaml"
  $M eval external "$C/models/$MODEL.yaml" "$OUT/help.yaml" --out "$OUT/help_$MODEL"
  for r in games_"$MODEL" hiddenbench_"$MODEL"_report volunteer_"$MODEL"_report; do echo "== $r"; sed -n '7,40p' "$OUT/$r/RESULTS.md"; done
  echo "== help_$MODEL (A_notes rows)"; grep -E '^\| (Cell|---|[a-z_]+ \| A_notes \| (did_help|told_user|did_told|did_silent|ignored_silent|read_note|hidden_pass) )' "$OUT/help_$MODEL/RESULTS.md" ;;
stop)
  $M serve stop server_json="$SERVER" --out "$SERVE_DIR-stop" --force ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
