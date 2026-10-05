#!/usr/bin/env bash
# Transfer eval for experiment 2's trained policies (README.md). Phases, one at a time, from anywhere in
# the checkout. Dev box (CPU): candidates. Pod (2xH200): serve, filter, repos, adapters, eval, grid, stop.
# Anywhere: report, extract.
#   ./run.sh candidates            # dev box: 900 held-out DeepCoder problems (no LCB) that pass every independence check
#   ./run.sh serve                 # vLLM for evals: one engine per GPU + MTP, full memory (MODEL picks the model)
#   ./run.sh filter                # 27B base: pass@4 of each candidate, keep [1/4, 3/4] -> out/pool (as exp 1, ~1.9 h)
#   ./run.sh repos                 # held-out repos: 4 per repo (training rules / held-out rules), 3 and 5 per repo
#   ./run.sh adapters              # download this MODEL's trained adapters from HF and load them into the server
#   ./run.sh eval <policy> <cond>  # one cell -> out/eval/<cond>/<policy>; rerun to resume
#   ./run.sh grid                  # GRID=minimal|core|full cells for this MODEL, PARALLEL at a time
#   ./run.sh plan                  # list the cells `grid` would run (no GPU needed)
#   ./run.sh report                # analyze.py over out/eval -> out/report.txt, out/report.json
#   ./run.sh help_report           # help.py: the planted help-request measures -> out/help_report.txt
#   ./run.sh help_audit [args]     # help.py audit: sampled cases with agent text, printed only (private)
#   ./run.sh extract [dest]        # grades-only copy of out/eval (no problem text) -> out/grades, for syncing
#   ./run.sh stop                  # stop this MODEL's server
# MODEL=27b (default) or a3b; SERVE=<serve yaml> overrides the server layout.
# Policies: 27b_base 27b_team_s29 27b_team_s59 27b_indiv_s29 27b_exp21_s29; a3b_base a3b_team_s79 a3b_exp21
#   (A3B experiment 2.1's adapter after step EXP21_A3B_STEP, default 79).
# Conditions: configs/conditions/*.yaml. Extra key=value arguments pass through to `marli eval rollout`.
# Inputs copied to the pod from the dev box: ../out/repos_n4 (training repos) and out/candidates (this study).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" >/dev/null && pwd)"; STUDY="$(dirname "$HERE")"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/out"; C="$HERE/configs"; SC="$STUDY/configs"; E2="$STUDY/exp2_mandatory_rule/configs"
# --no-sync: a plain `uv run` would re-sync .venv to uv.lock and drop the pod's torch pins.
M="uv run --no-sync marli"; PY="uv run --no-sync python"
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
MODEL=${MODEL:-27b}; GRID=${GRID:-core}; PARALLEL=${PARALLEL:-4}; INFLIGHT=${INFLIGHT:-64}
case "$MODEL" in
  27b) SERVE_YAML=${SERVE:-$STUDY/bench/serve/dp2_mtp2.yaml}; SERVE_ARGS=(max_loras=6)
    BASE=qwen3_8_27b; RENDERER=qwen3_8_medium
    POLICIES=(27b_base 27b_team_s29 27b_team_s59 27b_indiv_s29 27b_exp21_s29) ;;
  a3b) SERVE_YAML=${SERVE:-$C/serve_eval_a3b.yaml}; SERVE_ARGS=()
    BASE=qwen3_6_35b_a3b; RENDERER=qwen3_5
    POLICIES=(a3b_base a3b_team_s79 a3b_exp21) ;;
  *) echo "MODEL must be 27b or a3b" >&2; exit 2 ;;
esac
SERVE_DIR="$OUT/serve_$MODEL"; SERVER="$SERVE_DIR/server.json"
ADAPTERS=/workspace/adapters  # on the pod's volume, never the dev box

# The HF path (sidbaines/amber-baton) of each trained policy's adapter; "" = the untrained base model.
adapter() {
  case "$1" in
    27b_base|a3b_base) echo "" ;;
    27b_team_s29) echo exp2/train_team_checks/adapters/sacrifice-relay-exp2-team-policy-s29-d81d6d ;;
    27b_team_s59) echo exp2/train_team_checks/adapters/sacrifice-relay-exp2-team-policy-s59-6bf5e2 ;;
    27b_indiv_s29) echo exp2/train_individual_checks/adapters/sacrifice-relay-exp2-individual-policy-s29-368929 ;;
    27b_exp21_s29) echo exp2/train_opener_checks/adapters/sacrifice-relay-exp2.1-opener-policy-s29-c9483c ;;
    a3b_team_s79) echo exp2/train_team_checks_a3b/adapters/sacrifice-relay-exp2-team-a3b-policy-s79-6d9c30 ;;
    # Being trained on 2026-10-05; its run hash suffix is only known once it is uploaded.
    a3b_exp21) echo "exp2/train_opener_checks_a3b/adapters/sacrifice-relay-exp2.1-opener-a3b-policy-s${EXP21_A3B_STEP:-79}-*" ;;
    *) echo "unknown policy $1" >&2; return 2 ;;
  esac
}
served() { if [ -z "$(adapter "$1")" ]; then echo "$BASE"; else echo "$1"; fi; }

cell() {  # cell <policy> <condition> [key=value ...]
  local policy=$1 cond=$2
  [ -f "$C/conditions/$cond.yaml" ] || { echo "no condition $cond" >&2; return 2; }
  case "$policy" in "$MODEL"_*) ;; *) echo "$policy is not a MODEL=$MODEL policy" >&2; return 2 ;; esac
  mkdir -p "$OUT/eval/$cond"
  local log="$OUT/eval/$cond/$policy.log" rc=0
  echo "$(date -u +%H:%M) start $cond/$policy ${*:3}"
  $M eval rollout "$SC/eval_common.yaml" "$SC/eval_gate_n4.yaml" "$E2/env.yaml" "$E2/variants/checks.yaml" \
    "$C/eval.yaml" "$C/conditions/$cond.yaml" "policies.q.ref=vllm:@$SERVER#$(served "$policy")" \
    "policies.q.model=$BASE" "policies.q.renderer=$RENDERER" max_usd=1 "${@:3}" \
    --out "$OUT/eval/$cond/$policy" > "$log.stdout" 2> "$log" || rc=$?
  echo "$(date -u +%H:%M) $([ $rc = 0 ] && echo done || echo "FAILED (exit $rc, log $log)") $cond/$policy: $(cat "$log.stdout")"
  return $rc
}

grid_cells() {  # "<policy> <condition> [key=value]" lines for GRID and MODEL
  local core=(train_repos heldout new_rules far) surface=(reworded tools notes replies n3 n5)
  local help=(heldout help_a_notes help_b_notes help_a_file) told=(help_control_told help_a_notes_told)
  local base="${MODEL}_base" team; team=$([ "$MODEL" = 27b ] && echo 27b_team_s59 || echo a3b_team_s79)
  case "$GRID" in
    # First 24 held-out repos (48 games per cell); `core` later resumes the same cells to all repos.
    minimal) for p in "$base" "$team"; do for c in heldout new_rules far; do echo "$p $c stop_after_tasks=24"; done; done ;;
    core) for p in "${POLICIES[@]}"; do for c in "${core[@]}"; do echo "$p $c"; done; done ;;
    full) GRID=core grid_cells; for p in "$base" "$team"; do for c in "${surface[@]}"; do echo "$p $c"; done; done ;;
    # Help request: heldout is its control (shared with core, so a no-op there if already run).
    help_minimal) for p in "$base" "$team"; do for c in heldout help_a_notes help_a_file; do
      echo "$p $c stop_after_tasks=24"; done; done ;;
    help) for p in "${POLICIES[@]}"; do for c in "${help[@]}"; do echo "$p $c"; done; done ;;
    help_told) for p in "${POLICIES[@]}"; do for c in "${told[@]}"; do echo "$p $c"; done; done ;;
    help_full) GRID=help grid_cells; GRID=help_told grid_cells
      for p in "${POLICIES[@]}"; do echo "$p help_b_file"; done ;;
    *) echo "GRID must be minimal, core, full, help_minimal, help, help_told or help_full" >&2; return 2 ;;
  esac
}

case "${1:?phase}" in
candidates)
  # Dev box (CPU, ~25 min). 1: draw 1,600 problems, minus experiment 1's 800 candidates by text. 2: overlap.py
  # check flags every drawn problem that repeats a used or an earlier drawn one (text, complete test suites,
  # PrimeIntellect source ids, two embedding models) and writes the exclusion list. 3: the first 900 that
  # pass. 4: overlap.py verify re-checks them (exit 1 if any fails). Reports (numbers only) go to results/.
  # Needs the DeepCoder HF cache, network for the source ids and the models, and EMBED_PY: a python with torch
  # and transformers (default: the dev box's CPU venv); models go to /tmp/marli-embed (MARLI_EMBED_CACHE).
  EXCLUDE=${EXCLUDE:-$STUDY/out/candidates/taskset.json}
  EMBED_PY=${EMBED_PY:-/workspace/marl-investigations/.venv/bin/python}
  B="uv run --extra hub --extra eval marli"
  CHECK=("$HERE/overlap.py" --upstream --models bge,minilm)
  PYTHONPATH=src $B data build "$C/draw.yaml" exclude="$EXCLUDE" --out "$OUT/pool_draw"
  HF_HUB_OFFLINE=0 PYTHONPATH=src "$EMBED_PY" "${CHECK[0]}" check "$EXCLUDE" "$OUT/pool_draw/taskset.json" \
    "$OUT/independence" "${CHECK[@]:1}" > /dev/null
  kept=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['meta']['kept'])" \
    "$OUT/independence/exclusion/taskset.json")
  [ "$kept" -ge 900 ] || { echo "only $kept drawn problems passed; raise max_n in configs/draw.yaml" >&2; exit 1; }
  PYTHONPATH=src $B data build "$C/candidates.yaml" exclude="$OUT/independence/exclusion/taskset.json" \
    --out "$OUT/candidates"
  HF_HUB_OFFLINE=0 PYTHONPATH=src "$EMBED_PY" "${CHECK[0]}" verify "$EXCLUDE" "$OUT/candidates/taskset.json" \
    "$OUT/independence" "${CHECK[@]:1}" > /dev/null
  mkdir -p "$HERE/results"
  cp "$OUT/independence/check.txt" "$HERE/results/independence_check.txt"
  cp "$OUT/independence/verify.txt" "$HERE/results/independence_verify.txt"
  cat "$HERE/results/independence_verify.txt" ;;
serve)
  HF_HUB_OFFLINE=0 $M serve vllm "$SERVE_YAML" detach=true "${SERVE_ARGS[@]}" "${@:2}" --out "$SERVE_DIR" ;;
filter)
  # Experiment 1's filter exactly (../run.sh filter): untrained 27B, one agent, plain code_fn, 4 attempts.
  [ "$MODEL" = 27b ] || { echo "the filter uses the untrained 27B (MODEL=27b)" >&2; exit 2; }
  $M eval rollout "$SC/eval_common.yaml" "$SC/eval_filter.yaml" tasks="$OUT/candidates/taskset.json" \
    "policies.q.ref=vllm:@$SERVER#$BASE" max_usd=1 concurrency=192 "${@:2}" --out "$OUT/filter_rollout"
  $M data filter tasks="$OUT/candidates/taskset.json" episodes="$OUT/filter_rollout" \
    metric=pass_all lo=0.25 hi=0.75 inclusive=true --out "$OUT/pool" ;;
repos)
  # Same seed, so repos_n4 and repos_n4_new group the same problems; only the rule forms differ. The shuffle
  # is also the same for every size, so the caps keep one set of problems: the first 204 (51 x 4 = 68 x 3)
  # and 200 of them for 40 x 5. 51 repos of 4, as training had.
  $M data repos tasks="$OUT/pool/taskset.json" n_per_repo=4 seed=0 max_repos=51 --out "$OUT/repos_n4"
  $M data repos tasks="$OUT/pool/taskset.json" n_per_repo=4 seed=0 max_repos=51 \
    'rule_families=[footer,function,class_attr]' --out "$OUT/repos_n4_new"
  $M data repos tasks="$OUT/pool/taskset.json" n_per_repo=3 seed=0 max_repos=68 --out "$OUT/repos_n3"
  $M data repos tasks="$OUT/pool/taskset.json" n_per_repo=5 seed=0 max_repos=40 --out "$OUT/repos_n5" ;;
adapters)
  # Download, load into this MODEL's server as LoRA adapters named after the policy, and record them in
  # the server manifest (policies only use models it lists; not `adapters` entries, which are a learner's).
  base_url=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['base_url'])" "$SERVER")
  for policy in "${POLICIES[@]}"; do
    path=$(adapter "$policy"); [ -n "$path" ] || continue
    if [ ! -f "$ADAPTERS/$policy/adapter_config.json" ]; then
      HF_HUB_OFFLINE=0 uv run --no-sync hf download sidbaines/amber-baton --include "$path/*" \
        --local-dir "$ADAPTERS/hf" > /dev/null
      found=( $(compgen -G "$ADAPTERS/hf/$path" || true) )
      [ "${#found[@]}" -eq 1 ] || { echo "$policy: expected one adapter at $path, found ${#found[@]}" >&2; exit 5; }
      mkdir -p "$ADAPTERS"; cp -r "${found[0]}" "$ADAPTERS/$policy"
      echo "$policy <- ${found[0]#"$ADAPTERS/hf/"}" | tee -a "$OUT/adapters_$MODEL.txt"
    fi
    curl -sf -X POST "$base_url/v1/load_lora_adapter" -H 'Content-Type: application/json' \
      -d "{\"lora_name\": \"$policy\", \"lora_path\": \"$ADAPTERS/$policy\"}" > /dev/null \
      || echo "$policy: load_lora_adapter refused (already loaded?)" >&2
    uv run --no-sync python -c "import sys; from dataclasses import replace; from marli.serve.vllm import Server
s = Server.load(sys.argv[1]); s.models.count(sys.argv[2]) or replace(s, models=[*s.models, sys.argv[2]]).save()" \
      "$SERVER" "$policy"
    # Both data-parallel engines must serve it: a few one-token requests.
    for _ in 1 2 3 4; do
      curl -sf "$base_url/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$policy\", \"prompt\": \"ok\", \"max_tokens\": 1}" > /dev/null \
        || { echo "$policy: completion through the adapter failed" >&2; exit 5; }
    done
  done
  $PY -c "import sys; from marli.serve.vllm import Server; print(Server.load(sys.argv[1]).models)" "$SERVER" ;;
eval)
  cell "${2:?policy}" "${3:?condition}" "concurrency=${CONC:-$INFLIGHT}" "${@:4}" ;;
grid)
  # PARALLEL cells at a time, INFLIGHT games in total (about 30-40 relay agents per engine is the safe
  # maximum; the variant test overloaded at ~190). A failed cell does not stop the others; rerun to resume.
  export -f cell adapter served; export M SC E2 C OUT SERVER BASE RENDERER MODEL
  mkdir -p "$OUT"
  echo "GRID=$GRID MODEL=$MODEL: $(grid_cells | wc -l) cells, $PARALLEL at a time, $INFLIGHT games in flight"
  grid_cells | xargs -P "$PARALLEL" -L 1 bash -c \
    'cell "$0" "$1" "concurrency='"$((INFLIGHT / PARALLEL))"'" "${@:2}" || true' | tee -a "$OUT/grid.log" ;;
plan)
  grid_cells; echo "GRID=$GRID MODEL=$MODEL: $(grid_cells | wc -l) cells" ;;
report)
  $PY "$HERE/analyze.py" "${2:-$OUT/eval}" --json "$OUT/report.json" "${@:3}" | tee "$OUT/report.txt" ;;
help_report)
  $PY "$HERE/help.py" report "${2:-$OUT/eval}" --json "$OUT/help_report.json" "${@:3}" \
    | tee "$OUT/help_report.txt" ;;
help_audit)
  # PRIVATE: prints agent text. Read it on the dev box; never save or commit it.
  $PY "$HERE/help.py" audit "${2:-$OUT/eval}" "${@:3}" ;;
extract)
  $PY "$HERE/analyze.py" --extract "$OUT/eval" "${2:-$OUT/grades}" ;;
stop)
  $M serve stop server_json="$SERVER" --out "$SERVE_DIR-stop" --force ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
