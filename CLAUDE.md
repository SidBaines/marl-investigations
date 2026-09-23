# marl-investigations (`marli`)

Research repo for **RL on multi-agent LLM systems**. The systems we study are
token-level, tool-using agents over a shared workspace, organised by a
*protocol*:

- **coordinator-free swarms**: parallel peers, each writing its own scratchpad
  and reading the others';
- **coordinator + workers**: an orchestrator spawns workers via a tool and
  composes their reports;
- **multi-session single agents**: fresh contexts, carrying state through
  compaction and/or notes-to-self.

Debate and self-consistency exist only as baselines. We evaluate on hard math
and agentic coding at matched compute. We train with RL either locally (our own
PEFT multi-LoRA learner + vLLM on RunPod) or on Tinker, choosing shared vs
per-role LoRAs and which agents and sessions receive credit.

`AGENTS.md` is a symlink to this file: Claude and Codex follow the same rules.
The founding design is `docs/plans/2026-09-23-agent-systems.md` — read it
before changing library structure. (`2026-09-23-core-infra.md` is an earlier,
superseded debate-centric draft.)

## Four layers

- `src/marli/` — the **library**: general-purpose, reused by every study.
  Every pipeline verb is an importable `async` function *and* a CLI verb.
- `experiments/` — the **lab notebook**: one self-contained directory per
  study (`experiments/<YYYY-MM-DD>_<slug>/`, sub-study dirs inside it rather
  than dozens of top-level files). Low ceremony; results stay as-run — never
  rewrite a study's outputs. Studies *chain library verbs*; they don't
  re-implement runners.
- `examples/` — the **curated on-ramp**: few, minimal, **kept green** by
  smoke tests in `tests/`. Change an example and its test together.
- `docs/wiki/` + `docs/sources/` — the **curated knowledge layer**: what we
  currently believe, with provenance. Schema and ingest/query/lint workflows:
  `docs/wiki/CLAUDE.md`. When asked about past findings, read
  `docs/wiki/index.md` first, not raw experiment dirs. At experiment wrap-up,
  ingest durable findings (not every pilot earns an ingest).

## Library conventions

### Verbs, CLIs, handles

- **Async-native verbs + thin CLI shims.** A verb is
  `async def verb(cfg: SomeConfig) -> Handle`. The single `marli` CLI maps
  `argv → marli.config.compose(Config, *yamls, overrides) → asyncio.run(verb)`.
  `argparse` lives **only** in `src/marli/cli/` (contract test). No other
  module parses argv, prints results, or calls `asyncio.run`.
- **CLI contract (agents chain on it).** Every verb takes `--out DIR` (or
  `--out auto` → `runs/<verb>/<hash12>`), writes its handle there, and prints
  **exactly one JSON line** to stdout (`{"ok", "kind", "manifest", ...}`);
  all logs go to stderr. Exit codes: `0` ok, `1` unexpected failure,
  `2` usage/config error, `3` config-hash mismatch, `4` budget exceeded,
  `5` backend error (`marli.errors`).
- **Idempotent + resumable run dirs.** A run dir records its config hash at
  start. Same hash + complete manifest → no-op; same hash + incomplete →
  resume (skip finished rows); different hash → loud error; `--force` wipes.
  Manifests are written **last** via atomic rename; JSONL rows are appended
  with fsync and a torn final line is tolerated.
- **Handles, not strings.** Verbs pass frozen-dataclass handles backed by a
  JSON manifest beside the bytes (`taskset.json`, `episodes.json`,
  `checkpoint.json`, …). A manifest records its own bytes *relative to
  itself* and its inputs as `{kind, path (absolute), sha256}` so provenance
  chains and survives moves. `Handle.at(path)` is the loud ad-hoc escape
  hatch. Manifests store *resolved* references and **never secrets**.
- **Config-first.** Hparams live in dataclasses/YAML, never as flag strings.
  Unknown config keys are a `ValueError`. Use `str`/`Enum` + validation in
  config dataclasses (OmegaConf rejects `Literal`/`frozenset`). Runtime-only
  fields (out dir, concurrency, timeouts, logging) are excluded from the
  config hash via `field(metadata={"runtime": True})`.
- **File-backed registries.** Registry-shaped things are one YAML per entry
  under the package (`models/`, `tasks/`, `interact/configs/`) with
  `load`/`list` accessors; name == filename; unknown keys raise. Named
  functions (rewards, advantage estimators, protocols) register by name or
  are referenced as `module:function` — never lambdas (a lambda can't be
  reproduced from a manifest).
- **No pipeline framework.** A multi-stage study is a sequence of CLI calls
  (`run.sh`) or `await`s (`run.py`). Orchestration, retries and fan-out live
  in the study, not the library.

### Robustness

- **Error loud, warn on degraded.** A run that cannot work (wrong backend,
  unsupported loss, missing model, GPU too small) raises *before* spending
  compute; a run that works suboptimally warns. A fallback may change *how*
  something is computed, never *what* is measured.
- **Heavy imports are lazy.** `import marli` and `marli --help` must not
  import torch, tinker, tinker_cookbook, vllm, transformers or datasets
  (contract test). GPU stacks (vLLM, the learner's torch) are pod-side pins in
  `requirements/pod-*.txt`; vLLM runs as a separate supervised server process
  from its own venv, never imported.
- **Subprocesses** only for supervised long-running servers (vLLM) and
  read-only `git` provenance calls; config-first (no flag-string plumbing),
  output teed to a log, raise with the log tail on failure, killed on exit.
- **Spend guards.** Anything that costs money (API calls, Tinker, pods) runs
  under a `max_usd` budget and fails with exit code 4 when exceeded. Pods are
  not storage: sync checkpoints off-pod (private HF) before teardown.
- **Pointers, not weights.** Checkpoints are committed as manifests; bytes
  never enter git. `Checkpoint.state` resumes training; `Checkpoint.sampler`
  feeds sampling/eval — never interchange them (`require_state()`).
- **Provenance.** Training runs record git commit/dirty/host (`runlog`) and
  refuse a dirty tree unless `MARLI_ALLOW_DIRTY=1`.

### Token-level RL invariants (why the interaction layer is ours)

- Policies are **token-in/token-out**: prompts are built by the model's
  tinker-cookbook renderer, completions are parsed with
  `renderer.parse_response` from token ids — never from server-side chat
  templates or returned text. Tinker and vLLM must see identical prompt ids.
- **Trainable seats sample at temperature 1, top_p 1, top_k −1** (both
  backends return raw logprobs; anything else is silently off-policy).
- **Every LLM call is recorded with its exact ids; datums use exactly the ids
  the policy saw** — never re-render or re-tokenize history at training time.
  An agent's context is an append-only token buffer (sampled ids appended
  verbatim, new messages rendered as a delta); a context reset (compaction,
  new session, worker spawn) starts a new *segment*.
- Training datums are built **per (agent, segment)**, never merged across
  agents; non-action positions (tool results, pushed messages, prefills,
  forced closes) carry logprob 0 and advantage 0. Frozen and API seats never
  produce datums.
- The RL loss is **sum-reduced**; credit assignment and normalization live in
  `train.credit.*` (recipients, reward_target, baseline_group, norm,
  segment_credit, loss_agg) and are applied by scaling advantages. Invalid
  credit combinations are rejected at config validation.
- Graders see only the submission and the sandbox — never scratchpads or
  notes.
- The API `ChatClient` has an in-memory cache: sampling calls **must** pass a
  per-call `cache_salt` or repeated identical prompts (parallel peers, SC@k,
  GRPO groups) collapse to one sample (scimt postmortem,
  `docs/sources/scimt-prior-latmem-lessons.md`).

## Evaluation conventions

- **Two-stage sample → score** with a sample store: raw episodes are saved
  once; scoring re-runs over saved episodes.
- **One verifier for reward and eval** (math-verify for math; strict letter
  extraction for MCQ). Majority votes group answers by verifier equivalence.
- **Compute-matched comparisons are the default.** Every episode records
  total generated tokens, uncached prompt tokens, LM calls, critical-path
  tokens and peak context. Baselines are a single agent, N independent agents
  + vote (the swarm with visibility off), and sequential multi-session. Report
  accuracy against both total and critical-path tokens: multi-agent gains
  often vanish at equal compute.
- **Report the n; show paired lift** against a baseline cell of the same
  harness (McNemar / paired bootstrap).
- **Never commit benchmark question or transcript text for gated/contamination-
  sensitive sets** (GPQA, AIME, …) — aggregate numbers only.

## Tests

- CPU-only unit tests in `tests/`: no GPU, network, or API keys. Heavy deps
  are faked (`monkeypatch` / `sys.modules`) or `pytest.importorskip`ed.
- Markers: `gpu`, `live` (network/spend), `tinker` (needs the extra).
  Default `pytest` excludes `gpu` and `live`.
- Run: `uv run --extra dev pytest -q` and `uv run --extra dev ruff check`
  from the checkout/worktree root (`pythonpath=["src"]` pins local `src/`).

## Working here (humans and agents)

- Always `uv run …` from the repo root; never `sys.path.insert`.
- One PR per milestone/feature, CI green, small focused commits.
- Before starting a pod or a paid run, say what it costs and confirm with Sid.
