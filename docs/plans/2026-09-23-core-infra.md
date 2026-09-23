# Plan: core infra for `marl-investigations` (multi-agent RL on LLMs)

## Context

**The ask.** Sid wants a private repo (`SidBaines/marl-investigations`, owned
by Sid, not ArcadiaImpact) for studying multi-agent RL on LLMs. It should be
structured like `science-of-midtraining` (scimt, `/root/repos/science-of-midtraining`):
- a general-purpose core library covering training, serving/eval, dataset
  building, and a shared, swappable multi-agent **interaction** layer;
- an `experiments/` notebook layer whose studies chain library components.

Unlike scimt, which banned CLIs in #155, every component gets a CLI. That lets
Claude/Codex agents chain steps across models, datasets and protocols without
glue code. `/workspace/marl-investigations` is currently empty.

**Decisions (Sid, 2026-09-23).**
- **First focus: cooperative reasoning.** Debate, solver/critic and round-robin
  discussion on verifiable benchmarks (math, MCQ).
- **Topology.** A role→policy "seating" that supports every topology. v1 ships
  (a) shared-weights self-play and (b) a learner against frozen or API
  partners. Separately co-trained policies come later with no API change.
- **Scale.** Small dense models locally; big MoE through Tinker.
- **Training backends.** Native local trainer plus a Tinker adapter. The local
  learner is our own, **Tinker-shaped** (it exposes the same
  forward_backward / optim_step / save / sampler primitives), so **one
  algorithm loop serves both backends**. The v1 local backend is LoRA on one
  learner GPU, with vLLM on another GPU or colocated. Multi-GPU FSDP2 and full
  fine-tuning come later. Big MoE goes through the Tinker adapter.
- **Package/CLI:** `marli`.
- **Build mode:** Codex builds and Claude reviews (the `codex-driven-development`
  skill).
- **GitHub:** private repo, one PR per milestone, CI.

**Environment.**
- The dev box is CPU-only. It has `runpodctl`, `codex`, `uv 0.10.9` and
  Python 3.12.
- Keys are present for Tinker, W&B, HF (`HF_WRITE_TOKEN_PERSONAL`), OpenRouter,
  Anthropic and OpenAI.
- Current versions: tinker 0.30.1, tinker-cookbook 0.5.7, vllm 0.30.0,
  math-verify 0.9.0, textarena 0.7.4.

## Key design decisions (from research + review)

1. **Our own interaction layer.** No framework fits.
   - **Shape:** async-coroutine protocols with AEC/Concordia turn semantics and
     a visibility-tagged message board.
   - **Token-in/token-out policies.** String-level chat frameworks cause
     retokenization drift, and Qwen3 strips thinking from history; either
     makes RL silently off-policy.
   - **References:** verifiers v0.3 `Env.run(task, agents)`, tinker-cookbook's
     `Transition`, and SPIRAL's role-conditioned advantages.
   - **Shared by eval and RL.**
2. **Renderers come from `tinker_cookbook.renderers`.** Both the Tinker
   sampler and vLLM see identical prompt token ids. We use the cookbook as a
   library only (renderers, weight export, LR helpers, KL metric), never its
   `Env` or `rl.train`, for three reasons:
   - its `Env` is single-agent;
   - it does group-mean advantages only, with mixed roles;
   - it supports only one training client.
3. **Eval is our own harness.**
   - A benchmark registry and a two-stage sample→score store.
   - **One verifier (math-verify) for both reward and eval.**
   - **Compute-matched baselines as first-class protocols.** The literature
     (Debate-or-Vote; "Stop Overvaluing MAD"; the equal-inference-cost
     papers) says debate gains are mostly just voting.
   - Paired statistics.
   - An inspect_ai bridge is deferred.
4. **Credit assignment is pluggable.** The default is role-conditioned,
   leave-one-episode-out baselines. The field has converged on per-role
   grouping, RAE-EMA and Dr.MAS per-agent normalization, and warns against
   mixed-role baselines.
5. **Bonus.** `TinkerBackend` takes an explicit `base_url`. So open-source
   local Tinker servers (SkyRL `skyrl.tinker`, verl-tinker) remain a zero-code
   third backend option later. We don't depend on them.

## Inherited from scimt (keep)

- **Layering.**
  - `src/marli/` is the library.
  - `experiments/` is the lab notebook: results kept as-run, one directory per
    study, sub-study directories to avoid prior_coins-style bloat.
  - `examples/` is kept green by smoke tests.
  - `docs/wiki/` + `docs/sources/` is the curated layer. Adapt the schema and
    the ingest/query/lint workflows from `scimt/docs/wiki/CLAUDE.md`.
- **Handles.** Frozen dataclasses with JSON manifests beside the bytes, an
  `.at()` ad-hoc escape hatch, and manifests that nest their inputs so
  provenance chains. Pattern: `scimt/dataset.py`, `scimt/train/checkpoint.py`.
- **Sampler vs state split on `Checkpoint`.** Resume uses `state`, eval uses
  `sampler`, and `require_state()` guards the difference.
- **Config-first.**
  - Port `scimt/config.py`: OmegaConf `compose(cls, *yamls, overrides)`,
    `parse`, `save`.
  - Unknown keys raise `ValueError`.
  - Use Enum or str plus validation, not `Literal`/`frozenset`, which
    OmegaConf rejects. An M0 test covers this.
- **File-backed YAML registries.** `load_X`/`list_X`, name must equal the
  filename, unknown keys raise. Pattern: `scimt/model.py`. Named-function
  registries use `module:function`, never lambdas.
- **LLM client and judge.**
  - Port `scimt/utils/client.py` (`ChatClient`: OpenAI, Anthropic, OpenRouter,
    or any OpenAI-compatible endpoint, with a disk cache) and
    `scimt/utils/judge.py`, with an attribution header.
  - **Fix:** scimt's in-memory cache replays identical payloads. That would
    collapse Du-debate round 1, SC@k and GRPO groups into a single sample.
    APIPolicy must always pass a per-call `cache_salt`
    (seed:task:episode:agent:step:sample), and a test enforces it.
- **Eval rules.** Pure parsers plus a sync `aggregate`. Always report n and the
  lift against a baseline cell from the same harness.
- **Hygiene.**
  - Heavy imports are lazy, so `import marli` stays torch-free.
  - Unit tests are CPU-only and use fakes.
  - Extras are per stage.
  - GPU stacks live in `requirements/pod-*.txt`.
  - `runlog` records git provenance.
  - Training refuses a dirty tree unless `MARLI_ALLOW_DIRTY=1`.
  - `publish` goes to private HF.
- **Lessons to copy into `docs/sources/` at M0.** Sources are
  `/root/repos/scimt-prior-latmem/experiments/prior_latmem/LESSONS.md` and the
  docstrings in `git -C /root/repos/science-of-midtraining show
  f228dcf^:src/scimt/train/grpo.py`.
  - Count optimizer updates.
  - A pod is not storage.
  - Put timeouts around vLLM.
  - Log the fraction of zero-std groups (65% in scimt RLVR).
  - Truncation limits are relative to the budget.
  - Spend guards.
  - The fp32-logits OOM.

## Deliberate divergences from scimt

- **CLIs everywhere, as thin shims.** There is one `marli` entry point.
  - **Flow:** `argv → compose(Config, *yamls, dotted=overrides) →
    asyncio.run(VERB(cfg))`. Verbs stay importable async functions.
    argparse lives only in `src/marli/cli/`, enforced by a contract test.
  - **Output:** a required `--out DIR` (or `--out auto` →
    `runs/<verb>/<hash12>`).
  - **Stdout:** while a verb runs, stdout is redirected to stderr. It then
    prints exactly one JSON line: `{ok, kind, manifest, n, n_failed, usage,
    cost_usd, warnings}`.
  - **Exit codes** are distinct for config, hash mismatch, budget and backend
    errors.
- **Idempotency and resume.**
  - The config hash covers the canonical resolved config minus runtime-only
    fields, plus the sha256 of each input manifest's content.
  - An existing complete manifest with the same hash is a no-op. The same hash
    with incomplete output resumes by skipping finished keys. `--force` wipes.
    A different hash is a loud error.
  - Manifests are written last via atomic rename. Rows are appended with fsync,
    tolerating a torn final line. The output directory is locked.
- **Paths.**
  - A manifest's own bytes are recorded relative to the manifest.
  - Inputs are recorded as `{abs path, sha256}`.
  - `MARLI_DATA` is the root that lets outputs move between pod and dev box.
  - Manifests store *resolved* policy refs (`tinker://…`, base_url, renderer),
    never secrets.
- **Introspection.** `marli list <registry>`, `marli describe <verb>` (config
  schema and defaults), `marli inspect <manifest>` (provenance chain) and
  `marli status <out>` (reads `progress.json`). `docs/cli.md` is generated,
  and a test checks it is current.
- **CI from day one.** scimt has none.
- **Simpler `.gitignore`.** Ignore `**/out/`, `runs/` and caches; experiments
  commit `results/`.
- **`AGENTS.md`** is a symlink to `CLAUDE.md`, so Codex and Claude share the
  rules.

## Repo layout

```
CLAUDE.md  AGENTS.md->CLAUDE.md  README.md  pyproject.toml  uv.lock  .python-version
src/marli/
  config.py registry.py handles.py runlog.py tracking.py(wandb; off in CI) budget.py publish.py
  models/*.yaml model.py     # hf_id, renderer, on_tinker, local_ok, max_ctx, default_max_tokens, thinking, prices
  llm/{client,judge}.py
  policy/{base,refs,scripted,api,tinker,vllm}.py
  interact/{types,board,views,io,runner,seating}.py  interact/protocols/*.py  interact/configs/*.yaml
  tasks/*.yaml tasks/{loaders,prompts,verifiers,equivalence}.py
  rewards.py
  data/{build,filter}.py
  eval/{rollout,score,report,grid,stats}.py
  train/{types,datums,advantages,loop,checkpoint,export}.py
  train/backends/{base,fake,tinker}.py  train/backends/local/{learner,losses,backend,sync}.py
  serve/{vllm,_supervise}.py
  viewer/html.py
  cli/main.py
experiments/{README.md,_template/}  examples/  docs/{wiki,sources,plans,runbooks}/  tests/
requirements/pod-{train,vllm}.txt  scripts/pod_setup.sh  .github/workflows/ci.yml
.claude/skills/{marli-experiment,marli-on-pod}/SKILL.md
```

**Extras:**

| Extra | Contents |
|---|---|
| core | `httpx pyyaml omegaconf` |
| `tinker` | `tinker==0.30.1`, `tinker-cookbook==0.5.7`; torch comes from the CPU index on dev/CI; needed for renderers |
| `local` | torch, transformers (≤5.5.4, matching the cookbook), peft, optional liger-kernel; the pod-side learner |
| `eval` | `datasets`, `math-verify`, pebble |
| `wandb`, `hub`, `pods` | `pods` is bellhop-py |
| `dev` | pytest, pytest-asyncio, ruff |
| `all` | everything above |

- vLLM runs as a **separate supervised server** from its own venv
  (`requirements/pod-vllm.txt`, which owns its torch). This follows scimt's
  #209 carve-out.
- Pytest markers: `gpu`, `live` (network or spend) and `tinker` (needs the
  extra). Default CI runs neither the `live` nor the `gpu` tests.

## Core interfaces (sketch; Claude writes these as contract stubs before Codex implements)

```python
# policy/
@dataclass(frozen=True)
class Generation:
    text: str
    thinking: str | None
    prompt_ids: list[int] | None
    completion_ids: list[int] | None
    logprobs: list[float] | None
    termination: str                 # stop | length | malformed
    policy_id: str
    sampler_step: int | None
    usage: Usage

class Policy(Protocol):
    policy_id: str
    trainable: bool
    async def generate(self, messages, spec: ActionSpec, *, seed: int) -> Generation: ...
# ScriptedPolicy | APIPolicy(ChatClient) | TinkerPolicy(sampling_client, renderer) | VLLMPolicy(base_url, model, renderer)
# Refs: "tinker@<backend>:Qwen/Qwen3-8B", "ckpt:<dir>#step=N|final", "vllm:@server.json#<model>",
#       "api:openrouter/<model>", "scripted:<name>"
```

**Trainable-seat guard.** A trainable seat must sample at temperature 1,
top_p 1 and top_k −1. Both backends return raw logprobs regardless of sampling
params (tinker-feedback #148; vLLM `raw_logprobs`), so anything else would be
silently off-policy.

**VLLMPolicy token path.**
- Request: `prompt=list[int]` (from `renderer.build_generation_prompt(...).to_ints()`),
  `return_token_ids`, `logprobs=1`, `stop_token_ids` taken from the renderer
  (int stops only), `skip_special_tokens=false`, and explicit
  temperature/top_p/top_k/seed.
- Parse via `renderer.parse_response(ids) -> (Message, ParseTermination)`. Never
  use vLLM's `text`.
- The per-turn budget is `min(spec.max_tokens, max_ctx − len(prompt))`.

```python
# interact/
@dataclass(frozen=True)
class SeatSpec:
    agent: str
    role: str
    system: str | None               # fixed per episode -> stable prefix

@dataclass(frozen=True)
class Msg:
    sender: str
    kind: str                        # task | moderator | utterance | env | answer
    content: str
    visible_to: tuple[str, ...] | None
    step: int
    meta: dict

@dataclass
class ActionSpec:
    max_tokens: int | None = None    # default from model registry
    n: int = 1
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: list[str] | None = None
    prefill: str | None = None
    parser: str = "free"
    parse_retries: int = 0

class Protocol(ABC):                 # registered by name; configured via interact/configs/*.yaml
    def seats(self) -> list[SeatSpec]: ...
    async def run(self, task: Task, io: TurnIO) -> Outcome: ...

class TurnIO:
    async def turn(self, agent, spec, *, instruction=None, post=True) -> Turn: ...
    async def turns(self, specs: dict[str, ActionSpec], *, instruction=None, post=True) -> dict[str, Turn]: ...
        # snapshot all views -> gather -> post in SEAT order (never completion order)
    def post(self, msg: Msg) -> None: ...
    def view(self, agent) -> list[Message]: ...
    rng: random.Random
    usage: Usage

@dataclass(frozen=True)
class Seating:
    policies: dict[str, str]         # name -> ref
    by_role: dict[str, str]          # role -> policy name
    learners: tuple[str, ...]        # v1 validates len <= 1
```

**Views** (`views.py`, pluggable):
- `full_board`: the seat's own turns are rendered as `assistant`. Everything
  visible since its last turn, plus the instruction, is merged into one `user`
  message with speaker tags, which guarantees alternation.
- `latest_only`: for future game adapters.
- Only the text content is posted to the board; thinking stays private on the
  `Turn`.
- Per-seat `history: rerender | tokens`. `rerender` is template-faithful and is
  the default for thinking renderers. `tokens` appends the recorded tokens,
  giving an exact prefix so datums can be merged.
- Context overflow: `on_overflow: fail | drop_oldest_rounds`.

**Records.**
- `Turn` = `{agent, role, step, gen, parsed, reward, trainable, truncated,
  seed, latency_s}`.
- `Episode` = `{group_id, episode_idx, task_id, protocol, config_hash,
  backend, seats, events, turns, outcome, agent_rewards: {agent: {component:
  float}}, usage_by_policy, ok, errors}`.
- `Outcome` = `{final_answer, answers_by_agent_round, aggregation, votes}`.

**Runner** (`run_episode`, `run_group`):
- One semaphore per policy.
- Per-call seed = hash(run_seed, task, episode, agent, step, sample).
- Transient errors are retried inside the policy. A failed episode is marked
  `ok=False`, and eval scores it as wrong.
- `budget.max_usd` is checked before each call, against a dated price table in
  the model registry.
- `record_tokens` is off for eval and on for RL. RL tokens go to a per-step
  npz sidecar.

**Protocols (v1).**
- `single`.
- `self_consistency` (SC@k).
- `self_refine`.
- `debate`: Du et al., N agents × R rounds, simultaneous; aggregation by
  majority, judge or last.
- `solver_critic`: solve → critique → revise, repeated R times.
- `round_robin`.

Majority voting groups answers **by verifier equivalence** (math-verify,
canonical MCQ letter) with a seeded tie-break. A judge is just another seat
with role `judge`.

```python
# train/ — backend-agnostic algorithm loop over Tinker-shaped primitives
@dataclass
class Datum:                          # torch-free
    model_input: list[int]
    target_tokens: list[int]
    logprobs: list[float]
    advantages: list[float]
    mask: list[float]                 # mask is a sidecar only

class TrainerBackend(Protocol):       # FakeBackend | TinkerBackend | LocalBackend
    capabilities: Capabilities        # losses, max_learners, lora/full, kl
    async def forward_backward(self, learner, datums, loss: LossSpec) -> FwdBwdOut: ...   # per-datum train logprobs, loss:sum
    async def optim_step(self, learner, adam: AdamParams) -> dict: ...                     # betas (0.9, 0.95), grad_clip
    async def sync_sampler(self, learner, step) -> Policy: ...                             # Tinker: save_weights_and_get_sampling_client; local: vLLM LoRA hot-swap
    async def save(self, learner, name, kind: "state" | "sampler" | "both") -> CkptPaths: ...
    async def load(self, learner, state_path, with_optimizer: bool) -> None: ...
```

**Datum construction** (`datums.py`), per (episode, trainable agent). Datums
are never merged across agents.
- Walk the agent's turns. When an observation prefix-extends the accumulated
  sequence, append only the delta; otherwise flush and start a new sequence.
- Emit `Datum(model_input=acc[:-1], target_tokens=acc[1:], logprobs=lp[1:],
  advantages=adv[1:])`. Positions that are not the agent's own actions get
  logprob 0 and advantage 0.
- The mask is stripped before sending. Tinker RL losses accept only
  `target_tokens`, `logprobs` and `advantages`, and reject `weights`.
- Golden test: in the scalar-advantage case this equals the cookbook's
  `trajectory_to_data`.

**Loss semantics** are identical on both backends:
- `importance_sampling = -(exp(lp_θ − lp_q)·A).sum()`. Also `ppo` (clip),
  `cispo`, and `cross_entropy` (reserved for SFT).
- The loss is **sum-reduced**. `train.adv_norm: none | batch_tokens |
  per_sequence` rescales advantages, so protocols with different token volumes
  (single vs debate) are comparable. The effective token count is logged.

**Advantages** (`advantages.py` registry):
- The group key is (task_id, protocol hash, seating hash, role).
- Estimators:
  - `group_mean` (default): leave-one-episode-out, no std; with symmetric
    roles, identical copies are deduplicated.
  - `group_zscore`.
  - `rae_ema`: EMA per (protocol, role), γ=0.95, with its state persisted in
    loop_state.
  - `per_agent_zscore`.
- Credit is either `episode` (broadcast; the default) or `turn`. Turn credit
  requires fixed-round protocols and uses a per-step baseline.
- Rewards (`rewards.py`) are a weighted sum of logged components:
  `correct_team`, `correct_individual`, `delta_correct` (for critics) and
  `format`.

**RL step** (`loop.py`, synchronous and on-policy in v1):
1. Sample B tasks and run G episodes of each.
2. Compute rewards, then filter zero-std groups per (task, role).
3. Compute advantages and build datums.
4. For each learner: `forward_backward` + `optim_step`, submitted
   back-to-back.
5. If there are no informative datums, skip `optim_step` (Adam momentum would
   otherwise still move the weights) and count the skip.
6. `sync_sampler`.
7. Log:
   - reward and accuracy per role;
   - zero-std fraction;
   - `kl_sample_train` (cookbook metric);
   - effective tokens;
   - throughput;
   - `$`.

Around each step:
- `progress.json` is written every step.
- Every `save_every` steps, a `Checkpoint` row is appended (state + sampler).
- Resume restores weights, optimizer, data cursor, RNG and estimator state.
- An in-loop mini-eval runs on a held-out TaskSet.

Config shape: `learners: {name: {model, lora_rank, lr, init}}`. v1 requires
exactly one learner.

**TinkerBackend.**
- `backend: {kind: tinker, base_url}` is required and always passed
  explicitly. It raises if `$TINKER_BASE_URL` disagrees with the config, and
  the resolved URL goes into every manifest.
- Uses `create_lora_training_client(base_model, rank, train_unembed=…)`.
- Saves with `save_state(name, ttl_seconds=None)` and
  `save_weights_for_sampler(name, ttl_seconds=None)`. The cookbook's TTLs
  would otherwise leave manifests dangling.
- Resumes via `create_training_client_from_state_with_optimizer`.
- Frozen seats use `create_sampling_client(base_model=…)`.
- Capability check: `ppo`/`cispo` are allowed only where supported.

**LocalBackend** (v1: LoRA, one learner GPU).
- **`learner.py`:**
  - HF + PEFT LoRA on attention + MLP. Parity runs set Tinker's
    `train_unembed=False` to match.
  - bf16 with gradient checkpointing.
  - **Chunked logprobs**, never materializing `[T×V]` in fp32.
  - Microbatching by token budget.
  - AdamW with betas (0.9, 0.95), eps 1e-8, and gradient clipping.
- **`sync.py`:** after each optimizer step, `save_pretrained` the adapter to
  `$MARLI_SCRATCH/sampler/step_N`, then call `POST /v1/load_lora_adapter
  {lora_name: <learner>, lora_path, load_inplace: true}`. Frozen snapshots are
  loaded under distinct names, so "vs past self" comes for free.
- **Checkpoints.** The sampler is a PEFT adapter directory (servable by vLLM
  and loadable by HF). The state is the adapter plus optimizer, RNG and
  loop_state.
- **GPU placement.** Default is 2 GPUs (vLLM on GPU0, learner on GPU1). A
  1-GPU colocated mode (low vLLM `gpu_memory_utilization`) covers models
  ≤4B.

**vLLM server** (`serve vllm`):
- Flags: `--enable-lora --max-lora-rank --max-loras --generation-config vllm`
  (without this, Qwen3's T=0.6/top_k=20 silently applies), `--logprobs-mode
  raw_logprobs`, and `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
- Runs supervised, or with `--detach`, which writes `server.json` (pid,
  base_url, model, log); `serve status|stop` manage it.
- Backends accept `@server.json`.

**Handles:**

| Handle | Bytes | Manifest |
|---|---|---|
| `TaskSet` | `tasks.jsonl` | `taskset.json` |
| `EpisodeSet` | `episodes.jsonl` | `episodes.json` |
| `Scores` | `scores.jsonl` | `scores.json` |
| `Report` | `results.jsonl` + `RESULTS.md` | — |
| `Checkpoint` | — | `checkpoint.json`: backend, base_url, base_model, **renderer**, sampler, state, lora, train cfg, step; per-learner under `<out>/learners/<name>/`, with `checkpoints.jsonl` rows |

## CLI verbs (v1)

| Verb | In → Out |
|---|---|
| `marli list` · `describe` · `inspect` · `status` | introspection |
| `marli data build <benchmark> [split= max_n= seed=]` | → TaskSet |
| `marli data filter --episodes … lo= hi=` | → TaskSet of informative tasks (pass rate by verifier) |
| `marli eval rollout --tasks … protocol=<cfg> seating.policies.x=<ref> …` | → EpisodeSet (sample store, resumable) |
| `marli eval score --episodes …` | → Scores |
| `marli eval report --scores … baseline=<cell>` | → Report |
| `marli eval grid <grid.yaml>` | benchmarks × protocols × policies → all of the above |
| `marli train rl <cfg.yaml> backend.kind=local\|tinker …` | → Checkpoint(s); resumes on same hash |
| `marli export --ckpt …` · `marli publish --ckpt … repo=` | Tinker → PEFT adapter (cookbook `weights.download` + `build_lora_adapter`); → private HF |
| `marli serve vllm model= loras= [--detach]` · `serve status\|stop` | vLLM server + `server.json` |
| `marli view --episodes …` | → episodes.html (global + per-agent views) |

**Report contents:**
- accuracy with n and a Wilson CI;
- **paired** lift vs the baseline (McNemar + paired bootstrap);
- compute: generated tokens (primary), calls, prefill tokens;
- debate diagnostics: per-round accuracy, flip matrix, agreement rate;
- AIME as avg@k (k≥8) with a per-problem bootstrap;
- `n_failed`.

**Benchmarks (v1):**
- `gsm8k` (train/test);
- `math_train` (hendrycks_math train+test minus MATH-500, ≈12k);
- `math500`;
- `aime_2025` and `aime_2026` (MathArena; verify the licence);
- `gpqa_diamond` (gated);
- `mmlu_pro_2k` (seeded, stratified).

**Models (v1):**
- `qwen3_8b`: on Tinker and local; the parity model.
- `qwen3_4b_instruct_2507`: local only; fast dev.
- `qwen3_1_7b`: local only; smoke tests.
- `qwen3_5_4b`: Tinker smoke (verify the renderer).
- `qwen3_6_35b_a3b`: Tinker MoE.

## Milestones

**Execution** (every milestone is a PR with green CI, run via
`codex-driven-development`):
1. Claude pre-stages dependencies (`uv add`/`uv sync`, fixtures). Codex's
   sandbox has no network.
2. For the judgment-heavy modules, Claude writes contract stubs and golden
   tests first: `interact/types`/`io`, `datums`, `advantages`, `losses`,
   `learner`.
3. Codex implements each task.
4. Claude subagents review for spec compliance and then quality, re-running
   the tests themselves.
5. Claude runs the paid or GPU checks (pod/Tinker/API). Each has a `max_usd`
   guard, and I confirm with Sid before starting any pod.

**M0: Scaffold and contracts** (CPU)
- `git init`; pyproject, extras, markers and ruff; `.gitignore`; lockfile.
- `CLAUDE.md` with the conventions above, `AGENTS.md` symlink, README.
- docs skeleton: wiki schema, index and log; the lessons in `docs/sources/`;
  this plan in `docs/plans/`.
- `config`, `registry`, `handles` (hashing, atomic writes, locks, resume) and
  `runlog`.
- CLI skeleton (`list`/`describe`/`inspect`/`status`, the stdout contract,
  exit codes, `--out auto`).
- Contract tests: argparse only in `cli/`; `import marli` pulls in no
  torch/tinker/vllm/datasets.
- CI (ruff + `pytest -m "not gpu and not live"`).
- `gh repo create SidBaines/marl-investigations --private`, push, open the PR.

**M1: Interaction layer and policies** (CPU + a ≤$1 live smoke)
- Port the client and judge with the cache-salt fix and the budget guard;
  model registry.
- Policy base, refs, and the Scripted/API/Tinker/VLLM policies with the
  trainable-seat guard.
- `interact` core: types, board, views, io, runner, seating.
- The 6 protocols plus the YAML registry; equivalence-voting hook.
- Episode JSONL and `marli view`.
- Tests:
  - visibility and seat-order posting under shuffled completion order;
  - simultaneity;
  - **re-render invariance** (rebuilding a view from `events` reproduces the
    recorded prompt ids);
  - seeded determinism (byte-identical episodes);
  - failures;
  - cache salt.
- Live: debate on 10 GSM8K items with an `api:openrouter/...` model. Pin the
  OpenRouter provider (`allow_fallbacks=false`) and set reasoning effort
  explicitly.

**M2: Tasks, eval, data, and vLLM serving** (CPU + first pod, ≤$10)
- Task registry, loaders (test fixtures only, no network in tests) and
  prompts.
- Verifiers:
  - math-verify runs in a **spawn/forkserver process pool created before any
    Tinker client**, with a hard kill (pebble) on timeout, because SIGALRM is
    main-thread only and sympy can hang;
  - MCQ and numeric verifiers;
  - golden answer-variant tests.
- Rewards registry.
- `eval rollout`/`score`/`report`/`grid` and `stats`.
- `data build`/`filter`.
- A SIGKILL-resume test for rollout.
- GPQA/AIME hygiene: never commit question or transcript text, and a canary
  lint test enforces it. Map the personal HF token to `HF_TOKEN` for gated
  sets.
- `serve vllm`; `requirements/pod-vllm.txt`; `scripts/pod_setup.sh` (HF cache
  on the volume); `docs/runbooks/pods.md`; the skills `marli-on-pod` (built on
  `runpod-spinup`'s `create-pod-cuda.sh` + preflight) and `marli-experiment`
  (how to chain verbs).
- **Pod check** (1×H100):
  - serve Qwen3-8B;
  - token-path test: vLLM prompt ids equal the renderer's ids, the stop token
    is included, and the logprob lengths match;
  - `eval grid` pilot: gsm8k 100 × {single, sc@4, debate_3x2}.

**M3: Training core and the Tinker adapter** (CPU + a ≤$3 Tinker smoke)
- `train/types`, `datums`, `advantages`, `rewards` wiring, the backend
  protocol and `FakeBackend`.
- `loop.py`, `checkpoint`, `TinkerBackend`, the `train rl` CLI, wandb
  (personal entity, disabled in CI).
- Tests:
  - datum invariants: lengths; no `mask`/`weights` sent; advantage 0 where the
    mask is 0; golden vs the cookbook (`importorskip`);
  - no datum built for a non-learner seat;
  - leave-one-out and RAE goldens;
  - empty step skips `optim_step`;
  - resume;
  - backend URL guard;
  - CPU CLI end-to-end run from a different cwd, plus a hash mismatch on an
    input change.
- Live: 3 steps of `debate_2x2` on Qwen3-8B via Tinker. Check that:
  - the datums are accepted;
  - `kl_sample_train` ≈ 0 at step 0;
  - named, TTL-free paths come back;
  - a frozen base-model seat works;
  - resume reproduces the next-step loss.

The Tinker adapter lands before the local learner because it is thin and
gives a known-correct reference to validate our learner against. M3 and M4
can swap if Sid prefers.

**M4: Local learner backend** (pod, ≤$25)
- `losses.py` (IS/PPO/CISPO/CE, sum-reduced, with a CPU unit test against a
  reference implementation on a tiny random model).
- `learner.py`, `sync.py` (LoRA hot-swap), LocalBackend checkpoints and
  resume, GPU placement config.
- `export`/`publish`; `requirements/pod-train.txt`.
- **Pod check** (2×H100):
  - **logprob agreement**: the learner re-scores vLLM samples, and the IS
    ratio is ≈1 with a small per-token gap;
  - a 20-step run on Qwen3-4B-Instruct-2507 gsm8k, single + debate;
  - kill and resume.

**M5: First experiments** (the science, and end-to-end acceptance)
- `experiments/<date>_coop_baselines/`: Qwen3-8B (+1 API model) × 5
  benchmarks × 6 protocols, compute-matched, on pod vLLM. This is the
  debate-vs-vote baseline before any training.
- `experiments/<date>_backend_parity/`:
  - setup: Qwen3-8B LoRA r32 on gsm8k/math_train, `single` + `debate_2x2`
    self-play, 30 steps, 3 seeds per backend (≈$30 Tinker + ≈$40 pod);
  - acceptance: the backend difference is within the seed spread, the
    logprob gap is small, and both backends resume.
- Ingest the durable findings into the wiki.

**Designed for, deferred:**
- TextArena game adapter (competitive / mixed-motive).
- Co-trained separate learners.
- Multi-GPU FSDP2 and full fine-tuning locally.
- Async off-policy.
- SFT (`data sft`/`train sft`; cross_entropy is already a backend loss).
- inspect_ai bridge.
- LLM-judge rewards (oversight games).
- Synthetic task generation.
- bellhop-based `marli pod run`.
- SkyRL/verl-tinker as a third backend.

## Verification

- **CI (every PR):** `uv run --extra dev --extra eval pytest -m "not gpu and
  not live" -q` plus `ruff check`. This includes a CPU CLI end-to-end chain
  driven purely by the printed manifest paths:
  1. `data build` (fixture);
  2. `eval rollout` (debate, `scripted:` policies);
  3. `eval score`;
  4. `eval report`;
  5. `train rl` for 2 steps (FakeBackend);
  6. `eval rollout` with the `ckpt:` ref.
- **Live smokes, run by Claude with a `max_usd` guard:** API debate (M1),
  Tinker 3-step RL with resume (M3).
- **GPU:** pod checks in M2 (token path) and M4 (logprob agreement, run,
  resume); M5 parity acceptance across 3 seeds per backend.
- **Rough spend for the whole plan:** Tinker ≈ $35, RunPod ≈ $75, APIs ≈ $5.
  Pods are torn down after each check, and checkpoints are synced to private
  HF first.
