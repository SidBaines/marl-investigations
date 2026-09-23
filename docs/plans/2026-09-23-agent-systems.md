# Plan: core infra for `marl-investigations` (RL on multi-agent LLM systems)

## Context

**Goal.** A private repo (`SidBaines/marl-investigations`, owned by Sid) for
studying and RL-training **LLM agent systems**. It follows the
science-of-midtraining (scimt) layout: a general-purpose library
(`src/marli/`), an `experiments/` notebook whose studies chain library
verbs, and a docs wiki. Every component is also a CLI verb, so Claude/Codex
agents can chain them.

**What we study.** Sid redirected on 2026-09-23 from debate-style turn-taking
to agent systems:
1. **Coordinator-free swarms.** N agents run in parallel; each writes its own
   scratchpad in a shared space and reads the others'.
2. **Coordinator + workers.** An orchestrator spawns workers dynamically via a
   tool and composes their reports.
3. **Multi-session single agent.** Repeated sessions with a fresh context;
   state carries over via **compaction** and/or **notes it leaves itself**.

Debate and self-consistency stay as presets and baselines. Environments are
**hard math** and **agentic coding** (function-level first, then repo-level
SWE).

**Training axes.**
- Shared LoRA vs **one LoRA per role**.
- Reward to **all agents vs a subset**.
- In multi-session, reward to the **last session only vs all sessions**.
  No paper ablates this directly, so it is a gap we can fill.

**Decisions (Sid).**
- **Swarm scheduling:** lockstep by default, async as an option.
- **Swarm answer:** vote, finalizer, or `oracle_any` (eval only).
  Delivery defaults to **notify + pull**.
- **Coordinator:** workers are spawned dynamically.
- **Sessions:** end on the token budget or on `end_session()`.
- **Exhaustion:** default **force_final**.
- **SFT teacher:** a same-family model on Tinker (large Qwen3.5/3.6 or
  gpt-oss-120b) with visible thinking.
- **Models:** **Qwen3.5 primary** (validate the local LoRA setup early), plus
  **gpt-oss with native Harmony tools**. Qwen3-8B is the fallback.
- **Training backends:** our own Tinker-shaped **local learner** (PEFT LoRA,
  1 learner GPU + vLLM) and a **Tinker adapter**, sharing **one algorithm
  loop**. Big MoE runs on Tinker.
- **Sandboxes:** run **locally, wherever the loop runs**. That means Docker on
  a laptop or AWS box, and a restricted subprocess on RunPod.
- **Naming:** package and CLI are `marli`.
- **GitHub:** private repo, one PR per milestone, CI.
- **Build mode:** Codex builds, Claude reviews. Codex runs with
  `--dangerously-bypass-approvals-and-sandbox`, which Sid authorized; bwrap
  can't run in this container. Its environment is scrubbed of API/HF keys
  (`env -i PATH HOME …`).

**Where things can run.** A remote-learner service is deferred.

| Where the loop runs | Backend | Math | code_fn | SWE |
|---|---|---|---|---|
| Laptop / AWS box (Docker) | Tinker | ✓ | ✓ docker | ✓ docker (x86 images; slow under Apple-Silicon emulation) |
| RunPod pod (no Docker/userns) | local learner | ✓ | ✓ subprocess | ✗ |
| AWS GPU box (Docker) | local learner | ✓ | ✓ | ✓ |

**State.**
- M0 pre-stage is committed on `m0-scaffold`: pyproject, uv.lock,
  CLAUDE.md, lesson sources, and the superseded debate plan in
  `docs/plans/2026-09-23-core-infra.md`.
- No Codex task has landed yet.
- The worktree `/workspace/marl-wt/m0-docs` exists and is unused.

## Key design decisions (research-backed)

1. **Agents are token-level, tool-using coroutines over a shared workspace.**
   - A protocol is orchestration: roles, tools, permissions, limits, how
     agents start, how the answer is formed.
   - **Every LLM call is recorded** with its exact ids.
   - Invariant: **datums use exactly the ids the policy saw.** This follows
     SUPO, MemAgent, Delethink, verifiers v1 and prime-rl. It needs no custom
     attention masks, which Tinker lacks.
2. **Context lineages and segments.**
   - Token-capable seats keep an append-only buffer. Sampled ids are appended
     verbatim; new messages are rendered as a *delta*.
   - Resets start a new segment. Reset reasons: compaction, session, spawn,
     and rerender (fallback).
   - Result: one datum per segment.
   - Renderers that fail delta parity use **one segment per call**. This is
     chosen per renderer at config time.
3. **Native tool formats.** tinker-cookbook renderers provide them:
   - Qwen3.5 uses an XML format: `<tool_call><function=…>`.
   - gpt-oss uses Harmony `functions.*` on the commentary channel.
   - Qwen3 uses JSON inside `<tool_call>`, with
     `strip_thinking_from_history=False` and HF-style grouping of
     consecutive tool responses.
   - A text protocol is the fallback for renderers without tools.
4. **Multi-learner in v1.**
   - Config: `learners` plus `seating.by_role: role → learner | frozen ref`.
   - **Tinker:** one TrainingClient per learner. Submit every learner's
     fwd/bwd and optim before awaiting.
   - **Local:** PEFT multi-adapter. Per adapter: `set_adapter` → fwd/bwd →
     clip → AdamW. No mixed-adapter training batches.
   - **vLLM:** `--max-loras ≥ learners + snapshots`, `--max-lora-rank` set to
     the real rank, and **versioned adapter names**. This avoids the sticky
     `load_inplace` flag and a stale prefix cache.
5. **Credit assignment is a validated, config-driven ablation surface.** The
   default broadcasts the mean-centred outcome advantage to every trained unit.
6. **Compute-matched evaluation by construction.** Record total generated
   tokens, uncached prompt tokens, calls, critical path, and peak context.
   Baselines are a single agent, N-independent + vote (the swarm with
   visibility off), and sequential multi-session.
7. **Kept from the approved design and the first review:**
   - scimt conventions: handles/manifests, registries, config-first, lazy
     imports, sampler/state split, ChatClient **with a per-call
     cache_salt**, runlog;
   - the CLI contract: one JSON line, idempotent/resumable run dirs, exit
     codes, `describe`/`inspect`/`status`;
   - trainable seats sample at T=1 with raw logprobs;
   - Tinker datum keys and sum-reduced loss;
   - TTL-free named checkpoints;
   - explicit `base_url`;
   - vLLM `--generation-config vllm`;
   - math-verify in a spawn pool with hard kill;
   - CI from day one; `AGENTS.md` → `CLAUDE.md`.

## Interaction layer contracts (`marli.interact`; Claude writes these, Codex implements)

**Records** (`interact/types.py`, JSON round-trip):

```python
Call(call_id, episode_id, agent_id, role, policy_id, policy_version, segment_id, prompt_len,  # prompt = buf[:prompt_len]
     completion_ids, logprobs, termination: stop|length|malformed|budget|ctx|error,
     purpose: act|compact|carry|final|report, forced: bool, tool_calls, reads: list[WorkspaceRead],
     tick: int|None, seq: int, usage, timing)            # timing excluded from equality
SegmentInfo(segment_id, agent_id, session_idx, start_reason: start|compaction|session|spawn|rerender,
            carry_from, renderer, tokenizer_sha)
AgentInfo(agent_id, role, policy_id, parent, seat_key)  # deterministic ids e.g. "coord0", "coord0/w3", "peer2/s1"
Episode(episode_id, group_id, task_id, protocol, config_hash, backend, agents, segments, calls,
        events (async seq log), workspace_log, outcome, grades: {agent_id: {component: float}},
        limits_hit, metrics, replayable, ok, errors)      # token buffers go to an npz sidecar written before the row
Outcome(final_answer, submissions: {agent_id: answer}, aggregation, votes)
```

- **Raw grades** are stored at rollout. Rewards are computed in the training
  loop, because schedules depend on the step.
- **Graders** use `env.grade(submission, sandbox)`. Scratchpads and notes are
  structurally unreachable from them.
- `grade_individual` (on by default for swarms) grades each peer's own
  submission.

**Roles, seating, policies.**
- `Protocol.roles() -> [RoleSpec(role, tools, permissions, limits, count:
  int|"dynamic")]`.
- Seating maps every role, including `worker`/`finalizer`, to a learner or a
  frozen ref.
- Policies:
  - token-level: `sample(prompt_ids, SamplingSpec, seed) -> Sample`;
  - API seats: `ChatPolicy.chat(msgs)`. They keep a MessageLog, are never
    trainable, and their usage is tagged with the tokenizer, so token counts
    aren't compared across tokenizers.

**Scheduler = a gate, not a driver** (`interact/scheduler.py`).
- Agents are long-lived coroutines.
- API: `register(agent, seat_key)`, `await turn(agent) -> Ticket(tick, view,
  pushes)`, `tool_lock(agent, shared)`, `block/unblock/done`.
- **Lockstep:**
  - Tick t closes when every registered agent is waiting, blocked, or done.
  - Buffered workspace writes commit in `seat_key` order.
  - Shared-state tools (a shared sandbox's bash, workspace writes) run in
    seat order after the tick's generations. Pure tools run concurrently.
  - `spawn_workers` runs in the coordinator's tool phase. It registers the
    workers, whose first call is at t+1, and blocks the coordinator until
    every worker is done.
  - `Call.tick` is one global int.
  - v1 limits: `max_depth=1`, at most one spawn per turn, and tool calls
    within a turn run in order. `submit`, `end_session` and `return_report`
    end the agent's turn; later calls get an error result.
  - Sandbox filesystem effects are visible in seat order within a tick
    (documented).
- **Async:**
  - Free-running; writes commit immediately; there is an optional
    `wait_for_update(timeout)` tool.
  - Each agent has its own fixed budget share.
  - A global event log records `seq`, the version vector read, calls, tools,
    commits and pushes. `replayable=False`.
- **Determinism:** byte-identical episodes are guaranteed only with
  ScriptedPolicy. vLLM batching isn't batch-invariant. Async tests assert
  invariants using an injectable Clock.

**Limits and exhaustion** (`interact/limits.py`; separate from the `$` spend
guard in `budget.py`).

- **Limit levels:**
  - `call.max_tokens`
  - `agent.{max_gen_tokens, max_calls, final_reserve}`
  - `session.{max_gen_tokens, carry_reserve}`
  - `episode.{max_gen_tokens, max_ticks, max_wall_s}`
  - `worker.{…}`
  - `spawn.{max_per_call, max_total, max_depth=1}`
  - `ctx.max_ctx`, which must be ≤ both the model's and the backend's
    `max_seq_len`.
- **Per-call allocation:** `max_tokens = min(call cap, agent_rem −
  final_reserve, episode share (lockstep: rem/active), max_ctx −
  prompt_len)`.
- **Spawn reservation:** a spawn reserves `k × worker.max_gen_tokens` up
  front. If that doesn't fit, it returns an error that says how many workers
  are affordable.
- **`on_exhaust: force_final` (default)** makes one call with
  `final_reserve`.
  - It injects an instruction plus a prefilled submit-tool opener. The
    opener counts as observation (mask 0), and the call is marked
    `forced=True`.
  - A worker hitting its limit returns `[worker i: no report]`; that never
    masks the episode.
- **`on_no_tool_call`:** `nudge | end_agent | final_text_as_answer`.

**Delta rendering** (`render/`).

```python
class DeltaRenderer(Protocol):
    supports_delta: bool      # qwen3 True; qwen3_5/gpt_oss False until parity passes
    def initial(self, system, tools, msgs) -> list[int]
    def continuation(self, last_term, new_msgs) -> list[int]
    def parse(self, completion_ids) -> ParsedTurn
```

- `continuation` details:
  - a `length` termination gets a forced close (mask 0);
  - consecutive tool results go in one user block;
  - unparsed calls are kept verbatim and answered with an error result
    (with a synthetic id where needed).
- `parse` wraps the cookbook's `parse_response`.
- Parity tests run against HF `apply_chat_template` (single-query ReAct) and
  against the cookbook with `strip_thinking_from_history=False` (interleaved
  pushes).
- The CPU tests use a character-level `render/fake.py`.

**Delivery of others' state** (config `delivery`):
- `notify` (default): push a compact index each turn (writer, version,
  tokens, first line); agents `read_scratchpad(agent, version|latest)` to
  pull contents.
- `push`: new writes since the agent last saw them, appended as one user
  message. In lockstep they come from the tick-start snapshot; in async, from
  live state. Capped.
- `pull`: tools only.
- `view: latest | full | last_k` shapes the payload.
- Agents see their own writes immediately. The buffer is append-only;
  history is never rewritten.

**Context managers** (`interact/context.py`):
- Kinds: `none | compaction(threshold) | notes(cap) | tail(m) |
  session-carry`.
- **Compaction** is a policy Call with `purpose=compact`, appended to the
  ending segment. The summary is trained with that segment, SUPO-style; the
  last action/observation pair is dropped. The new segment is
  `initial(system, task, carry)`.
- **Tail** splices in the last m ids as prefill (observation tokens).
- **Session budget exhaustion** cuts the session, then forces a carry call
  from `carry_reserve`. The session budget counts compaction calls.
- Notes and the sandbox persist across sessions. Carry is capped.
- Validation: `threshold ≤ max_ctx − compact_reserve`.

**Protocols (v1):**

| Protocol | Mechanics | Answer |
|---|---|---|
| `single` | ReAct loop with env tools | own submit |
| `multi_session` | S sessions; carry = `compaction` \| `notes` \| `both` \| `tail`; `submit` ends the episode | submit |
| `swarm` | N peers; own scratchpad, others via `delivery`/`view`; lockstep \| async. A peer is done on `submit` (no resubmit) or after `force_final`. The episode ends when all peers are done, at `max_ticks`, or when the budget runs out (then `force_final` for everyone still active). Optional `stop_on_consensus: k`. | `vote` (verifier equivalence, None excluded, seeded tie-break) \| `finalizer: separate\|peer0` · `oracle_any` (eval only) |
| `coordinator` | `spawn_workers([{task, context}])`. Workers get a fresh context with their subtask plus a recap of the query, and must `return_report`. `worker_sandbox: shared`. | coordinator submit |
| presets | `independent` (swarm, `delivery=pull`, no read tools → SC@N + vote) · `debate` (swarm: `tools=[]`, `auto_publish=final_text`, `delivery=push`, `max_ticks=R`) · `self_refine` (multi_session + notes) | — |

## Environments and sandboxes

**`Sandbox` interface** (async): `start`, `exec(cmd, timeout)`,
`read/write_file`, `reset`, `close`. Output is truncated. There is **one
sandbox per episode**, and every call is a stateless `bash -lc` from the
workspace root. Grading happens in a fresh directory that holds only the
solution and the hidden tests.

- **`docker`** (laptop / AWS): `--network none`, cpu/mem/pids limits,
  per-task or per-repo images, a warm pool, teardown on exit.
- **`subprocess`** (pods / dev boxes):
  - one directory and one uid per episode, from a uid pool;
  - `PR_SET_NO_NEW_PRIVS`;
  - rlimits: AS, FSIZE, NOFILE, CPU, NPROC;
  - seccomp denying non-AF_UNIX sockets, ptrace and mount;
  - Landlock ABI ≥ 4 for filesystem and TCP (refuse to start below 4);
  - a scrubbed environment;
  - a killpg teardown of that uid.

  Hidden tests never sit on the box outside the grader directory. A remote
  `execd` server with a bearer token over an SSH tunnel is deferred.

**Environments:**
- **Math.**
  - Eval: AIME 2025/2026, HMMT Feb 2025/2026 (MathArena, CC BY-NC-SA:
    internal only, never commit text), BeyondAIME (CC0), IMO-AnswerBench.
  - Train: DeepMath-103K, POLARIS-53K, DAPO-17k (deduplicated).
  - `data filter` recomputes difficulty with our own policy (keep
    0 < pass@k < 1) and n-gram decontaminates.
  - Optional `python` tool running in the sandbox.
- **`code_fn`** (function-level agentic coding).
  - Workspace: `problem.md`, `examples/`, `solution.py`. Tools: `bash` +
    `submit`.
  - Reward: hidden tests (all-pass, with the pass fraction logged). Mirrors
    tinker-cookbook `code_rl`.
  - Train on DeepCoder-Preview; evaluate on the LiveCodeBench v6 date window.
- **`swe`** (later; needs Docker).
  - Train on SWE-smith-py (≈250 per-repo images). Evaluate on SWE-bench
    Verified (Epoch images) plus a SWE-rebench fresh window.
  - A mini-swe-agent-style loop: bash only, 60 s per command, 10k-character
    truncation, 30–50 steps.
  - Hidden tests are restored before grading.
- **gpt-oss stretch goal:** map our tools to its built-in `python`/`container`
  namespaces. It drops about 30 points off its native harness.

## Training (`marli.train`)

**Datums.**
- Built per (agent, segment):
  - `model_input=buf[:-1]`, `target_tokens=buf[1:]`;
  - logprobs and advantage on that agent's own action tokens only;
  - 0 on tool results, pushes, forced closes, prefill and carry.
- Routed to learners via seating. The mask is a sidecar only.

**Credit pipeline** (`train/credit.py`; the docstring and golden tests are the
spec). The steps run in this order:
1. Drop `ok=False` episodes; drop groups with fewer than 2 episodes.
2. Rewards: `r_{e,a}` from grades.
   - `reward_target` is a per-role map, e.g. `{peer: individual, finalizer:
     team, coordinator: team, worker: team}`.
   - `mix(α) = α·team + (1−α)·own`.
   - `aux_rewards` `{name, weight, schedule}`, annealed and capped.
3. Overlong flags: `filters.overlong: {mode: none|mask_no_answer|mask_forced,
   scope: episode|agent}`.
4. Baselines:
   - `episode`: leave-one-out mean of R over the group.
   - `role` (default): leave-one-out by episode, episode-weighted. For each
     other episode containing the role, take the mean of that role's rewards
     there. Same-episode peers are excluded. If no other episode has the
     role, A=0 and a counter increments. `unit: instance` is an option.
   - `rae`: an EMA per (protocol_hash, role) with γ=0.95, using the
     pre-update value. Persisted.
5. Zero-variance filter over the same groups.
6. `norm`: `mean` (default) | `mean_std` | `per_learner`. `per_learner` warns
   when the learner has a single role.
7. `segment_credit` over units u=1..U, with `unit: session (default for
   multi_session) | segment`:
   - `all`: w=1;
   - `last`: w_U=1 (units with weight 0 emit no datums);
   - `geometric(γ)`: w_u=γ^(U−u);
   - `normalize: none | sum_to_one`.
8. `recipients`: which roles emit datums.
9. `loss_agg`, applied by scaling advantages because Tinker sums:
   - `token_sum`;
   - `token_mean_per_learner`: divide by N_ℓ;
   - `agent_mean`: divide by `n_{e,a}·M_ℓ`.
10. Build datums.

**Config validation rejects:**
- an `episode` baseline with `individual`/`mix`;
- `individual` for a role with no submission;
- `oracle_any` as a reward;
- recipients that are frozen or API seats;
- an idle learner (unless `allow_idle`);
- `mean_std` together with `rae`;
- G < 2 with a group baseline;
- local learners on different bases, rank > `max_lora_rank`, or
  `max_loras` < learners + snapshots;
- a compaction threshold, or a context, larger than the backend allows.

It warns when `segment_credit ≠ all` but U ≡ 1.

**Worked goldens** (from the review) are committed as tests:
1. Coordinator, per-role LoRAs, G=3, R=[1,0,1]:
   - A = [+.5, −1, +.5] for the coordinator and, episode-weighted, for the
     workers;
   - with `token_mean_per_learner`, divide by 850 (coordinator) and 1100
     (workers);
   - `last` with unit=segment drops e3's first coordinator segment.
2. Swarm N=3, G=2, own-correct e1=[1,0,1], e2=[0,0,1]:
   - `individual`: [+⅔, −⅓, +⅔] / [−⅔, −⅔, +⅓];
   - `team`: ±1;
   - `mix(.5)`: [+⅚, +⅓, +⅚] / [−⅚, −⅚, −⅓].

**Loop (sync v1).**
1. B tasks × G episodes.
2. Credit.
3. Per learner, `forward_backward` + `optim_step`, all submitted together.
   Skip `optim_step` for a learner with no informative datums.
4. Versioned `sync_sampler`.
5. Log per role and per learner: reward, length, calls, workers spawned,
   cross-reads, emergent correctness, grad norm, zero-std fraction,
   `kl_sample_train`, and a **role-capture** share.
6. Write `progress.json`.

Checkpoints are per learner (state + sampler, TTL-free). Resume restores RAE
and the data cursor. **`train sft`** (cross_entropy) plus **`data sft`**
build rejection-filtered chat data from a same-family teacher on Tinker.

## Compute metrics (`eval/compute.py`, computed offline from Calls)

- **`gen_tokens`:** completion ids including stop and thinking, excluding
  prefill, forced closes and injected tokens. `total_gen` is the sum over all
  agents and calls, broken down by role and purpose.
- **Prompt tokens:** `prompt_tokens` is `prompt_len`. `uncached_ideal` is
  `prompt_len` minus the longest common prefix with earlier calls of the same
  `policy@version` in this episode.
- **`calls`:** counted by purpose.
- **Critical path.** The DAG edges are:
  - each agent's calls in sequence;
  - the spawn call → each worker's first call;
  - each worker's last call → the coordinator's next call;
  - in lockstep, tick t → tick t+1;
  - each peer's last call → the finalizer.

  `CP_tokens` is the longest path weighted by `gen_tokens`; `CP_calls` uses
  unit weights (Kimi CriticalSteps).
- **Peak context:** `peak_ctx`, and `peak_active_ctx` (a KV proxy).

Reports give realized mean/p50/p90 and **accuracy vs `total_gen` and vs
`CP_tokens` at ≥3 budget levels**. Wall-clock is logged but never used for
matching.

## CLI verbs (v1)

| Verb | In → Out |
|---|---|
| `list` · `describe` · `inspect` · `status` | introspection |
| `data build <taskset>` · `data filter` · `data sft` | → TaskSet / SFT set |
| `eval rollout --tasks … protocol=<cfg> seating.by_role.x=<ref>` | → EpisodeSet (resumable) |
| `eval score` · `eval report` · `eval grid <grid.yaml>` | → Report: acc with n + CIs, paired lift, acc-vs-tokens and acc-vs-CP, per-session accuracy, oracle_any, conformance rates |
| `train rl` · `train sft` | → Checkpoint(s) per learner |
| `export` · `publish` | Tinker → PEFT; → private HF |
| `serve vllm` (multi-LoRA) · `serve status\|stop` | supervised server + `server.json` |
| `view --episodes …` | HTML: per-agent timeline, workspace diffs, worker tree, calls |

## Repo layout

```
src/marli/  config registry handles rundir runlog errors verbs budget tracking publish  cli/
  models/*.yaml model.py   llm/{client,judge}   policy/{base,refs,scripted,api,tinker,vllm}
  render/{base,fake,qwen3,qwen3_5,gpt_oss,tools,rerender}
  interact/{types,workspace,tools,agent,context,limits,scheduler,system,records}  interact/protocols/{single,multi_session,swarm,coordinator,presets}  interact/configs/*.yaml
  envs/{base,math,code_fn,swe}  envs/sandbox/{base,subprocess,docker}
  tasks/*.yaml tasks/{loaders,verifiers,equivalence}  rewards  data/{build,filter,sft}
  eval/{rollout,score,report,grid,stats,compute}
  train/{types,datums,credit,advantages,loop,sft,checkpoint,export}  train/backends/{base,fake,tinker}  train/backends/local/{learner,losses,backend,sync}
  serve/{vllm,_supervise}  viewer/html
```

## Milestones and Codex tasks

**Conventions.**
- One PR per milestone.
- `[C]` = Claude writes contract stubs and golden tests first.
- Codex tasks own their `src` files plus `tests/<pkg>/test_<mod>.py`.
- Claude owns conftest, fixtures, the pyproject extras, and the verb table
  (`verbs.py`, pre-populated with `module:function` rows).
- W1/W2/W3 are parallel waves.
- Claude runs every paid or GPU step, with `max_usd`, after confirming with
  Sid.

**M0: scaffold.**
- Claude: sync CLAUDE.md to the agent-systems framing (credit via
  `train.credit.*`); add this plan as `docs/plans/2026-09-23-agent-systems.md`
  and mark the debate plan superseded; `errors.py`, stubs, `verbs.py`,
  conftest.

| ID | Wave | Scope |
|---|---|---|
| M0-1 | W1 | config |
| M0-2 | W1 [C] | handles + rundir |
| M0-3 | W1 | registry + runlog |
| M0-4 | W2 | CLI + contract tests |
| M0-5 | W2 | CI, wiki skeleton, experiment template, cli.md check |

The final step creates the GitHub repo and opens the PR.

**M1a: tokens and policies.**
- Claude: `interact/types`, `policy/base`, `render/base` + `fake`, seeds,
  Qwen3 HF golden tokens.

| ID | Wave | Scope |
|---|---|---|
| M1-1 | W1 | llm client + judge port (cache_salt) |
| M1-2 | W1 | models registry + budget |
| M1-3 | W1 [C] | render qwen3 + tools + rerender, with parity tests |
| M1-4 | W1 | scripted policy + refs |
| M1-5 | W2 | api/tinker/vllm policies with mocked transports + trainable guard |

**M1b: interaction core.**
- Claude writes goldens: a lockstep 2-peer trace, a nested spawn trace,
  segment boundaries, the `force_final` layout, and hand-computed CP and
  uncached tokens.

| ID | Wave | Scope |
|---|---|---|
| M1-6 | W1 | workspace + tools |
| M1-7 | W1 [C] | context managers |
| M1-8 | W2 [C] | agent + limits |
| M1-9 | W2 [C] | scheduler + system |
| M1-10 | W1 [C] | records + compute |
| M1-11 | W3 | `single` protocol + a walking-skeleton end-to-end test |

**M2: protocols, math, eval, serving, first pod.**

| ID | Wave | Scope |
|---|---|---|
| M2-1 | W1 | multi_session |
| M2-2 | W1 [C] | swarm + presets + vote/finalizer (Du-debate golden) |
| M2-3 | W1 | coordinator + `spawn_workers` |
| M2-4 | W1 | tasks registry, verifiers pool, equivalence |
| M2-5 | W1 | envs/base + math + sandbox/base + subprocess + rewards |
| M2-6 | W2 | eval rollout + score (SIGKILL resume) |
| M2-7 | W2 | stats + report + grid |
| M2-8 | W1 | data build + filter |
| M2-9 | W1 | serve vllm + pod requirements + runbook + `marli-on-pod` skill |
| M2-10 | W2 | viewer |

- **Pod check (≤$15)** is the model-support spike, run on Qwen3.5-4B/9B and
  gpt-oss-20b:
  - HF+PEFT load, and LoRA targets for the hybrid layers;
  - fwd/bwd memory at 16k/32k;
  - vLLM LoRA r32;
  - HF-vs-vLLM logprob agreement;
  - native tool-call token round-trip;
  - stop-token inclusion.

  If the spike fails for a family, that family runs Tinker-only and the
  local learner uses Qwen3 dense.
- **Pilots:** a conformance pilot (valid tool calls, spawn, return_report,
  cross-reads, end_session) and a compute-matched pilot on AIME/HMMT across
  all protocols.
- **SFT gate:** if conformance is low, SFT warm-start before RL.

**M3: training core + Tinker** (can overlap with M2; it only consumes
Episodes).
- Claude: `train/types`, the credit config, validation tests, and both
  worked goldens.

| ID | Wave | Scope |
|---|---|---|
| M3-1 | W1 | datums |
| M3-2 | W1 [C] | advantages + credit |
| M3-3 | W1 | backends base + fake |
| M3-4 | W1 | TinkerBackend (fake tinker module; multi-client) |
| M3-5 | W2 | loop + checkpoint + `train rl` |
| M3-6 | W2 | sft + `data sft` |
| M3-7 | W3 | tracking + CPU CLI end-to-end chain |

- Live Tinker smoke (≤$5, token-sized): coordinator with shared vs per-role
  LoRA, and multi_session with credit last vs all, 3 steps each.

**M4: local learner (pod, ≤$25).**
- Losses (CPU reference tests), a PEFT multi-adapter learner with chunked
  logprobs, versioned vLLM multi-LoRA sync, export/publish,
  `requirements/pod-train.txt`.
- Checks:
  - per-adapter IS ratio ≈ 1;
  - a 20-step per-role-LoRA run;
  - kill + resume;
  - YaRN/`max_seq_len` parity.

**M5: coding.**
- The `docker` sandbox and the `code_fn` env (DeepCoder / LCB v6).
- The `swe` env on local Docker, run from the laptop or AWS box: eval first,
  then small RL via Tinker.

**M6: first experiments.**
- Compute-matched baselines across families, on Qwen3.5-4B/9B, gpt-oss-20b
  and Qwen3.6-35B-A3B (Tinker).
- Training pilots:
  - shared vs per-role LoRA;
  - reward to all vs coordinator-only;
  - multi-session credit last vs all;
  - backend parity.
- Wiki ingest.

**Deferred:**
- Competitive/mixed-motive games.
- Async off-policy.
- Hierarchical/branch-LOO (C3) and HiMPO memory credit.
- KV-sharing parallel inference.
- inspect bridge.
- FSDP multi-GPU.
- Hosted sandboxes.
- A remote-learner service.
- A deterministic virtual-time async mode.

## Risks to watch (instrumented)

- **Role capture** with a shared LoRA, and a **terminal accuracy cliff** with
  per-role LoRAs when a role has several slots. Track per-role shares and
  grad norms.
- **Serial collapse / spurious parallelism.** Keep aux rewards annealed and
  capped.
- **Summary collapse.** Use overlong masking and carry caps.
- **Train/test protocol mismatch.** Evaluate with and without the protocol.
- **Reasoning solipsism.** Track cross-reads and emergent correctness.
- **Stragglers** in sync RL. Use `episode.max_wall_s`, and log the rollout/CP
  ratio.
- **Qwen3.5 hybrid architecture support.** Gated by the M2 spike.

## Verification

- **CI:** `uv run pytest -q` (CPU, no network) + `ruff`. It includes a CLI
  end-to-end run with scripted policies:
  1. `data build`;
  2. `eval rollout` × each protocol;
  3. `score`;
  4. `report`;
  5. `train rl` for 2 steps on FakeBackend with 2 learners;
  6. `eval rollout` from `ckpt:`.

  Plus lockstep byte-determinism, the credit goldens, and validation
  rejections.
- **Live, run by Claude under `max_usd`:**
  - an API-model swarm/coordinator smoke (M1/M2);
  - the Tinker multi-learner smoke (M3).
- **GPU:**
  - M2 spike and pilots;
  - M4 IS-ratio, resume and seq-len checks;
  - M6 parity and baselines.
