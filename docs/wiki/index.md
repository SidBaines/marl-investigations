# Research wiki

The catalog of curated knowledge and its sources; start here when querying past findings.

## Concepts

- [Compute-matching](concepts/compute-matching.md) — A shared episode token budget is a cap, not a match — compare protocols on realized total_gen and cp_tokens; the eval report flags cells more than 10% off the baseline's mean total_gen.
- [Budget splitting truncates thinking-model chains](concepts/budget-splitting-truncation.md) — Splitting a fixed episode budget across parallel agents gives each agent less than a thinking model's typical chain, so peers run to their caps and accuracy falls (SC@4, swarm and debate at 32k with Qwen3.5-4B).
- [Aggregation loss (oracle vs voted)](concepts/aggregation-loss.md) — In voting protocols the voted answer is often wrong when some peer was right; oracle_any bounds what better aggregation could recover, and even that bound does not beat single in the untrained 32k pilot.
- [Multi-session carry: notes vs compaction](concepts/multi-session-carry.md) — One agent's budget split into 3 fresh-context sessions, carrying notes-to-self or a compaction summary; on the fixed harness notes ties single on AIME25 at ~60% of the tokens, but most episodes submit in session 1, so compute is not matched.
- [Unanswered episodes: read the answered rate before accuracy](concepts/unanswered-episodes.md) — A low answered rate means accuracy is measuring a harness or budget-design failure rather than reasoning; the known cases are the multi-session nudge-strike bug (fixed in 76aa955) and debate's per-round call cap.
- [On-policy check: kl_sample_train](concepts/on-policy-check.md) — The mean sampler-minus-trainer logprob over action tokens; near zero means the trainer scored exactly the token ids the sampler produced; measured 2e-4 to 1e-3 in the Tinker RL smoke.
- [Zero-variance groups at small G](concepts/zero-variance-groups.md) — With a group baseline and small groups, most groups have identical rewards, so advantages are zero and the group is dropped; at B=2×G=2 the Tinker RL smoke's multi-session runs barely stepped and could not compare credit schemes.

## Entities

Models and backends:

- [Qwen3.5-4B](entities/qwen3-5-4b.md) — Qwen/Qwen3.5-4B, a 4B thinking model (renderer qwen3_5) served and LoRA-trained on Tinker; the policy of the compute-matched baselines pilot and the Tinker RL smoke.
- [Tinker (backend)](entities/tinker.md) — Hosted sampling and LoRA-training backend; marli policy refs `tinker:<model>` and `backend: tinker` learners, with exact sampled ids, raw logprobs and versioned sampler sync.

Benchmarks and datasets:

- [AIME 2025](entities/aime-2025.md) — MathArena/aime_2025, 30 integer-answer competition problems (CC BY-NC-SA, contamination-sensitive, never commit text); scored with math-verify.
- [HMMT February 2025](entities/hmmt-feb-2025.md) — MathArena/hmmt_feb_2025, 30 LaTeX-answer competition problems (CC BY-NC-SA, contamination-sensitive, never commit text); scored with math-verify.
- [POLARIS-53K](entities/polaris-53k.md) — POLARIS-Project/Polaris-Dataset-53K, a 53k-problem math RL training set with LaTeX answers and a per-task 7B pass-rate difficulty field; used by the Tinker RL smoke.

Protocols:

- [Protocol: single](entities/protocol-single.md) — One solver agent with a submit tool; the baseline cell of every comparison.
- [Protocol: independent (SC@k)](entities/protocol-independent.md) — N independent samples with no cross-peer tools or deliveries, aggregated by verifier-equivalent vote — self-consistency, i.e. the swarm with visibility off.
- [Protocol: swarm (coordinator-free)](entities/protocol-swarm.md) — N parallel peers sharing versioned scratchpads (notify or push delivery), each with its own answer and credit, aggregated by vote or a separate finalizer.
- [Protocol: coordinator + workers](entities/protocol-coordinator.md) — A coordinator spawns fresh-context workers via spawn_workers and composes their return_report outputs; the RL smoke trains it with a shared or per-role LoRA.
- [Protocol: multi-session](entities/protocol-multi-session.md) — One agent's budget divided across fresh-context sessions with a carry mode (notes, compaction, both, tail); each session is a new training segment.
- [Protocol: debate (baseline only)](entities/protocol-debate.md) — N peers publish replies over lockstep rounds (push delivery, no tools, final text, full vote); kept only as a baseline.

## Syntheses

- [Do agent systems beat a single agent at matched compute on hard math (untrained)?](syntheses/agent-systems-vs-single-matched-compute.md) — Current answer: no. At a 32k episode budget with untrained Qwen3.5-4B on AIME25/HMMT25 (n=30, G=1), no protocol beats single; parallel splitting loses at matched compute, and coordinator and notes only tie on AIME25 at ~60% of the tokens [pilot].

## Sources

- [Compute-matched baselines pilot (untrained Qwen3.5-4B, AIME25 + HMMT25)](../sources/compute-matched-baselines-pilot.md) — Untrained Qwen3.5-4B (Tinker) on AIME25 and HMMT25, n=30 each, G=1, seven protocol cells at a 32k-token episode budget, plus a multi-session rerun on the fixed harness (AIME25 only); no cell beats single.
- [Compute-matched baselines pilot — eval-report tables (aggregates only)](../sources/compute-matched-baselines-results.md) — The three `marli eval report` tables behind the compute-matched baselines pilot (AIME25 pilot, HMMT25 pilot, AIME25 multi-session rerun) — accuracy with Wilson CIs, oracle, answered rate, compute percentiles, paired lift with bootstrap CI and McNemar p, and the >10% compute-match flag.
- [Tinker RL smoke (shared vs per-role LoRA; session credit all vs last)](../sources/tinker-rl-smoke.md) — A 3-step, B=2×G=2 Tinker RL plumbing smoke for Qwen3.5-4B LoRA — coordinator with shared vs per-role LoRA, multi-session credit all vs last; validates the training stack (kl_sample_train ≈ 0, sampler sync, checkpoints), not a science result.
- [scimt GRPOOptions (TRL GRPO backend knobs, with rationale comments)](../sources/scimt-grpo-options.md) — The GRPOOptions dataclass from scimt's hf_grpo backend — its comments record RL-ops lessons (pods are not storage, budget-relative truncation, server-mode vLLM, zero-std/zero-gradient guards).
- [Lessons from scimt prior-latmem](../sources/scimt-prior-latmem-lessons.md) — Ops postmortems from scimt's prior-latmem experiment — reasoning-token budgets, the always-on ChatClient cache collapsing samples, per-batch persistence, kill-testing crash paths, spend control.
- [scimt RLVR throughput matrix (TRL GRPO + vLLM, Gemma-4 26B)](../sources/scimt-rlvr-throughput-matrix.md) — Measured throughput/memory matrix and failure modes from scimt's TRL-GRPO RLVR runs — fp32 logits-upcast OOMs, gradient checkpointing, vLLM engine hangs, ninja on PATH.
