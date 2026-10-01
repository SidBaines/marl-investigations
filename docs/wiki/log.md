# Wiki log

## [2026-10-01] ingest | Sacrifice relay experiment 1 + throughput benchmark

Second experiment ingest. Sources added, both from branch m6-harness @ 0bd4647
(no PR yet):

- [Sacrifice relay, experiment 1](../sources/sacrifice-relay-experiment-1.md)
  (README; status pilot: one seed, one arm, 30 of 80 planned steps).
- [Speed and memory benchmark](../sources/sacrifice-relay-throughput-bench.md)
  (bench/README; status partial: direct measurements, each run once).

Headline, recorded in the new synthesis
[does RL teach sacrifice?](syntheses/does-rl-teach-sacrifice.md):

- Not seen yet. Qwen3.8-27B ran on the local backend, on the code_rules relay
  (N=4, m=3), with team reward for 30 steps of 4 repos × G=4.
- Team score rose by +0.086 (z=2.3), mostly from more contributors reaching CI.
- The sacrifice rate fell from 10.6% to 7.2% (z=−1.7), even though
  sacrificing paid off for the team (+0.31 [+0.21, +0.42]) and the advantage
  favoured it for contributors 1–2.
- Reviews are triggered by an earlier contributor's note.

Ops headline: on 2×H200, time-sharing the GPUs (vLLM TP2 + MTP + sleep mode,
data-parallel learner) cut a relay step from ≈25 to ≈14–15 min. Logprobs stayed
exact.

Pages created:

- concepts:
  [rewarded-choice-not-learned](concepts/rewarded-choice-not-learned.md),
  [reactive-information-sharing](concepts/reactive-information-sharing.md),
  [token-budget-binding](concepts/token-budget-binding.md),
  [speculative-decoding-mtp](concepts/speculative-decoding-mtp.md),
  [gpu-time-sharing](concepts/gpu-time-sharing.md).
- entities: [qwen3-8-27b](entities/qwen3-8-27b.md),
  [local-backend](entities/local-backend.md),
  [deepcoder](entities/deepcoder.md),
  [env-code-rules](entities/env-code-rules.md) (a new "Environments" group in
  the index), [protocol-relay](entities/protocol-relay.md).
- synthesis: [does-rl-teach-sacrifice](syntheses/does-rl-teach-sacrifice.md).

Pages updated:

- [on-policy-check](concepts/on-policy-check.md): local-backend
  `kl_sample_train`, IS ratio and adapter effect drift. The "local path not
  measured" caveat is struck through.
- [zero-variance-groups](concepts/zero-variance-groups.md): the payoff check's
  recomputed advantages, including the drop, match the log under team reward.
- [unanswered-episodes](concepts/unanswered-episodes.md): case 3, contributors
  who never reach CI.
- [budget-splitting-truncation](concepts/budget-splitting-truncation.md): the
  per-call cap in agentic coding.
- [tinker](entities/tinker.md): link to the local backend.
- [qwen3-5-4b](entities/qwen3-5-4b.md): same architecture class now trained
  locally at 27B.
- [index.md](index.md).

Tensions recorded:

- The pre-registered early-vs-late readout (steps 0–9 vs 70–79) became 0–9 vs
  20–29.
- The pre-training gate (stop if the probe rate is below 3%) is not reported.
- The sacrifice-rate denominator conditions on reaching CI, which itself rose.
- The source names the per-call cap as the main constraint, while scimt found
  that raising a thinking cap barely helps.
- The registry still says Qwen3.8-27B `local: unverified`.
- The DeepCoder build did not pass the `lcb_v6` exclude.
- The learner-copy memory estimate in the serve config is superseded by the
  measurement.

Candidate follow-ups:

- The individual-reward and solo arms, plus seeds.
- 80 steps.
- More choice events per step, or decision-level credit.
- Effort `low` vs a larger per-call cap.
- Read transcripts to check the "note" heuristic.
- Report the share of zero-variance groups.
- Update the qwen3_8_27b registry `local` field.

## [2026-09-24] ingest | Compute-matched baselines pilot + Tinker RL smoke

First experiment ingest. Sources added:

- [Compute-matched baselines pilot](../sources/compute-matched-baselines-pilot.md)
  (README @ 4317902, status pilot).
- [Its eval-report tables](../sources/compute-matched-baselines-results.md):
  three RESULTS.md files, verbatim and concatenated.
- [Tinker RL smoke](../sources/tinker-rl-smoke.md) (README @ f62bc7a, status
  pilot; infrastructure validation only).

Both come from branch m3-harness, PR #4 (open).

Headline, recorded in the new synthesis
[agent systems vs single at matched compute](syntheses/agent-systems-vs-single-matched-compute.md):
untrained Qwen3.5-4B at a 32k budget (n=30, G=1) shows no protocol beating
`single` on AIME25/HMMT25. Parallel splitting loses at matched compute.
Coordinator and notes tie on AIME25 at ~60% of the tokens. Everything is
[pilot].

The pilot's multi-session numbers are struck through (nudge-strike harness
bug, 76aa955) and replaced by the AIME25 rerun. HMMT25 multi-session is marked
unmeasured.

Pages created:

- concepts: [compute-matching](concepts/compute-matching.md),
  [budget-splitting-truncation](concepts/budget-splitting-truncation.md),
  [aggregation-loss](concepts/aggregation-loss.md),
  [multi-session-carry](concepts/multi-session-carry.md),
  [unanswered-episodes](concepts/unanswered-episodes.md),
  [on-policy-check](concepts/on-policy-check.md),
  [zero-variance-groups](concepts/zero-variance-groups.md). The last three
  cross-link the scimt sources.
- entities: [qwen3-5-4b](entities/qwen3-5-4b.md), [tinker](entities/tinker.md),
  [aime-2025](entities/aime-2025.md), [hmmt-feb-2025](entities/hmmt-feb-2025.md),
  [polaris-53k](entities/polaris-53k.md), and the protocol cards
  [single](entities/protocol-single.md),
  [independent](entities/protocol-independent.md),
  [swarm](entities/protocol-swarm.md),
  [coordinator](entities/protocol-coordinator.md),
  [multi-session](entities/protocol-multi-session.md),
  [debate](entities/protocol-debate.md).
- synthesis: [agent-systems-vs-single-matched-compute](syntheses/agent-systems-vs-single-matched-compute.md).

Also updated [index.md](index.md) and removed the `.gitkeep` placeholders.

Tensions recorded:

- The pre-registered test (lift vs SC@4) was not computed.
- The rerun's `single` baseline predates the harness fix.
- Compaction's session-1 accuracy (4/14) is far below notes' (12/17), and its
  forced finals (16) outnumber the episodes that reached session 3 (12).
- The fix commit counts 13/30 unanswered; the report implies 14/30.
- The smoke's "both learners stepped" check passed partly on a zero-advantage
  step.

Candidate follow-ups:

- 64k budgets; G≥2 or a second seed.
- A debate budget redesign.
- Finish the HMMT25 multi-session rerun.
- A lift-vs-SC@4 report.
- `single` at 8k/16k budgets, for matched `cp_tokens`.
- An RL pilot at B≥8, G≥4.

## [2026-09-23] schema | wiki created

Adapted the scimt wiki schema for multi-agent RL in [CLAUDE.md](CLAUDE.md)
and created [index.md](index.md) and the empty concept, entity, and synthesis
directories. Three scimt lesson sources were seeded in `docs/sources/`:

- [scimt GRPOOptions](../sources/scimt-grpo-options.md)
- [Lessons from scimt prior-latmem](../sources/scimt-prior-latmem-lessons.md)
- [scimt RLVR throughput matrix](../sources/scimt-rlvr-throughput-matrix.md)

No concept pages yet.
