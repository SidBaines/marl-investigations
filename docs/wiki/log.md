# Wiki log

## [2026-10-05] ingest | Sacrifice relay: experiments 1–3 and the evals (study report)

Third experiment ingest. Source added:

- [Sacrifice relay: experiments 1–3 and the evals](../sources/sacrifice-relay-experiments-1-3-and-evals.md)
  (REPORT.md, status partial: one seed per run, but several arms, two models,
  and eval comparisons with 95% intervals).
- It comes from branch evals-session-1005 @ 3d6551f. That branch merges
  m6-harness, exp2-eval and evals-coop. No PR.

Headline, recorded in the rewritten
[sacrifice synthesis](syntheses/does-rl-teach-sacrifice.md):

- Team reward taught the first contributor to reveal the rule when that cost
  it nothing and paid the team. This is experiment 2, 0/1 scoring:
  - Qwen3.8-27B: 17% → 80% over 60 steps;
  - Qwen3.6-35B-A3B: 38% → 77% over 80 steps.
- Team reward removed reviewing when a costly review did not pay. In
  experiment 3 (A3B, 0/1/3 scoring), informed followers scored the bonus
  only 14% of the time, and reviewing fell from 21% to 3%.
- Individual reward never taught reviewing.
- The one costly sacrifice that did pay (experiment 1) was still not learned.
- The learned behaviour transferred to held-out problems and, for the 27B,
  to a fully changed format. It did not make either model more cooperative in
  standard evals, apart from a small HiddenBench gain on the A3B.
- The old answer ("not seen yet") is struck through under History.

Pages created:

- concepts:
  [sacrifice-pays-in-practice](concepts/sacrifice-pays-in-practice.md),
  [behaviour-transfers-across-format](concepts/behaviour-transfers-across-format.md),
  [harness-dependent-helpfulness](concepts/harness-dependent-helpfulness.md).
- entities: [qwen3-6-35b-a3b](entities/qwen3-6-35b-a3b.md),
  [standard-coop-evals](entities/standard-coop-evals.md) (the external-eval
  layer and its four suites; listed under "Benchmarks and datasets").
- The candidate concept "MoE routing near-ties break fixed-probe adapter
  checks" was folded into
  [on-policy-check](concepts/on-policy-check.md) as a section, not a page. It
  is one observation on one model, and that page already owns the hot-load
  check.

Pages updated:

- [does-rl-teach-sacrifice](syntheses/does-rl-teach-sacrifice.md): new
  current answer, an evidence table across all eight runs, hypothesis
  statuses, history, tensions.
- [rewarded-choice-not-learned](concepts/rewarded-choice-not-learned.md):
  experiments 2 and 3 bracket experiment 1; evidence for spill-over.
- [reactive-information-sharing](concepts/reactive-information-sharing.md):
  proactive first-mover review was learned in experiment 2 and held on new
  problems; the position question is struck through.
- [token-budget-binding](concepts/token-budget-binding.md): reaching CI rose
  in every run, and was the main gain under 0/1/3 scoring.
- [zero-variance-groups](concepts/zero-variance-groups.md): the share of
  silent groups under sparse 0/1 scoring (42–49% → 8% under team reward;
  40–45% under individual reward).
- [on-policy-check](concepts/on-policy-check.md): 27B and A3B bounds over
  longer runs, the MoE fixed-probe drift, and the 5e-3 abort line.
- [env-code-rules](entities/env-code-rules.md): scoring options, prompt
  variants, transfer settings, held-out rule forms, help-note settings, and
  realised payoff.
- [qwen3-8-27b](entities/qwen3-8-27b.md): experiment 2, evals, chat-template
  effort.
- [local-backend](entities/local-backend.md): MoE support, pause and resume,
  frozen adapter seats, one server for two harnesses, new ops notes.
- [deepcoder](entities/deepcoder.md): reworded copies inside DeepCoder, and
  the held-out set's construction; the lcb tension is updated.
- [protocol-relay](entities/protocol-relay.md): `opener_role`, `relay_n3` and
  `relay_n5`, and where each run used it.
- [index.md](index.md): new pages, the new source, and refreshed
  descriptions.

That is 16 pages: above the schema's typical 4–12, because the source covers
three experiments and an eval session.

Tensions recorded:

- The report's summary counts experiment 2 as a sacrifice ("give up its own
  CI run"). Experiment 2's design notes say the review costs a contributor
  without the rule nothing. The synthesis follows the design notes.
- Experiment 3 vs experiment 1 changes the model, batch, adapter and three
  prompt settings at once. Experiment 3 vs experiment 2's A3B team arm
  isolates the scoring.
- The A3B's `far` drop is attributed to the new rule kinds, but the
  single-change cells were not run.
- The report says 1,200 contributors in the `far` games. The results file has
  800 (4 cells × 50 games × 4).
- The A3B HiddenBench gain is the only significant effect among many tests,
  and its sample was extended after a promising first look.
- Experiment 3's "informed followers scored 3 only 14%" sits next to "83% of
  their submissions passed", which may use different denominators.
- "Helpfulness depends on the harness" bundles framing, note placement, task
  difficulty, reporting channel and sampling settings.
- Inside the relay, the help note cost the 27B team policy team score (−0.085)
  and contributor 1's reviews (−19 points). The report does not mention this.
- The A3B's hot-load tolerance of 0.5 nats weakens the serving-side guard.
- Two concurrency guides conflict: "80–100 agents per GPU" came from single
  agents, "30–40 relay agents per engine" from relay games.

Candidate follow-ups:

- A second seed of experiment 2 (team vs individual) and of experiment 3 vs
  experiment 2's A3B team arm.
- Experiment 3 transcripts: why informed A3B followers skip the rule. Then a
  variant where the rule is required for the 1 point too, or the bonus is
  larger.
- Experiment 1's regime with only the `checks` sentence added. This tests the
  thin-signal explanation.
- Experiment 2.1 with team reward for followers.
- Transfer eval single-change cells, especially new rule forms alone.
- HiddenBench with 10 discussions per task, on the individual and 2.1
  policies.
- The relay vs Inspect help test, varying only note placement and framing.
- A fixed-probe check that works for MoE models.
- The solo control, still never run.

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

## [2026-10-05] ingest-fix | Corrections to the sacrifice-relay study report

Corrected the report (`experiments/2026-09-25_sacrifice-relay/REPORT.md`) after review and re-copied it verbatim into
[the source page](../sources/sacrifice-relay-experiments-1-3-and-evals.md) (provenance updated):
- experiment 2's review is information gathering at no cost to the reviewer, not a sacrifice;
- 800 (not 1,200) contributors in the `far` games;
- the A3B's `far` drop is probably, not certainly, the new rule kinds;
- the bases of experiment 3's 14% and 83%;
- the HiddenBench result framed as a lead;
- "harness" dependence reworded as setting dependence with its confounds;
- the help note's cost to the trained 27B.

Also fixed "keeps about half" to "about a third of its gain" for the A3B under `far` in
[the synthesis](syntheses/does-rl-teach-sacrifice.md),
[behaviour vs format](concepts/behaviour-transfers-across-format.md) and the
[A3B entity](entities/qwen3-6-35b-a3b.md).
