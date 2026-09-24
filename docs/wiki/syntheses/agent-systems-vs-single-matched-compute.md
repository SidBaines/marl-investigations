---
type: synthesis
title: Do agent systems beat a single agent at matched compute on hard math (untrained)?
description: "Current answer: no. At a 32k episode budget with untrained Qwen3.5-4B on AIME25/HMMT25 (n=30, G=1), no protocol beats single; parallel splitting loses at matched compute, and coordinator and notes only tie on AIME25 at ~60% of the tokens [pilot]."
resource: docs/sources/compute-matched-baselines-pilot.md
tags: [synthesis, compute-matching, protocols, math, untrained, baselines]
timestamp: 2026-09-24
---

# Do agent systems beat a single agent at matched compute on hard math (untrained)?

## Current answer [pilot]

**No, not in the one regime measured so far.** Untrained Qwen3.5-4B was run at a
32k-token episode budget on AIME25 and HMMT25. No agent system beat `single`.

- **Parallel protocols lose, including at matched compute.** These are SC@4,
  swarm4 and debate3. On HMMT25, all three are within 10% of `single`'s mean
  `total_gen`, and each loses by 0.30–0.50 with McNemar p ≤ 0.004. On AIME25,
  swarm4 (+7% compute) loses by 0.20 (p=0.03).
- **The coordinator and notes-to-self multi-session tie `single` on AIME25
  while using about 40% fewer tokens.** That is not compute-matched, so it is
  not a win. The coordinator is worse on HMMT25 (−0.20, p=0.11).
  Multi-session has no valid HMMT25 number yet.
- **The main mechanism is budget splitting.** This model's single chain
  usually runs to about 31k tokens, so each 8k peer is truncated. Aggregation
  loses a few more correct answers on top of that.

Every result is one seed, n=30 per benchmark, G=1, one model, one budget, and
no training. A different budget could reverse the parallel-protocol result.
Treat this as a baseline to beat, not a law.

## Regime

- **Model and backend:** `tinker:Qwen/Qwen3.5-4B` ([entity](../entities/qwen3-5-4b.md)).
  It is a thinking model with renderer `qwen3_5`, sampled at T=1, top_p=1,
  top_k=−1, and untrained.
- **Benchmarks:** [AIME25](../entities/aime-2025.md) and
  [HMMT25](../entities/hmmt-feb-2025.md), n=30 each. G=1 episode per task, one
  seed, lockstep, no python tool, math-verify.
- **Compute:** a 32,768 generated-token cap per episode. Realized compute
  differs by cell (see [compute-matching](../concepts/compute-matching.md)).
- **Statistics:** accuracy has Wilson 95% CIs. Lift is the paired difference
  vs `single` with a 2,000-resample task-bootstrap 95% CI, discordant W/L and
  exact McNemar p. There is no correction for multiple comparisons: about 14
  lifts were tested.
- **Sources:** [pilot README](../../sources/compute-matched-baselines-pilot.md)
  and [report tables](../../sources/compute-matched-baselines-results.md). Total
  spend was $14.00 (pilot) plus about $2.60 (rerun).

## Evidence

"Matched?" means within 10% of `single`'s mean `total_gen` on that benchmark
(the report's rule). `total_gen` and `cp_tokens` are means, AIME25 / HMMT25.

| Cell | AIME25 acc | AIME25 lift (W/L, p) | HMMT25 acc | HMMT25 lift (W/L, p) | total_gen | cp_tokens | Matched? (A / H) |
|---|---|---|---|---|---|---|---|
| [single](../entities/protocol-single.md) | 0.533 [0.361, 0.698] | — | 0.533 [0.361, 0.698] | — | 25.5k / 29.0k | 25.5k / 29.0k | — |
| [coordinator](../entities/protocol-coordinator.md) | 0.567 [0.392, 0.726] | +0.033 [−0.067, +0.133] (2/1, 1.00) | 0.333 [0.192, 0.512] | −0.200 [−0.400, +0.000] (2/8, 0.11) | 15.8k / 17.7k | 13.8k / 15.2k | no / no |
| [ms3_notes](../entities/protocol-multi-session.md), fixed harness | 0.500 [0.332, 0.668] | −0.033 [−0.167, +0.067] (1/2, 1.00) | — | — | 15.3k / — | 15.3k / — | no / — |
| [ms3_compaction](../entities/protocol-multi-session.md), fixed harness | 0.400 [0.246, 0.577] | −0.133 [−0.300, +0.033] (2/6, 0.29) | — | — | 13.5k / — | 13.5k / — | no / — |
| ~~ms3_notes, pilot~~ | ~~0.467~~ | ~~−0.067 (3/5, 0.73)~~ | ~~0.333~~ | ~~−0.200 (2/8, 0.11)~~ | | | superseded † |
| ~~ms3_compaction, pilot~~ | ~~0.300~~ | ~~−0.233 (1/8, 0.039)~~ | ~~0.333~~ | ~~−0.200 (2/8, 0.11)~~ | | | superseded † |
| [swarm4](../entities/protocol-swarm.md) | 0.333 [0.192, 0.512] | −0.200 [−0.367, −0.067] (0/6, 0.031) | 0.233 [0.118, 0.409] | −0.300 [−0.467, −0.133] (0/9, 0.004) | 27.2k / 29.2k | 8.8k / 9.8k | **yes / yes** |
| [sc4](../entities/protocol-independent.md) | 0.367 [0.219, 0.545] | −0.167 [−0.333, −0.033] (1/6, 0.125) | 0.200 [0.095, 0.373] | −0.333 [−0.500, −0.167] (0/10, 0.002) | 30.6k / 30.8k | 7.7k / 7.7k | no (+20%) / **yes** |
| [debate3](../entities/protocol-debate.md) ‡ | 0.133 [0.053, 0.297] | −0.400 [−0.567, −0.233] (0/12, <0.001) | 0.033 [0.006, 0.167] | −0.500 [−0.667, −0.333] (0/15, <0.001) | 30.5k / 31.2k | 10.4k / 10.4k | no (+20%) / **yes** |

† The pilot's multi-session rows came from the
[nudge-strike harness bug](../concepts/unanswered-episodes.md). They are
replaced by the AIME25 rerun on the fixed harness (76aa955). HMMT25 has no
replacement yet: the rerun stopped at 7 / 6 of 30 episodes.
‡ Answered rate was 0.233 / 0.067, because the per-round call cap truncated
reasoning before an answer. This is a budget-design failure.

## Why (mechanisms, each [pilot])

1. **Budget splitting truncates chains.** Parallel peers run to their 8k caps.
   `single` itself has a p50 near the 32k cap. See
   [budget splitting](../concepts/budget-splitting-truncation.md).
2. **Aggregation loses correct answers.** On AIME25, swarm4 has oracle 0.533
   and voted 0.333. On HMMT25, sc4 has oracle 0.367 and voted 0.200. Even the
   oracle does not exceed `single`. See
   [aggregation loss](../concepts/aggregation-loss.md).
3. **Some cells do not use their compute.** The coordinator and multi-session
   agents submit early. More than half of multi-session episodes submit in
   session 1. See [compute-matching](../concepts/compute-matching.md) and
   [multi-session carry](../concepts/multi-session-carry.md).

## Why only [pilot]

The schema reserves [firm] for multi-seed, multi-model or multi-benchmark
evidence that is CI-backed. What exists here:

- One seed, G=1, and n=30 per benchmark. The Wilson CIs are about ±0.17.
- One model, one budget, and untrained policies only.
- A harness fix mid-study. Only the multi-session cells were rerun, and the
  rerun's `single` baseline predates the fix.

The strongest single claim is the matched-compute swarm4 loss. It holds on
both benchmarks, and both bootstrap CIs exclude 0. That makes it the best
candidate for promotion to [partial] once a second seed or a G≥2 run confirms
it. Until then it stays [pilot], like everything else on this page.

## Tensions

- **The pre-registered test was against a different baseline.** The study's
  falsification criterion was a paired lift vs **SC@4** at matched mean
  `total_gen`. The committed report computes lift vs **single** only. Because
  SC@4 is itself well below `single`, the hypothesis ("multi-agent gains over
  SC@4 vanish at matched compute") has not been tested as written. [open]
- **The coordinator's sign flips.** It ties on AIME25 (+0.03) and is worse on
  HMMT25 (−0.20, CI upper bound 0.000). At n=30 this cannot tell a
  benchmark-dependent effect from noise. [open]
- **Latency is a separate question.** Parallel cells cut the critical path to
  about a third of `single`'s. No `single` cell at an ~8–10k budget exists, so
  accuracy at matched `cp_tokens` is unknown. [open]

## Open questions (what would change the answer)

- **64k episode budgets** (this model's Tinker context is 65,536), or fewer,
  longer-budget peers. This tests whether the loss comes from truncation. [open]
- **G≥2** (maj@k, avg@k, tighter CIs) and more seeds. [open]
- **Debate budget redesign:** per-round caps that fit a thinking model. [open]
- **Finish the HMMT25 multi-session rerun** (about $2 by the README's
  estimate). [open]
- **Match realized compute** for coordinator and multi-session: raise
  per-session and coordinator budgets, add a "use all sessions" instruction,
  or add `single` cells at lower budgets. [open]
- **Lift vs SC@4** from the saved Scores, to test the hypothesis as written.
  [open]
- **Trained policies.** Everything above is untrained. Whether RL (shared vs
  per-role LoRA, session credit) closes the gap is the program's actual
  question. So far only the plumbing has been validated
  ([Tinker RL smoke](../../sources/tinker-rl-smoke.md);
  [on-policy check](../concepts/on-policy-check.md),
  [zero-variance groups](../concepts/zero-variance-groups.md)). [open]
