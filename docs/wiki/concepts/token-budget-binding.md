---
type: concept
title: "When the token budget binds, reaching the scorer is the main lever"
description: "With 12,288 tokens per contributor and a 6,144-token per-call cap, about 30–40% of contributors never reach CI on code_rules at the start of training; in every sacrifice-relay run (two models, team and individual reward) reaching CI rose (e.g. 60% → 74%, 63% → 79%), and under 0/1/3 scoring it was the main source of the team-score gain; the per-call cap, not the total budget, is what cuts thinking off [partial]."
resource: experiments/2026-09-25_sacrifice-relay/configs/base.yaml
tags: [compute, budgets, thinking-models, agentic-coding, training, rl, measurement]
timestamp: 2026-10-05
---

# When the token budget binds, reaching the scorer is the main lever

**The phenomenon.** A thinking model on a tool-using task can use up its token
budget before it takes the scored action (here, running CI). Every such
episode scores 0 whatever the quality of its work. When many episodes end
this way, two things follow:

1. The easiest reward for RL to find is "get to the scorer sooner", not "solve
   better".
2. Any behavioural rate that conditions on reaching the scorer changes its
   denominator as training proceeds.

[partial]

**Regime.** [Qwen3.8-27B](../entities/qwen3-8-27b.md) with renderer
reasoning effort `medium`, served by vLLM on the
[local backend](../entities/local-backend.md). Function-level coding on
[DeepCoder](../entities/deepcoder.md) train problems. Budgets: 12,288 generated
tokens per agent, at most 6,144 per call, context at most 24,576, up to 30
calls. Sources: [experiment 1](../../sources/sacrifice-relay-experiment-1.md)
and the [benchmark](../../sources/sacrifice-relay-throughput-bench.md).

## Evidence

**Untrained, single agent** (the difficulty filter: plain `code_fn` with no
house rule, 800 problems × 4 attempts = 3,200 episodes). [partial]

- 22% of episodes ran out of budget (702 of 3,200). Among the problems kept for
  the study, it was 23%.
- Of those, 95% had at least one turn stopped mid-thought at the 6,144-token
  per-call limit, and 74% had two or more.

**Raising the total budget barely helps.** 16 of those out-of-budget
episodes were continued at a 20,480-token total budget (`eval continue`).
1 of 16 then passed, and 15 ran out again, mostly on more turns at the
per-call limit. The source concludes that the per-call limit (or reasoning
effort), not the total budget, is the main constraint. [pilot] (n=16)

**Untrained relay** (16 relays = 64 contributors, unfiltered problems): half
of all contributors hit the 12,288-token budget, and 37.5% never ran CI.
[pilot]

**Under RL** (team-reward relay, 30 steps, one seed): [partial]

| | Steps 0–9 | Steps 20–29 |
|---|---|---|
| Contributors who reached CI | 61.4% | 70.2% (+8.8 points) |
| Base tests pass when submitted | 83.3% | 85.9% (+2.6 points) |
| Team score | 0.545 ± 0.054 | 0.631 ± 0.049 (+0.086, z=2.3) |

The source attributes most of the team-score gain to reaching CI more often:
the model spends fewer turns before running CI. About 30% of contributors
still never reach CI by steps 20–29. Other effects of the binding budget:

- 6 of 93 rule hand-offs failed because the reviewer ran out of tokens before
  writing notes.
- Followers who knew the rule reached CI only 66% of the time (see
  [reactive information gathering](reactive-information-sharing.md)).

## Later runs: reaching CI rises under every reward (2026-10-05) [partial]

Same budgets in every run (12,288 tokens per contributor, 6,144 per call),
the same 51 repos, one seed each. Source:
[study report §3](../../sources/sacrifice-relay-experiments-1-3-and-evals.md).

| Run | Reached CI, steps 0–9 → last 10 |
|---|---|
| Exp 2 team, 27B, 0/1 scoring, 60 steps | 60% → 74% |
| Exp 2 team, [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md), 0/1, 80 steps | 63% → 79% |
| Exp 2.1, 27B, followers who knew the rule | 59% → 70% |
| Exp 3 team, A3B, 0/1/3, 31 steps | 65% → 74% |
| Exp 3 individual, A3B, 0/1/3, 31 steps | 66% → 78% |

- **Under 0/1/3 scoring it is again the main gain.** In experiment 3, both
  arms' team scores rose only through reaching CI and submitting more often
  (the 1-point route). Meanwhile the review rate fell to near 0. [partial]
- **Under 0/1 scoring it limits the team.** In experiment 2's first 30 steps
  (27B), followers who started knowing the rule scored in only 46–47% of
  cases: about 68% reached CI, and some failed the base tests. More reviewing
  cannot fix that. [partial]
- **The cost of a distraction shows up as tokens.** In the transfer eval, a
  planted help note made the 27B team policy's contributors generate about
  850 more tokens each, and score 8.5 points less often
  ([harness-dependent helpfulness](harness-dependent-helpfulness.md)).
  [partial]
- **The same confound applies to experiment 3.** Its review rate is measured
  among contributors who ran CI, a pool that grew from 65% to 74%. [open]

## Reading

- **Read the reached-CI rate before the score.** It is the coding analogue of
  the answered rate in [unanswered episodes](unanswered-episodes.md). [partial]
- **It confounds behavioural readouts.** The sacrifice rate in
  [rewarded but not learned](rewarded-choice-not-learned.md) is measured among
  contributors who ran CI. That pool grew from 359 to 419 between the early and
  late blocks. Whether the newly included contributors behave differently was
  not checked. [open]
- **Same family as [budget splitting](budget-splitting-truncation.md).** A
  thinking model's chain runs into a cap. There the cap was each parallel
  agent's share of a math episode budget. Here it is the per-call cap in an
  agentic coding loop.

## Tensions

- The source points to the per-call cap as the main constraint, which implies
  that raising it should help. That is untested here. scimt's Gemma-4 26B
  thinking runs (TRL, a different model and harness) found the opposite for a
  per-completion cap: raising it from 4,096 to 8,192 left about 28–34% of
  completions truncated, because "the tail is non-terminating thinking"
  ([scimt throughput matrix](../../sources/scimt-rlvr-throughput-matrix.md),
  finding 7). Lower reasoning effort may be the better lever. [open]

## Open questions

- Compare reasoning effort `low` (renderer `qwen3_8_low`) with a larger
  per-call cap, at the same total budget. [open]
- Filter problems on "reaches CI" as well as pass rate, so RL's easiest gain
  does not swamp the behaviour under study. [open]

See also: [compute-matching](compute-matching.md).
