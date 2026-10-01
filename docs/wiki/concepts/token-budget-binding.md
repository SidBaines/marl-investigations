---
type: concept
title: "When the token budget binds, reaching the scorer is the main lever"
description: "With 12,288 tokens per contributor and a 6,144-token per-call cap, about 30–38% of Qwen3.8-27B contributors never reach CI on code_rules; RL's first gain was reaching CI more often (61% → 70%) rather than passing more tests, and the per-call cap, not the total budget, is what cuts thinking off [partial]."
resource: experiments/2026-09-25_sacrifice-relay/configs/base.yaml
tags: [compute, budgets, thinking-models, agentic-coding, training, rl, measurement]
timestamp: 2026-10-01
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
