---
type: concept
title: Budget splitting truncates thinking-model chains
description: Splitting a fixed episode budget across parallel agents gives each agent less than a thinking model's typical chain, so peers run to their caps and accuracy falls (SC@4, swarm and debate at 32k with Qwen3.5-4B).
resource: experiments/2026-09-23_compute-matched-baselines/configs/grid.yaml
tags: [compute, budgets, thinking-models, protocols, swarm, debate]
timestamp: 2026-10-01
---

# Budget splitting truncates thinking-model chains

**The phenomenon.** At a fixed episode budget, N parallel agents each get
roughly 1/N of it. A thinking model's reasoning chain has a typical length. When
the per-agent budget is shorter than that, every agent is truncated. In that
case, going wider (more agents) costs depth (chain length), and depth wins.

**Regime:** untrained `tinker:Qwen/Qwen3.5-4B` ([entity](../entities/qwen3-5-4b.md))
on Tinker. Benchmarks are [AIME25](../entities/aime-2025.md) and
[HMMT25](../entities/hmmt-feb-2025.md), n=30 each. G=1, one seed, lockstep,
math-verify, and a 32,768-token episode cap. Accuracy has Wilson 95% CIs. Lift
is the paired difference vs `single` with a task-bootstrap 95% CI and exact
McNemar p. Sources: [pilot README](../../sources/compute-matched-baselines-pilot.md)
and [report tables](../../sources/compute-matched-baselines-results.md).

## Per-agent budgets in the pilot

| Cell | Per-agent budget (call cap) |
|---|---|
| single | 32,768 (call ≤ 16,384) |
| sc4 / swarm4 | 4 × 8,192 (call ≤ 8,192), 512 final reserve |
| coordinator | coordinator 16,384 + ≤ 4 workers × 4,096 |
| debate3 | 3 × 10,922 over 2 rounds (call ≤ 5,461) |
| ms3_* | 1 agent, 3 sessions × ~10.7k |

## Evidence [pilot]

- **A single chain already fills most of the budget.** `single` has
  `total_gen` p50 31,234 on AIME25 and 31,772 on HMMT25, against a 32,768 cap.
  Peak context p50 is 32,270 on AIME25. So even the baseline is budget-bound at
  32k on most problems.
- **Parallel peers run to their caps.** sc4 averages 30,603 / 4 ≈ 7,650
  generated tokens per peer on AIME25. Its `cp_tokens` (the longest peer) has
  p50 7,695 and p90 7,704, which is essentially constant at the 8,192 − 512
  cap. debate3 `cp_tokens` is 10,410 at both p50 and p90, so both round caps
  are hit on essentially every episode.
- **Accuracy falls with the split** (vs `single`, 0.533 on both benchmarks):

| Cell | AIME25 acc | lift (W/L, p) | HMMT25 acc | lift (W/L, p) |
|---|---|---|---|---|
| sc4 | 0.367 [0.219, 0.545] | −0.167 [−0.333, −0.033] (1/6, p=0.125) | 0.200 [0.095, 0.373] | −0.333 [−0.500, −0.167] (0/10, p=0.002) |
| swarm4 | 0.333 [0.192, 0.512] | −0.200 [−0.367, −0.067] (0/6, p=0.03) | 0.233 [0.118, 0.409] | −0.300 [−0.467, −0.133] (0/9, p=0.004) |
| debate3 | 0.133 [0.053, 0.297] | −0.400 [−0.567, −0.233] (0/12, p<0.001) | 0.033 [0.006, 0.167] | −0.500 [−0.667, −0.333] (0/15, p<0.001) |

  On HMMT25, all three cells are within 10% of `single`'s mean `total_gen`, so
  these are compute-matched losses. On AIME25, only swarm4 is within 10%. See
  [compute-matching](compute-matching.md).
- **The coordinator splits less and loses less.** Its coordinator seat gets
  half the budget. It scores 0.567 on AIME25 (+0.033 [−0.067, +0.133]) and
  0.333 on HMMT25 (−0.200 [−0.400, +0.000]), and it spends about 40% fewer
  tokens, so it is not compute-matched.
- **Debate is the extreme case.** Its 5,461-token per-round call cap cuts
  reasoning before a `\boxed{}` answer. The answered rate is 0.233 on AIME25 and
  0.067 on HMMT25 (see [unanswered episodes](unanswered-episodes.md)). The
  pilot README reads this as a budget-design problem for thinking models, not
  as evidence against debate.

## Interpretation

[pilot] This is a property of the **(model, budget)** pair, not of the
protocols. This model's single chain usually runs to about 31k (p50), so any
split of a 32k budget across parallel chains truncates them. Larger budgets
(this model's Tinker context is 65,536), fewer and longer-budget peers, or a
non-thinking model could change the ranking. None of these has been run.
[open]

## Related prior evidence (other sources)

- scimt found the same failure with an API reasoning model (gpt-5 family).
  Reasoning tokens are drawn from `max_completion_tokens` before any visible
  text, so a 4k budget produced truncated outputs and 3% yield
  ([prior-latmem lessons](../../sources/scimt-prior-latmem-lessons.md), item 1).
- scimt's GRPO abort gate treats truncation rate as relative to the budget: 5%
  is "alarming for a 512-token direct cell and normal for a 4096-token
  thinking one" ([GRPOOptions](../../sources/scimt-grpo-options.md)). A single
  budget is not neutral across model types.

## Related: the per-call cap in agentic coding

The same model-budget interaction appears in a coding agent's loop. With
[Qwen3.8-27B](../entities/qwen3-8-27b.md) (effort `medium`) at 12,288 tokens
per agent and 6,144 per call, 22% of single-agent
[DeepCoder](../entities/deepcoder.md) episodes ran out of budget. 95% of those
had a turn cut off mid-thought at the per-call cap. Continuing 16 of them at a
20,480-token total rescued 1. [partial] See
[token budget binding](token-budget-binding.md).

## Tensions

- The truncation mechanism is inferred from token counts sitting at the caps.
  The transcripts have not been read.
- Truncation is not the whole gap. Aggregation also loses correct answers: on
  AIME25, swarm4 oracle is 0.533 and its voted accuracy is 0.333. See
  [aggregation loss](aggregation-loss.md).

## Open questions

- 64k episode budgets; N=2 peers × 16k; debate with per-round caps that fit a
  thinking model; G≥2. [open]

See also: [the synthesis](../syntheses/agent-systems-vs-single-matched-compute.md).
