---
type: concept
title: Aggregation loss (oracle vs voted)
description: In voting protocols the voted answer is often wrong when some peer was right; oracle_any bounds what better aggregation could recover, and even that bound does not beat single in the untrained 32k pilot.
resource: src/marli/eval/score.py
tags: [evaluation, aggregation, voting, swarm, self-consistency, debate]
timestamp: 2026-09-24
---

# Aggregation loss (oracle vs voted)

**Definition.** `oracle_any` is the fraction of episodes in which at least one
agent's *own* answer is verifier-correct (`src/marli/eval/score.py`). The
voted accuracy is the accuracy of the episode's submitted answer. For voting
protocols, that answer is a majority over verifier-equivalent answer groups.
The gap between the two is the correct answers that aggregation threw away.
For protocols with one answering agent (single, coordinator, multi-session),
`oracle_any` equals accuracy by construction. `oracle_any` is eval-only: using
it as a training reward is rejected at config validation
(`src/marli/train/credit.py`).

**Regime:** untrained `tinker:Qwen/Qwen3.5-4B` ([entity](../entities/qwen3-5-4b.md))
on Tinker. Benchmarks are [AIME25](../entities/aime-2025.md) and
[HMMT25](../entities/hmmt-feb-2025.md), n=30 each. G=1, one seed, lockstep,
math-verify, and a 32,768-token episode cap. Every voting peer is truncated
(see [budget splitting](budget-splitting-truncation.md)). Sources:
[pilot README](../../sources/compute-matched-baselines-pilot.md) and
[report tables](../../sources/compute-matched-baselines-results.md).

## Measured [pilot]

| Cell | AIME25 voted (Wilson 95%) | AIME25 oracle | lost | HMMT25 voted (Wilson 95%) | HMMT25 oracle | lost |
|---|---|---|---|---|---|---|
| [sc4](../entities/protocol-independent.md) | 0.367 [0.219, 0.545] | 0.367 | 0/30 | 0.200 [0.095, 0.373] | 0.367 | 5/30 |
| [swarm4](../entities/protocol-swarm.md) | 0.333 [0.192, 0.512] | 0.533 | 6/30 | 0.233 [0.118, 0.409] | 0.333 | 3/30 |
| [debate3](../entities/protocol-debate.md) | 0.133 [0.053, 0.297] | 0.200 | 2/30 | 0.033 [0.006, 0.167] | 0.067 | 1/30 |
| [single](../entities/protocol-single.md) (reference) | 0.533 [0.361, 0.698] | = acc | — | 0.533 [0.361, 0.698] | = acc | — |

"lost" = oracle − voted, in episodes. It counts episodes where a peer was right
and the submitted answer was not.

## Reading

- Aggregation throws away correct answers in 4 of the 6 voting cells, up to
  6/30 episodes (swarm4 on AIME25). [pilot]
- **Even the oracle does not beat `single`.** swarm4's AIME25 oracle (0.533)
  equals `single`'s accuracy. sc4's HMMT25 oracle (0.367) is below it. At this
  budget, better aggregation alone would not turn a parallel protocol into a
  win. The truncated peers are the binding constraint. [pilot]
- `oracle_any` is a best-of-N upper bound that uses N draws. It is not an
  achievable accuracy and not a compute-matched number.
- swarm4 (shared scratchpads, notify delivery) vs sc4 (no visibility) on
  AIME25: oracle 0.533 vs 0.367, but voted 0.333 vs 0.367. It is tempting to
  say that visibility diversifies peers but the vote does not use it. At n=30,
  G=1 this difference cannot be told apart from noise. [open]

## Open questions

- Aggregators other than a plain vote. The `swarm_n4_finalizer` config
  (a finalizer reads the committed pads after all peers finish) exists but has
  not been run. [open]
- Whether the gap persists when peers are not truncated (64k budget, or fewer
  peers). [open]
- G≥2 to report maj@k and avg@k alongside the per-episode vote. [open]

See also: [compute-matching](compute-matching.md),
[the synthesis](../syntheses/agent-systems-vs-single-matched-compute.md).
