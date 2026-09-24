---
type: entity
title: "Protocol: swarm (coordinator-free)"
description: N parallel peers sharing versioned scratchpads (notify or push delivery), each with its own answer and credit, aggregated by vote or a separate finalizer.
resource: src/marli/interact/configs/swarm_n4.yaml
tags: [protocol, swarm, scratchpads, voting]
timestamp: 2026-09-24
---

# Protocol: `swarm` (coordinator-free)

- **Configs:**
  - `swarm_n4`: 4 peers with `delivery.mode: notify`, aggregated by a
    verifier-equivalent vote.
  - `swarm_n4_push`: peers receive the latest scratchpad contents directly.
  - `swarm_n4_finalizer`: a fresh finalizer reads every peer's latest pad and
    submission.

  Implementation: `src/marli/interact/protocols/swarm.py`. Peers keep
  independent answers and credit.
- **In the compute-matched pilot (cell `swarm4` = `swarm_n4`):** 4 × 8,192 per
  agent (call ≤ 8,192).
  - AIME25: 0.333 [0.192, 0.512], oracle 0.533, lift −0.200 [−0.367, −0.067]
    (0/6, p=0.03).
  - HMMT25: 0.233 [0.118, 0.409], oracle 0.333, lift −0.300
    [−0.467, −0.133] (0/9, p=0.004).
  - Mean `total_gen` was 27.2k / 29.2k, within 10% of `single` on both
    benchmarks, so these are compute-matched losses. `cp_tokens` was
    8.8k / 9.8k.

  Regime: untrained [Qwen3.5-4B](qwen3-5-4b.md) on Tinker, n=30, G=1. See
  [budget splitting](../concepts/budget-splitting-truncation.md) and
  [aggregation loss](../concepts/aggregation-loss.md). [pilot]
- The push and finalizer variants have not been run. [open]
