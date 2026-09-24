---
type: entity
title: "Protocol: debate (baseline only)"
description: N peers publish replies over lockstep rounds (push delivery, no tools, final text, full vote); kept only as a baseline.
resource: src/marli/interact/configs/debate_n3_r2.yaml
tags: [protocol, debate, baseline, voting]
timestamp: 2026-09-24
---

# Protocol: `debate` (baseline only)

- **Config:** `debate_n3_r2` has 3 peers and 2 lockstep rounds. Replies are
  published with push delivery. There are no tools; answers come from final
  text and a full vote. `max_ticks` equals the number of rounds. Implementation:
  a [swarm](protocol-swarm.md) preset in
  `src/marli/interact/protocols/presets.py`.
- **Role:** a baseline only (`CLAUDE.md`). It is not a system under study.
- **In the compute-matched pilot (cell `debate3`):** 3 × 10,922 per agent,
  with a per-round call cap of 5,461.
  - AIME25: 0.133 [0.053, 0.297], answered 0.233, lift −0.400
    [−0.567, −0.233] (0/12).
  - HMMT25: 0.033 [0.006, 0.167], answered 0.067, lift −0.500
    [−0.667, −0.333] (0/15).

  The per-round cap truncates reasoning before an answer. This is a
  budget-design problem for thinking models, not evidence against debate
  ([unanswered episodes](../concepts/unanswered-episodes.md)). Regime:
  untrained [Qwen3.5-4B](qwen3-5-4b.md) on Tinker, n=30, G=1. [pilot]
- Needs per-round caps that fit a thinking model before it can serve as a
  baseline. [open]
