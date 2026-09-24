---
type: entity
title: "Protocol: single"
description: One solver agent with a submit tool; the baseline cell of every comparison.
resource: src/marli/interact/configs/single.yaml
tags: [protocol, baseline, single-agent]
timestamp: 2026-09-24
---

# Protocol: `single`

- **Config:** `src/marli/interact/configs/single.yaml` (`protocol: single`,
  empty config). One `solver` seat with `submit`. Implementation:
  `src/marli/interact/protocols/single.py`.
- **Role:** the baseline of every grid. Paired lift is reported against it.
- **In the compute-matched pilot:** budget 32,768 per episode (call ≤ 16,384).
  On AIME25 it scored 0.533 [0.361, 0.698] (answered 0.967; mean / p50
  `total_gen` 25.5k / 31.2k). On HMMT25 it scored 0.533 [0.361, 0.698]
  (answered 1.000; 29.0k / 31.8k). Regime: untrained
  [Qwen3.5-4B](qwen3-5-4b.md) on Tinker, n=30, G=1. [pilot]
- Its chains are themselves budget-bound at 32k
  ([budget splitting](../concepts/budget-splitting-truncation.md)). A `single`
  cell at lower budgets would give matched-`cp_tokens` baselines for the
  parallel cells. [open]
