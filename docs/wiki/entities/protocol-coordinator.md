---
type: entity
title: "Protocol: coordinator + workers"
description: A coordinator spawns fresh-context workers via spawn_workers and composes their return_report outputs; the RL smoke trains it with a shared or per-role LoRA.
resource: src/marli/interact/configs/coordinator_default.yaml
tags: [protocol, coordinator, workers, delegation]
timestamp: 2026-09-24
---

# Protocol: `coordinator` + workers

- **Configs:**
  - `coordinator_default`: coordinator tools `spawn_workers` and `submit`;
    worker tool `return_report`; no worker scratchpads.
  - `coordinator_scratch`: workers also write scratchpads that the coordinator
    can read.

  Implementation: `src/marli/interact/protocols/coordinator.py`. Spawning
  blocks, and workers start with fresh contexts.
- **In the compute-matched pilot (cell `coordinator`):** coordinator 16,384 +
  ≤ 4 workers × 4,096 (`spawn: max_per_call 4, max_total 4`).
  - AIME25: 0.567 [0.392, 0.726], lift +0.033 [−0.067, +0.133] (2/1,
    p=1.00).
  - HMMT25: 0.333 [0.192, 0.512], lift −0.200 [−0.400, +0.000] (2/8,
    p=0.11).
  - Mean `total_gen` was 15.8k / 17.7k, about 40% below `single`, so these are
    **not compute-matched**. Mean LM calls were 4.3 / 6.1.

  Regime: untrained [Qwen3.5-4B](qwen3-5-4b.md) on Tinker, n=30, G=1. See
  [compute-matching](../concepts/compute-matching.md). [pilot]
- **In the Tinker RL smoke:** `coord_shared` (one LoRA seats coordinator and
  workers) vs `coord_perrole` (`coord` and `work` LoRAs). Both ran 3 steps.
  This is a plumbing check only; see
  [zero-variance groups](../concepts/zero-variance-groups.md) for why the
  per-role evidence is weak. [pilot]
