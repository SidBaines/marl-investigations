---
type: entity
title: POLARIS-53K
description: POLARIS-Project/Polaris-Dataset-53K, a 53k-problem math RL training set with LaTeX answers and a per-task 7B pass-rate difficulty field; used by the Tinker RL smoke.
resource: src/marli/tasks/polaris_53k.yaml
tags: [dataset, math, training-data, rl]
timestamp: 2026-09-24
---

# POLARIS-53K

Registry entry: `src/marli/tasks/polaris_53k.yaml`.

| Field | Value |
|---|---|
| HF dataset | `POLARIS-Project/Polaris-Dataset-53K`, split `train` |
| Fields | `problem` → prompt, `answer`, `difficulty` (a 7B model's pass rate, e.g. `7/8`) |
| Answer format | LaTeX (math-verify) |
| License | Apache-2.0 |
| `commit_text` | false. Training data; do not commit text |

**Used in:** the [Tinker RL smoke](../../sources/tinker-rl-smoke.md), with 32
shuffled tasks (`max_n=32 shuffle=true seed=0`). The `difficulty` field could
be used to avoid [zero-variance groups](../concepts/zero-variance-groups.md).
[open]
