---
type: entity
title: Qwen3.5-4B
description: Qwen/Qwen3.5-4B, a 4B thinking model (renderer qwen3_5) served and LoRA-trained on Tinker; the policy of the compute-matched baselines pilot and the Tinker RL smoke.
resource: src/marli/models/qwen3_5_4b.yaml
tags: [model, qwen, thinking, tinker]
timestamp: 2026-09-24
---

# Qwen3.5-4B

Registry entry: `src/marli/models/qwen3_5_4b.yaml` (name `qwen3_5_4b`).

| Field | Value |
|---|---|
| HF / Tinker id | `Qwen/Qwen3.5-4B` (policy ref `tinker:Qwen/Qwen3.5-4B`) |
| Family / renderer / tool format | `qwen3_5` / `qwen3_5` / `qwen3_5_xml` |
| Thinking | yes |
| Architecture | `Qwen3_5ForConditionalGeneration` (vision-language) |
| Context | registry `max_ctx` 32,768; `tinker_max_ctx` 65,536 |
| Default max tokens | 8,192 |
| Tinker prices (USD / 1M tokens, as of 2026-09-23) | prefill 0.33 (cached prefill 80% off), sample 1.01, train 0.74 |
| Local (PEFT + vLLM) | unverified: LoRA support on this VL architecture is still to be checked |

## Observed behaviour

- **Untrained, 32k episode cap:** `single` scored 0.533 [0.361, 0.698] on
  [AIME25](aime-2025.md) and 0.533 [0.361, 0.698] on
  [HMMT25](hmmt-feb-2025.md) (Tinker, n=30 each, G=1, math-verify, Wilson 95%).
  Single chains typically run to the cap (p50 `total_gen` 31.2k / 31.8k),
  which is why splitting the budget across agents hurts
  ([budget splitting](../concepts/budget-splitting-truncation.md)). [pilot]
- **Tinker LoRA r=32:** `kl_sample_train` was 2e-4 to 1e-3 in a 3-step RL
  smoke ([on-policy check](../concepts/on-policy-check.md)). [pilot]
- **Cost:** the 420-episode pilot (2 benchmarks × 7 cells × 30) cost $14.00,
  and the 4-run RL smoke cost $0.42.

Sources: [compute-matched baselines](../../sources/compute-matched-baselines-pilot.md),
[Tinker RL smoke](../../sources/tinker-rl-smoke.md).
