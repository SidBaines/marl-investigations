---
type: entity
title: Tinker (backend)
description: "Hosted sampling and LoRA-training backend; marli policy refs `tinker:<model>` and `backend: tinker` learners, with exact sampled ids, raw logprobs and versioned sampler sync."
resource: src/marli/train/backends/tinker.py
tags: [backend, tinker, training, sampling, lora]
timestamp: 2026-09-24
---

# Tinker (backend)

One of marli's two training/sampling backends. The other is local: a PEFT
multi-LoRA learner with vLLM on RunPod.

- **Sampling:** policy ref `tinker:<hf id>` (`src/marli/policy/tinker.py`)
  keeps the exact sampled ids and raw logprobs. Prompts are built by the
  model's tinker-cookbook renderer. Trainable seats must sample at T=1, top_p=1,
  top_k=−1.
- **Training:** `learners: {name: {backend: tinker, base_model, rank,
  learning_rate}}` (`src/marli/train/backends/tinker.py`). Each LoRA has its
  own optimizer. Each step runs `forward_backward` and then `optim_step`,
  reports `loss:sum` and [`kl_sample_train`](../concepts/on-policy-check.md),
  and syncs a new sampler version (`save_weights_for_sampler`). Checkpoint
  manifests record both Tinker state and sampler paths.
- **Pricing:** per model, in the registry (`tinker_prices`, USD / 1M tokens;
  cached prefill 80% off; `src/marli/budget.py`). Spend runs under `max_usd`.
- **Measured:** the RL smoke validated the chain end to end on
  [Qwen3.5-4B](qwen3-5-4b.md) LoRA r=32, 3 steps × 4 runs, $0.42
  ([source](../../sources/tinker-rl-smoke.md)). The compute-matched pilot
  sampled 420 episodes for $14.00
  ([source](../../sources/compute-matched-baselines-pilot.md)). [pilot]
