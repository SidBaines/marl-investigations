---
type: concept
title: "On-policy check: kl_sample_train"
description: The mean sampler-minus-trainer logprob over action tokens; near zero means the trainer scored exactly the token ids the sampler produced; measured 2e-4 to 1e-3 in the Tinker RL smoke.
resource: src/marli/train/backends/tinker.py
tags: [training, rl, on-policy, token-level, tinker, diagnostics]
timestamp: 2026-09-24
---

# On-policy check: `kl_sample_train`

**Definition.** For RL losses, each learner step reports `kl_sample_train`.
This is the mean, over action positions (mask = 1), of the sampler's recorded
logprob minus the trainer's logprob for the same token from `forward_backward`.
It is `None` for the cross-entropy (SFT) loss. The Tinker implementation is in
`src/marli/train/backends/tinker.py`, and the fake backend reports 0.0.

**Why it matters.** marli's token-level RL invariants (`CLAUDE.md`) say that
datums use exactly the ids the policy saw, and that trainable seats sample at
T=1, top_p=1, top_k=−1. If history is re-rendered or re-tokenized, the
renderer is wrong, or sampling runs at a non-unit temperature, the trainer
scores different tokens or a different distribution than the sampler. The gap
then moves away from zero. This metric is the cheap per-step check for that.
scimt's TRL backend documents the same issue: with top_p/top_k truncation,
"a truncated rollout distribution is not exactly the policy it is scored
against", because the importance ratio uses full-softmax logprobs
([GRPOOptions](../../sources/scimt-grpo-options.md)).

## Measured [pilot]

In the [Tinker RL smoke](../../sources/tinker-rl-smoke.md), `kl_sample_train`
stayed between 2e-4 and 1e-3 at every step of all four runs.

- **Regime:** [Qwen3.5-4B](../entities/qwen3-5-4b.md) with LoRA r=32 on
  [Tinker](../entities/tinker.md), lr 1e-5, and an importance-sampling loss
  (sum-reduced, no grad clip). 3 steps, B=2 tasks × G=2 episodes, on 32
  shuffled [POLARIS-53K](../entities/polaris-53k.md) tasks (seed 0).
  Protocols were [coordinator](../entities/protocol-coordinator.md) (shared and
  per-role LoRA) and [multi-session notes](../entities/protocol-multi-session.md)
  (credit all and last). Budgets were small: episode 8,192, call 3,072, ctx
  16,384.
- **Reading.** Sampler and trainer agreed on the exact token ids, including in
  multi-agent (coordinator + workers) and multi-segment (multi-session)
  episodes. Sampler versions bumped on every stepped learner, so later steps
  also exercised synced-sampler sampling. [pilot]

## Caveats

- This is a mean logprob difference (a k1-style estimator). It can sit near
  zero while per-token differences cancel. No threshold has been set for what
  a real mismatch looks like, and none was deliberately induced to calibrate
  one. [open]
- The run was tiny: 3 steps, few datums (see
  [zero-variance groups](zero-variance-groups.md)). It validates the plumbing
  and says nothing about learning.
- It checks sampler/trainer agreement on Tinker only. The local (PEFT + vLLM)
  path has not been measured. [open]

See also: [Tinker](../entities/tinker.md).
