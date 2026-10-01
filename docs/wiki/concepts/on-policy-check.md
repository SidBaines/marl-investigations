---
type: concept
title: "On-policy check: kl_sample_train"
description: "The mean sampler-minus-trainer logprob over action tokens; near zero means the trainer scored exactly the token ids the sampler produced; measured 2e-4 to 1e-3 in the Tinker RL smoke (Qwen3.5-4B) and at most 7.5e-4 over 30 local-backend steps with MTP sampling (Qwen3.8-27B), where the adapter hot-load check adds a second, serving-side test."
resource: src/marli/train/backends/tinker.py
tags: [training, rl, on-policy, token-level, tinker, local-backend, diagnostics]
timestamp: 2026-10-01
---

# On-policy check: `kl_sample_train`

**Definition.** For RL losses, each learner step reports `kl_sample_train`.
This is the mean, over action positions (mask = 1), of the sampler's recorded
logprob minus the trainer's logprob for the same token from `forward_backward`.
It is `None` for the cross-entropy (SFT) loss. The Tinker implementation is in
`src/marli/train/backends/tinker.py`, the local one in
`src/marli/train/backends/local/losses.py`, and the fake backend reports 0.0.
The local backend also reports the mean **IS ratio** (importance-sampling
ratio): exp(trainer logprob − sampler logprob) per action token, averaged.
Exact agreement gives `kl_sample_train` = 0 and an IS ratio of 1.

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

## Measured on Tinker [pilot]

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

## Measured on the local backend [partial]

- **Regime:** [Qwen3.8-27B](../entities/qwen3-8-27b.md) with LoRA r=32 on the
  [local backend](../entities/local-backend.md) (PEFT learner + vLLM),
  importance-sampling loss, 2×H200. Sampling used
  [MTP speculative decoding](speculative-decoding-mtp.md) (2 draft tokens,
  rejection sampling). The workload was the
  [code_rules relay](../entities/env-code-rules.md) (N=4, 4 repos × G=4, up to
  24k context). Sources:
  [benchmark](../../sources/sacrifice-relay-throughput-bench.md) and
  [experiment 1](../../sources/sacrifice-relay-experiment-1.md).

| Run | `kl_sample_train` | Mean IS ratio | Adapter effect drift |
|---|---|---|---|
| Benchmark step 0 (vLLM on GPU 0, learner on GPU 1) | 6e-4 | 1.00003 | 0.014 nats |
| Option-2 test, steps 0 / 1 ([time-shared GPUs](gpu-time-sharing.md)) | 7e-4 / 6e-4 | 0.99995 / 1.00006 | 0.022 / 0.022 nats |
| Overnight trial, 30 steps, lr 4e-5 | ≤ 7.5e-4 at every step | 1.0000 ± 0.0001 | 0.016–0.028 nats |

- **Reading.** Speculative decoding, putting vLLM to sleep and waking it, and
  a data-parallel learner all kept sampler and trainer in agreement. The
  values are the same magnitude as Tinker's, with a different model and
  backend, so compare magnitudes only. [partial]
- **A second check: adapter effect drift.** If vLLM served the wrong
  adapter, `kl_sample_train` would show it only after a whole batch had been
  sampled from it. The hot-load check catches it before sampling. After every
  hot-load, the local backend scores a fixed probe text in both engines, with
  and without the adapter. It then compares the adapter's *effect* (adapted
  minus base logprob), which cancels the ≈0.03-nat kernel difference the
  engines show on the base model. The step fails above 0.05 nats
  (`src/marli/train/backends/local/backend.py`).

## Caveats

- This is a mean logprob difference (a k1-style estimator). It can sit near
  zero while per-token differences cancel. No threshold has been set for what
  a real mismatch looks like, and none was deliberately induced to calibrate
  one. [open]
- The Tinker smoke was tiny: 3 steps, few datums (see
  [zero-variance groups](zero-variance-groups.md)). It validates the plumbing
  and says nothing about learning.
- ~~It checks sampler/trainer agreement on Tinker only. The local (PEFT +
  vLLM) path has not been measured. [open]~~ The local path has now been
  measured (above) for Qwen3.8-27B over 30 steps. [partial]
- A healthy value does not mean training is working. In the 30-step trial,
  sampler and trainer agreed at every step, yet the behaviour the advantage
  favoured became rarer
  ([rewarded but not learned](rewarded-choice-not-learned.md)).

See also: [Tinker](../entities/tinker.md),
[local backend](../entities/local-backend.md).
