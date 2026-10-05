---
type: concept
title: "On-policy check: kl_sample_train"
description: "The mean sampler-minus-trainer logprob over action tokens; near zero means the trainer scored exactly the token ids the sampler produced; measured 2e-4 to 1e-3 on Tinker (Qwen3.5-4B), at most 7.6e-4 over up to 60 local-backend steps (Qwen3.8-27B) and at most 2.0e-3 over 80 steps on a mixture-of-experts model (Qwen3.6-35B-A3B), where the fixed-probe hot-load check fails (about 0.17 nats, from near-tied expert routing) and had to be loosened to 0.5."
resource: src/marli/train/backends/tinker.py
tags: [training, rl, on-policy, token-level, tinker, local-backend, diagnostics, moe]
timestamp: 2026-10-05
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
  engines show on the base model. The step fails above 0.05 nats by default
  (`src/marli/train/backends/local/backend.py`). The tolerance is set by the
  runtime field `local_adapter_check_tol` (4c34b0f).

## Longer local runs, two models (2026-10-02 to 10-05) [partial]

All on the local backend, 2×H200, LoRA rank 32, importance-sampling loss, the
code_rules relay, one seed per run. Source:
[study report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md)
(§3, §5) and the experiment READMEs.

| Model | Runs | Steps | `kl_sample_train` | Mean IS ratio |
|---|---|---|---|---|
| [Qwen3.8-27B](../entities/qwen3-8-27b.md) | experiment 2 team | 60 (30 + resumed 30) | ≤ 7.4e-4 every step | 1.000 ± 0.0002 |
| Qwen3.8-27B | experiment 2 individual, 2.1 | 30 each | ≤ 7.6e-4 | 1.000 |
| [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md) | experiment 2 team | 80 | ≤ 2.0e-3 | 1.000 |
| Qwen3.6-35B-A3B | experiment 2.1; experiment 3 team and individual | 31 each | ≤ 1.9e-3; ≤ 1.8e-3 | 1.000 |

- **Resuming keeps agreement.** The 27B team run resumed from its step-29
  checkpoint on a new pod, with no change in the metric. [partial]
- **The A3B sits about 3× higher than the 27B,** still far below the 5e-3
  abort line that experiments 2 and 3 used. Why it is higher is not
  explained. [open]

## Fixed-probe check on a mixture-of-experts model [partial]

On Qwen3.6-35B-A3B, the hot-load check reported about **0.17 nats** of
adapter effect drift, against about 0.03 for the 27B. That failed the 0.05
tolerance, so the A3B preflight said NO-GO, but only on the fixed-probe
checks. Sampled-token agreement was fine: `kl_sample_train` about 1.6e-3, IS
ratio 1.000.

- **The likely cause: near-tied expert routing.** Each token picks its top 8
  of 256 routed experts. On the fixed probe, which is off-policy text the
  model did not sample, some tokens have near-tied router scores. A tiny
  numerical difference between vLLM and the learner can then pick a different
  expert set, so the two engines' logprobs differ for reasons unrelated to the
  adapter. On text the model actually sampled, the agreement holds. This
  explanation comes from the source and was not tested directly. [pilot]
- **What was done.** Runs on the A3B set `local_adapter_check_tol=0.5` and
  keep the per-step `kl_sample_train > 5e-3` abort as the real guard. The
  registry entry records this.
- **What it costs.** At 0.5 nats the probe check would let through a wrong
  adapter whose effect is smaller than that. A mix-up would then show only in
  `kl_sample_train`, after a batch had been sampled from the wrong policy.
  [open]
- **Generalisation.** One MoE model, one serving stack (vLLM, routed experts
  and router frozen, LoRA on attention and the shared expert). A probe drawn
  from the policy's own samples, or a routing-insensitive comparison, might
  restore the check for MoE models. Untested. [open]

## Caveats

- This is a mean logprob difference (a k1-style estimator). It can sit near
  zero while per-token differences cancel. ~~No threshold has been set for
  what a real mismatch looks like~~ Experiments 2 and 3 abort a run above
  5e-3, but that line was not calibrated: no mismatch was deliberately
  induced to see where a real one lands. [open]
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
- A failing fixed-probe check does not always mean the policy is off: on the
  MoE model the probe failed while sampled tokens agreed (above).

See also: [Tinker](../entities/tinker.md),
[local backend](../entities/local-backend.md).
