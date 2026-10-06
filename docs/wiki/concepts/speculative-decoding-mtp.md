---
type: concept
title: "MTP speculative decoding for on-policy RL sampling"
description: "The model's own multi-token-prediction head drafts 2 tokens and vLLM checks them by rejection sampling, so temperature-1 samples and returned logprobs stay exact; on Qwen3.8-27B it sped latency-bound relay sampling 1.8× at 16 agents, gave only +5% at 128 agents (where it costs KV-cache room), and kept kl_sample_train ≤ 7.5e-4 over 30 RL steps [partial]."
resource: experiments/2026-09-25_sacrifice-relay/bench/serve/mtp2.yaml
tags: [throughput, vllm, speculative-decoding, mtp, sampling, on-policy, ops, local-backend]
timestamp: 2026-10-01
---

# MTP speculative decoding for on-policy RL sampling

**What it is.** In *speculative decoding*, a cheap drafter proposes the next
few tokens. The full model then checks all of them in one forward pass and
keeps the prefix it accepts. *MTP* (multi-token prediction) uses a small extra
prediction layer that ships with the model as the drafter. For
[Qwen3.8-27B](../entities/qwen3-8-27b.md) it is a 1-layer head. vLLM accepts or
rejects drafts by *rejection sampling*, so accepted tokens follow exactly the
full model's distribution.

**Why it matters for RL.** marli's trainable seats sample at temperature 1,
top_p 1, top_k −1. The trainer must score the same distribution the sampler
drew from (`CLAUDE.md`, token-level RL invariants). A speed-up that changed the
distribution would make training silently off-policy. Rejection sampling
keeps the distribution unchanged in principle, but the logprobs vLLM returns
must also be the full model's, so this was checked on real training steps.

## Measured speed-up [partial]

Regime: vLLM, Qwen3.8-27B in bf16, one H200 per engine unless noted, 2 draft
tokens per step, T=1, top-p 1. The workload was single-agent code rollouts
(the filter setup) on the first n [DeepCoder](../entities/deepcoder.md)
candidates. Per-agent decode is completion tokens divided by call latency, a
lower bound. Source:
[benchmark](../../sources/sacrifice-relay-throughput-bench.md).

| Serving | Agents | Per-agent decode p50 | Total gen tok/s | Mean episode | Prefix-cache hits |
|---|---|---|---|---|---|
| bf16, TP1 (baseline) | 16 | 48 tok/s | 504 | 130 s | 71% |
| + MTP | 16 | 85 tok/s | 898 | 81 s | 61% |
| TP2 across both GPUs + MTP | 16 | 132 tok/s | 1,478 | 53 s | 56% |
| bf16, TP1 | 128 (≈56 in vLLM on average) | 27 tok/s | 1,543 | 225 s | 69% |
| + MTP | 128 | 27 tok/s | 1,623 | 219 s | 39% (KV 97% full) |

Terms: *TP1/TP2* means the model runs on one GPU, or is split across two
(tensor parallel). *Prefix-cache hits* is the share of prompt tokens whose
attention state was reused from earlier requests. The *KV cache* is the GPU
memory holding that state for in-flight sequences.

- **About 1.8× at relay concurrency.** 63% of drafted tokens were accepted.
- **+5% at 128 concurrency.** The source's explanation is the KV cache. MTP
  shrinks it from 954k tokens to 771k at 0.9 utilization (TP1), the cache fills
  (97%), and prefix-cache hits fall from 69% to 39%. The general reason for
  smaller gains at high load also applies: with many sequences in flight the
  GPU is already busy, so checking drafts is no longer nearly free.
- **Rule of thumb:** use MTP for latency-bound phases, such as the sequential
  relay's training step. It is roughly neutral for high-concurrency evals.
  About 80–100 concurrent agents per GPU is best there. Above that the KV cache
  overflows and prefix-cache hits collapse.

## Logprobs stay exact [partial]

Two numbers check this on each training step. `kl_sample_train` is the mean
sampler-minus-trainer logprob over the tokens the policy sampled. The *IS
ratio* is exp(trainer logprob − sampler logprob) per sampled token,
averaged. Exact agreement gives 0 and 1. See
[on-policy check](on-policy-check.md).

| Run (all with MTP) | `kl_sample_train` | Mean IS ratio |
|---|---|---|
| First training step, original layout | 6e-4 | 1.00003 |
| Option-2 integration test, steps 0 / 1 | 7e-4 / 6e-4 | 0.99995 / 1.00006 |
| Overnight trial, 30 steps | ≤ 7.5e-4 at every step | 1.0000 ± 0.0001 |

These are the same magnitude as Tinker without speculative decoding (2e-4 to
1e-3 for Qwen3.5-4B). That is a different model and backend, so compare
magnitudes only.

## Caveats

- One model, one pod, and each configuration measured once.
- Faster decoding does not shorten a relay step in proportion. The sampling
  phase ends with the slowest relay, and tool and test time does not shrink.
  See [GPU time-sharing](gpu-time-sharing.md).
- MTP was moved from port 8001 to 8010 because RunPod's nginx holds 8001. See
  the ops notes on the [local backend](../entities/local-backend.md).

See also: [local backend](../entities/local-backend.md).
