---
type: concept
title: "Time-sharing GPUs between the sampler and the learner"
description: "With vLLM on one GPU and the LoRA learner on the other, each GPU idles for about half of every RL step; putting both on both GPUs (vLLM tensor-parallel with sleep mode, the learner data-parallel) cut a Qwen3.8-27B relay step on 2×H200 from about 25 to about 14–15 min, averaged 14.0 min/step over a 30-step run, and kept logprobs exact [partial]."
resource: experiments/2026-09-25_sacrifice-relay/bench/serve/tp2_mtp2_sleep.yaml
tags: [throughput, memory, vllm, sleep-mode, data-parallel, tensor-parallel, training, ops, local-backend]
timestamp: 2026-10-01
---

# Time-sharing GPUs between the sampler and the learner

**The problem.** On-policy RL alternates two phases: sample rollouts, then
train on them. If the sampler (vLLM) has GPU 0 and the learner has GPU 1, each
GPU sits idle for about half of every step. [partial]

**The fix ("option 2").** Both phases use both GPUs, one after the other.

- **vLLM runs tensor-parallel across both GPUs (TP2)**, meaning one model
  split across them. It uses [MTP speculative decoding](speculative-decoding-mtp.md)
  and is capped at 45% of each GPU's memory.
- **The learner is data-parallel**, meaning a full resident copy of the base
  model on each GPU, each training on part of the batch. Microbatches are
  ordered longest first.
- **vLLM sleeps while the learner trains.** Sleep level 1 moves vLLM's weights
  to CPU memory and frees its KV cache. vLLM wakes before the next adapter is
  loaded and checked. The backend never leaves the server asleep: closing and
  failed steps wake it (`src/marli/train/backends/local/backend.py`).

**Regime.** [Qwen3.8-27B](../entities/qwen3-8-27b.md), LoRA r=32, on the
[local backend](../entities/local-backend.md). Pod: 2×H200 SXM. The workload is
the `relay_team` training step, 4 repos × G=4 = 16 relays of 4 contributors
each, context up to 24k. Source:
[benchmark](../../sources/sacrifice-relay-throughput-bench.md).

## Feasibility spikes [partial]

These come from `sleep_spike.py` and `coresident_spike.py`, with vLLM at TP2 +
MTP + sleep mode and 0.45 utilization.

- **Sleep** took 14 s the first time and 0.7 s after that. GPU memory dropped
  from 64.5 to 5.8 GiB per GPU.
- **Wake** took 0.8–0.9 s. A loaded LoRA adapter survived, and logprobs were
  identical after waking (max difference 0.0) for both the adapter and the
  base model.
- **Co-residence:** the learner's resident model copy (≈52 GiB) fits beside
  the awake server, for 115 GiB on GPU 0.
- **Training with vLLM asleep:** a 23,571-token train step took 14–15 s
  (1,578–1,724 tok/s), peaking at 78.7 GiB.

## Integration test (`train_opt2`) [partial]

Same 4 repos × G=4 workload, 2 steps.

| | Step 0 | Step 1 (through the adapter) | Original layout |
|---|---|---|---|
| Sampling | 588 s (117 tok/s per agent) | 586 s (88 tok/s per agent) | 651 s / 818 s |
| Training | 305 s for 904k tokens (≈3,000 tok/s) | 244 s for 838k tokens | 838 s for 847k tokens (≈1,000 tok/s), peak 94.8 GiB |
| vLLM sleep / wake | 19.9 s / 1.2 s | 0.9 s / 1.1 s | — |
| `kl_sample_train`, mean IS ratio | 7e-4, 0.99995 | 6e-4, 1.00006 | 6e-4, 1.00003 |
| Adapter hot-load check (effect drift) | 0.022 nats | 0.022 nats | 0.014 nats |
| **Whole step** | **≈15 min** | **≈14 min** | ≈25–27 min |

- Training throughput rose about 3× (≈1,000 → ≈3,000 tok/s). That is more
  than the 2× a second GPU alone would give, and the source does not say why.
- Sampling improved less than decode speed. The sampling phase ends with the
  slowest relay: four contributors near their budgets, about 48k tokens in
  sequence. Tool and test time does not shrink either. A sequential protocol's
  sampling time is set by its critical path (see
  [compute-matching](compute-matching.md) on critical-path tokens).
- The adapter weights updated as expected, and the step-0 adapter matches the
  single-GPU run's.
- The run took 30.6 min end to end, including starting 2 learner replicas.

## In production [partial]

The overnight team-reward trial ran on this layout: 30 steps in 7 h, 14.0 min
per step, with no errors, restarts or stalls. Over all 30 steps,
`kl_sample_train` stayed at or below 7.5e-4 and adapter effect drift at
0.016–0.028 nats
([experiment 1](../../sources/sacrifice-relay-experiment-1.md)).

## How to turn it on

- **Server:** `enable_sleep_mode: true` (this also sets
  `VLLM_SERVER_DEV_MODE=1`, which the sleep and wake endpoints need),
  `tensor_parallel_size: 2`, and `gpu_memory_utilization: 0.45`. See
  `bench/serve/tp2_mtp2_sleep.yaml`.
- **Training:** the `train rl` runtime fields `local_devices` (for example
  `[cuda:0, cuda:1]`, for a data-parallel learner) and
  `local_sleep_sampler: true`. Runtime fields are excluded from the config
  hash. The backend refuses `sleep_sampler` if the server was started without
  sleep mode.

## Prior practice (scimt)

scimt's TRL GRPO backend uses the same mechanism with vLLM inside the trainer
process. Its comments say level 2 discards weights and forces a full
re-push (≈49 GiB for a 26B model) every update, while level 1 offloads them to
host RAM (about 1 s to restore)
([GRPOOptions](../../sources/scimt-grpo-options.md)). The ≈1 s wake measured
here matches. scimt also warns that `peak_reserved` reads as virtual memory
under sleep mode (up to 218 GiB on a 141 GiB card), so judge memory by whether
the run completes
([throughput matrix](../../sources/scimt-rlvr-throughput-matrix.md)).

## Tensions

- The serve config's comment estimates the learner's resident copy at about
  62 GB per GPU. The spike measured ≈52 GiB (about 56 GB). The measurement
  supersedes the estimate.
- The first sleep took 14 s in the spike and 19.9 s in the integration test's
  step 0. Later sleeps took about 1 s in both.

See also: [local backend](../entities/local-backend.md).
