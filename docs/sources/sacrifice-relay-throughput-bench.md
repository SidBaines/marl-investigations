---
type: source
title: "Sacrifice relay: speed and memory benchmark (vLLM serving, MTP speculative decoding, two-GPU time-sharing) for Qwen3.8-27B on 2×H200"
description: "Measured vLLM serving speed of Qwen3.8-27B (TP1, MTP speculative decoding, TP2, DP2) at 16 and 128 concurrent agents, learner memory and step time, and the 'option 2' layout (vLLM TP2 + MTP + sleep mode, data-parallel learner) that cut a relay RL step on 2×H200 from about 25 to about 14–15 min with logprobs still exact."
resource: https://github.com/SidBaines/marl-investigations/blob/m6-harness/experiments/2026-09-25_sacrifice-relay/bench/README.md
source_date: 2026-09-25
status: partial
provenance: "Copied verbatim on 2026-10-01 from branch m6-harness @ 0bd4647, experiments/2026-09-25_sacrifice-relay/bench/README.md (unchanged since ea25ec4). History: scaffolding @ 61551db, benchmark results and option-2 feasibility spikes @ 74a8c9b, option-2 integration test @ ea25ec4, all 2026-09-25. No PR yet (m6-harness unmerged at ingest). Serving configs: bench/serve/*.yaml; spikes: bench/sleep_spike.py, bench/coresident_spike.py; samplers: bench/metrics.py, bench/episodes.py. Pod nu638jwugw3f2i (2×H200 SXM, SECURE, $9.18/hr), deleted 2026-09-26. Status is partial: direct measurements, but each configuration was measured once, on one pod, one model and one workload."
tags: [benchmark, throughput, memory, ops, vllm, speculative-decoding, mtp, sleep-mode, data-parallel, local-backend, qwen3.8-27b, runpod]
timestamp: 2026-10-01
---

# Speed and memory benchmark (before the sacrifice-relay runs)

Status: done (2026-09-25), including the option-2 integration test. Pod `nu638jwugw3f2i` (`marli-bench`), 2×H200 SXM,
SECURE, CUDA 13.0 host, $9.18/hr.

## Question

How fast can we sample Qwen3.8-27B for this study, and does the learner fit?

1. **Serving config.** Which vLLM setup is fastest? Candidates: the baseline
   (TP1 on GPU 0, bf16), MTP speculative decoding, data-parallel over both
   GPUs (eval phases only) and TP2 (a reference for a future time-shared
   layout). Two workloads are measured:
   - **~16 concurrent agents**, the relay training step: latency-bound,
     because contributors run in turn;
   - **~128 concurrent agents**, filter/gate/solo: throughput-bound.
2. **Learner.** What are peak memory and step time on GPU 1 at our context
   (24k), how long does an adapter sync take, and how much does serving
   through the LoRA adapter slow decoding?
3. **The first 100 filter problems**, kept for the study. This is
   `../run.sh filter 100`: a paused run that `../run.sh filter` later resumes
   for the other 700 without re-sampling these.

## Method

- `bench.sh serve <variant>` starts `serve/<variant>.yaml` and a watcher that
  samples vLLM `/metrics` and `nvidia-smi` every 5 s (`metrics.py`).
- `bench.sh load <variant> <concurrency> <n>` runs throwaway single-agent
  code rollouts (the filter setup) on the first n candidate problems, one
  episode each. It then summarises two things over the run's wall-clock
  window:
  - server metrics: generation tok/s, inter-token latency, prefix-cache hit
    rate, KV usage, preemptions and spec-decode acceptance;
  - rollout facts (`episodes.py`).
- Sampling is T = 1 and top-p 1 in every variant. MTP speculative decoding
  uses rejection sampling, so the sampled distribution is unchanged.

  Before MTP is used for training we still check `kl_sample_train` and the IS
  ratio on a training step, because the returned logprobs must be the target
  model's.
- Learner: 2 `train rl` steps of `relay_team` (4 repos × G = 4) on 4
  unfiltered repos built from the first 16 candidates. The learner runs on
  GPU 1 against the serving config chosen above.

## Deviations

- Install: identical pins, but every wheel was fetched in parallel from its
  index into a local directory and installed offline
  (`uv pip compile --emit-index-annotation`, sha256 checked per file). PyPI
  served under 1 MB/s per connection from this datacenter. Torch, NVIDIA and
  triton wheels come from `download.pytorch.org/whl/cu130`.
- MTP moved from port 8001 to 8010: RunPod's nginx holds 8001 and proxies to
  8000, which fooled the launcher's readiness check. The launcher is not fixed
  yet: it should also check that its child is alive.
- The 128-way loads crashed once on a sandbox cleanup race between two rollout
  processes (fixed in `58fe127`) and were resumed. Their throughput comes from
  the first attempt's window.

## Results

Episodes are single-agent code rollouts on the first n candidates. Speeds are
measured by vLLM (per-call decode = completion tokens / call latency, a lower
bound).

| Serving (one GPU unless noted) | Agents | Per-agent decode p50 | Total gen tok/s | Mean episode | Prefix-cache hits |
|---|---|---|---|---|---|
| bf16, TP1 (baseline) | 16 | 48 tok/s | 504 | 130 s | 71% |
| + MTP (2 draft tokens, 63% accepted) | 16 | 85 tok/s | 898 | 81 s | 61% |
| **TP2 across both GPUs + MTP** | 16 | **132 tok/s** | 1,478 | 53 s | 56% |
| bf16, TP1 | 128 (≈56 in vLLM on average) | 27 tok/s | 1,543 | 225 s | 69% |
| + MTP | 128 | 27 tok/s | 1,623 | 219 s | 39% (KV 97% full) |
| DP2 (one engine per GPU), kept filter run | 256 (147 mean) | — | 2,784 (both GPUs) | — | 33% (KV over-full) |

- **MTP**: 1.8× at relay concurrency; +5% at 128 concurrency, where it fills
  the KV cache. Logprobs stay exact for training: on the first training step
  `kl_sample_train` = 6e-4 and the mean IS ratio = 1.00003.
- **Client cap**: the vLLM HTTP client capped in-flight requests at 100, httpx's
  default (fixed in `0783671`). Before the fix, 256-way runs peaked at 100.
- **Per-GPU concurrency**: about 80–100 per GPU is best for high-concurrency
  evals. Above that the KV cache overflows and prefix-cache hits collapse.
- **vLLM startup**: 320 s cold, 120–275 s warm. TP1 KV capacity is 954k
  tokens at 0.9 utilization; with MTP it is 771k.

**Relay training step, original layout** (MTP vLLM on GPU 0, learner on GPU 1;
4 unfiltered repos × G = 4):

| | Step 0 (base) | Step 1 (through the LoRA adapter) |
|---|---|---|
| Sampling | 651 s, 602k tokens, 74 tok/s per agent | 818 s, 591k tokens, 62 tok/s per agent |
| Training | 838 s for 847k datum tokens (≈1,000 tok/s), peak 94.8 GiB | (stopped to free the GPUs) |
| Adapter hot-load check | effect drift 0.014 nats (tolerance 0.05) | — |

Each GPU idles for about half of every step (sampling uses only GPU 0,
training only GPU 1).

**Option 2 feasibility** (`sleep_spike.py`, `coresident_spike.py`; vLLM TP2 +
MTP + sleep mode at 0.45 utilization):
- Sleep (level 1): 14 s the first time, 0.7 s after that. GPU memory drops from
  64.5 to 5.8 GiB per GPU.
- Wake: 0.8–0.9 s. A loaded LoRA adapter survives, and logprobs are identical
  after waking (max difference 0.0) for both adapter and base.
- The learner's resident model copy (≈52 GiB) fits beside the awake server:
  115 GiB on GPU 0.
- With vLLM asleep, a 23,571-token train step took 14–15 s (1,578–1,724
  tok/s), peaking at 78.7 GiB.
- Projected relay step with option 2: ≈13–14 min, down from ≈25 min.

**Option 2 integration test** (`train_opt2`). Same 4 repos × G = 4 as above.
vLLM runs TP2 + MTP + sleep mode on both GPUs. The learner is data-parallel,
one resident replica per GPU, with longest-first microbatches.

| | Step 0 | Step 1 (through the adapter) | Original layout (for comparison) |
|---|---|---|---|
| Sampling | 588 s (117 tok/s per agent) | 586 s (88 tok/s per agent) | 651 s / 818 s |
| Training | 305 s for 904k tokens (≈3,000 tok/s) | 244 s for 838k tokens | 838 s for 847k tokens |
| vLLM sleep / wake | 19.9 s / 1.2 s | 0.9 s / 1.1 s | — |
| `kl_sample_train`, mean IS ratio | 7e-4, 0.99995 | 6e-4, 1.00006 | 6e-4, 1.00003 |
| Adapter hot-load check (effect drift) | 0.022 nats | 0.022 nats | 0.014 nats |
| **Whole step** | **≈15 min** | **≈14 min** | ≈25–27 min |

- The run took 30.6 min end to end, including 2 learner replicas starting up.
- Adapter weights updated as expected: max |B| was 2e-5 after step 0 and 4e-5
  after step 1. The step-0 adapter matches the single-GPU run's.
- Sampling improves less than decode speed: the phase ends with the slowest
  relay (four contributors near the budget, about 48k tokens in sequence),
  and tool and test time does not shrink.

**Behaviour** (untrained model, 16 relays = 64 contributors, unfiltered
problems):
- Half of all contributors hit the 12,288-token budget; 37.5% never ran CI.
- 3 of the 38 contributors with a real choice (they started without the rule
  and ran CI) chose to reveal the rule: about 8%, plausibly 3–21%.

**Filter, first 100 of 800 problems** (kept; paused):
- 400/400 episodes.
- Passes out of 4: 0 → 24 problems, 1 → 9, 2 → 10, 3 → 21, 4 → 36. That puts
  40 of 100 in the kept band, projecting about 320 problems and about 80 relay
  repos.

**Spend**: about $26 up to 16:55 UTC (pod `nu638jwugw3f2i`, $9.18/hr).
