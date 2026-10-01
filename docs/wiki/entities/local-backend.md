---
type: entity
title: "Local backend (PEFT LoRA learner + vLLM)"
description: "marli's self-hosted training/sampling backend: a PEFT multi-LoRA learner plus a separately supervised vLLM server, with per-step adapter hot-load and a probe check (effect drift ≤ 0.05 nats), optional GPU time-sharing (vLLM sleep mode plus a data-parallel learner); validated at 30 RL steps of Qwen3.8-27B LoRA r=32 on 2×H200."
resource: src/marli/train/backends/local/backend.py
tags: [backend, local, training, sampling, lora, peft, vllm, runpod, ops]
timestamp: 2026-10-01
---

# Local backend (PEFT LoRA learner + vLLM)

One of marli's two training and sampling backends. The other is
[Tinker](tinker.md). It was built in M4
([PR #6](https://github.com/SidBaines/marl-investigations/pull/6), "PEFT
multi-LoRA + vLLM hot-load, validated on an H100").

## How it works

- **Learner.** A PEFT multi-LoRA learner (`src/marli/train/backends/local/`).
  Torch is pinned pod-side (`requirements/pod-*.txt`). Learners are declared as
  `learners: {name: {backend: local, base_model, rank, learning_rate}}`. It
  uses the same sum-reduced losses as Tinker
  (`src/marli/train/backends/local/losses.py`).
- **Sampler.** vLLM runs as a separate supervised server process from its own
  venv (`marli serve vllm`, with `python: /opt/vllm/bin/python` in the study).
  Policies refer to it as `vllm:@<server.json>#<model>`. Serve-config keys
  used in the study: `cuda_visible_devices`, `tensor_parallel_size`,
  `gpu_memory_utilization`, `max_model_len`, `max_lora_rank`, `max_loras`,
  `learner_ranks`, `extra_args` and `enable_sleep_mode`.
- **Adapter publication.** After each step, the new adapter is exported and
  hot-loaded into vLLM under a versioned name, which protects the prefix
  cache. The old one is then evicted.
- **Adapter hot-load check.** A fixed probe text is scored by both vLLM and
  the learner, with and without the adapter.
  - *Effect drift* is the mean absolute difference, per token, between the two
    engines' estimates of the adapter's effect (adapted minus base logprob).
    Comparing effects cancels the ≈0.03-nat kernel difference the engines show
    even on the base model.
  - The step fails if drift exceeds `adapter_check_tol` (0.05 nats), or if vLLM
    ignores the adapter.
- **GPU time-sharing (optional).** `local_sleep_sampler: true` and
  `local_devices: [cuda:0, cuda:1]` are runtime fields of `train rl`. vLLM
  sleeps during training, and the learner is data-parallel. See
  [GPU time-sharing](../concepts/gpu-time-sharing.md).

## Measured

All on [Qwen3.8-27B](qwen3-8-27b.md), LoRA r=32, 2×H200 SXM. Sources:
[benchmark](../../sources/sacrifice-relay-throughput-bench.md) and
[experiment 1](../../sources/sacrifice-relay-experiment-1.md). [partial]

| Run | Layout | `kl_sample_train` | Mean IS ratio | Effect drift | Step time |
|---|---|---|---|---|---|
| Benchmark, step 0 | vLLM (MTP) on GPU 0, learner on GPU 1 | 6e-4 | 1.00003 | 0.014 nats | ≈25–27 min |
| Option-2 test, steps 0 / 1 | both GPUs, time-shared | 7e-4 / 6e-4 | 0.99995 / 1.00006 | 0.022 / 0.022 nats | ≈15 / ≈14 min |
| Overnight trial, 30 steps | both GPUs, time-shared | ≤ 7.5e-4 every step | 1.0000 ± 0.0001 | 0.016–0.028 nats | 14.0 min average |

The overnight trial had no errors, restarts or stalls. It wrote a checkpoint
after every step, and it resumes with the same `--out`. See
[on-policy check](../concepts/on-policy-check.md) for what the first two
columns mean.

## Ops notes (from the benchmark and trial)

- **Installs.** PyPI served under 1 MB/s per connection from the RunPod
  datacenter. Instead, every wheel was fetched in parallel into a local
  directory and installed offline (`uv pip compile
  --emit-index-annotation`, sha256 checked per file). Torch, NVIDIA and triton
  wheels came from `download.pytorch.org/whl/cu130`. On the pod, use
  `uv run --no-sync`. A plain `uv run` re-syncs to `uv.lock` and drops the
  pod's torch pins (`run.sh` comment).
- **Port 8001.** RunPod's nginx holds port 8001 and proxies it to 8000. This
  fooled the launcher's readiness check. The launcher should also check that
  its child process is alive, which was not yet fixed as of the benchmark.
- **Client cap.** The vLLM HTTP client capped in-flight requests at 100
  (httpx's default) until `0783671`.
- **Concurrency.** About 80–100 concurrent agents per GPU is best for
  high-concurrency evals. Above that the KV cache overflows and prefix-cache
  hits collapse. For speed by serving config, see
  [MTP speculative decoding](../concepts/speculative-decoding-mtp.md).
- **Sandbox race.** A sandbox cleanup race between two rollout processes
  crashed 128-way loads once (fixed in `58fe127`).
- **vLLM startup.** 320 s cold, 120–275 s warm.
- **Persistence.** Pods are not storage. Adapters for every step, trainer
  states every 5 steps, and metrics went to the public HF repo
  `sidbaines/amber-baton` (no prompts or transcripts). Rollouts, which contain
  problem text, went to the dev box. The pod was deleted after both copies
  were verified.
- **Cost.** A 2×H200 SXM SECURE pod at $9.18/hr. The whole sacrifice-relay
  session cost about $153: benchmark, filter, option-2 test, continue test,
  and the 30-step trial (2026-09-25 14:22 → 2026-09-26 07:03 UTC).
