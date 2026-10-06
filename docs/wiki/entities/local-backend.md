---
type: entity
title: "Local backend (PEFT LoRA learner + vLLM)"
description: "marli's self-hosted training/sampling backend: a PEFT multi-LoRA learner plus a separately supervised vLLM server, with per-step adapter hot-load and a probe check (effect drift ≤ 0.05 nats by default), optional GPU time-sharing, pause and resume from per-step checkpoints, and mixture-of-experts support (routed experts frozen); validated for up to 60 steps of Qwen3.8-27B and 80 steps of Qwen3.6-35B-A3B, LoRA r=32 on 2×H200; also serves frozen adapters and chat-endpoint evals."
resource: src/marli/train/backends/local/backend.py
tags: [backend, local, training, sampling, lora, peft, vllm, runpod, ops, moe]
timestamp: 2026-10-05
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

**Longer runs (2026-10-01 to 10-05)**, same pod type and time-shared layout,
one seed each. Source:
[study report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md)
and the experiment READMEs. [partial]

| Model | Runs | Steps | `kl_sample_train` | Step time |
|---|---|---|---|---|
| Qwen3.8-27B | experiment 2 team (resumed once on a new pod), individual, 2.1 | 60, 30, 30 | ≤ 7.6e-4 | 13.3–13.4 min at 4 repos × 4 |
| [Qwen3.6-35B-A3B](qwen3-6-35b-a3b.md) | experiment 2 team | 80 | ≤ 2.0e-3 | 10.1 min at 8 repos × 4 |
| Qwen3.6-35B-A3B | experiment 2.1; experiment 3 team and individual | 31 each (paused) | ≤ 1.9e-3 | about 11.6 min at 8 × 4 |

None of these runs had errors, restarts or abort-rule triggers.

## Mixture-of-experts models (2026-10-02) [partial]

The learner trains [Qwen3.6-35B-A3B](qwen3-6-35b-a3b.md), and its routed
experts stay frozen (5ea9b20, 64dcf29).

- **Fused experts.** The routed experts are fused 3-D parameters
  (transformers 5.5.4), not linear layers, and the learner requires
  `grouped_mm`.
- **What the LoRA covers.** It targets attention, linear attention and the
  shared expert. The routed experts and the router are frozen.
- **Matching vLLM.** vLLM's `--lora-target-modules` must be restricted to the
  same list (`lora_target_modules` in the serve config). Otherwise vLLM
  silently ignores the weights for modules it did not wrap, and wraps the
  experts with zero adapters.
- **The fixed-probe hot-load check fails on this model.** It reports about
  0.17 nats of drift, from near-tied top-8 routing on off-policy probe text,
  while sampled-token agreement is fine. The runtime field
  `local_adapter_check_tol` (4c34b0f) sets the tolerance; the A3B runs used
  0.5 ([on-policy check](../concepts/on-policy-check.md)).
- **Training the routed experts** would need new learner code that adapts
  them identically in both engines. Not built.

## Pausing and resuming [partial]

- **Every step saves a full resume point on the pod.** For the A3B that is an
  adapter of about 160 MB and a trainer state of about 0.5 GB.
- **Pause only after a step's checkpoint manifest is written.** Stopping
  mid-step can leave a partial save.
- **On HF, keep the trainer states for steps 9, 19, 29 and the last step.**
  To resume, restore the run dir and the last state on a new pod, then rerun
  the same command with the same `--out`. The config hash must match: any
  later commit with the same composed config works.
- **Used for:** experiment 2's 29 → 59 continuation, and three runs paused on
  2026-10-05 (A3B 2.1, experiment 3 team and individual).
- **Uploads.** Hourly incremental HF uploads and a run's final upload share a
  `/tmp` stage dir. Stop the hourly ones before the final upload.

## Serving frozen adapters and chat evals (2026-10-02 to 10-05)

- **Frozen seats.** The relay's `opener_role` seats contributor 1 on a frozen
  vLLM adapter beside the learner's (e3f376a); experiment 2.1 used it.
  - Adapters loaded by hand must also be listed in `server.json` `models`, or
    the policy builder refuses them.
  - Keep them out of `adapters`, which is the learner's bookkeeping.
- **One server, two harnesses.** The reasoning parser
  (`--reasoning-parser qwen3`) and the tool-call parser
  (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`) leave
  `/v1/completions` untouched.
  - So the token-level relay eval and the chat-based
    [standard cooperation evals](standard-coop-evals.md) shared one
    data-parallel server per model.
  - A preflight checks the parsers, JSON mode and template parity.

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
  and the 30-step trial (2026-09-25 14:22 → 2026-09-26 07:03 UTC). The whole
  study to 2026-10-05, including the eval session, cost about $834.

## Ops notes added 2026-10-05 (from the study report §5)

- **Check GPU clocks under load on every new pod**
  (`nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks_event_reasons.active`).
  - On 10-01 one H200 sat in thermal slowdown at 86 °C and 345 MHz, so
    sampling ran 2.6× slower. The pod was replaced.
  - About 1,500–1,800 MHz at the 700 W cap is normal. Mild slowdown
    (79 °C, 1,800–1,965 MHz) cost about 3%.
- **Broken NVLink SHARP on some hosts.** vLLM TP2 failed with an NCCL
  "unhandled cuda error". `NCCL_NVLS_ENABLE=0` fixes it, and is now the
  default in every `run.sh`.
- **Relay agents overload an eval server sooner than single agents.**
  - About 65 relay agents per engine filled the KV cache: about 2.8k
    preemptions, then 600 s client timeouts, and 22 of 34 games failed.
  - Use about 30–40 relay agents per engine.
  - The single-agent difficulty filter (192 agents on two engines) survived
    about 3k preemptions per engine.
- **Relay games, HiddenBench and FAIRGAME are latency-bound.** Throughput
  comes from games in flight: raise them while KV usage is low (the A3B went
  from 36 to 60 relay games at 28% KV).
- **Pods have no Docker.** Sandboxes must be subprocess-based (see the
  `marli` Inspect sandbox in
  [standard cooperation evals](standard-coop-evals.md)).
- **Tension with the concurrency note above.** "80–100 concurrent agents per
  GPU" came from single-agent rollouts. Relay agents carry long, growing
  contexts, so the safe level is much lower (30–40 per engine). Size by
  workload, not by a single number. [partial]
