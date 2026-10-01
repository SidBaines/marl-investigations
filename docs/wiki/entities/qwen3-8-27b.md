---
type: entity
title: Qwen3.8-27B
description: "Qwen/Qwen3.8-27B, a 27B thinking model (Qwen3.5 architecture, renderer qwen3_8_medium in the sacrifice relay) with its own MTP draft head; LoRA r=32 trained on the local backend (PEFT + vLLM) on 2×H200 in the sacrifice-relay study; not on Tinker."
resource: src/marli/models/qwen3_8_27b.yaml
tags: [model, qwen, thinking, local-backend, vllm, lora]
timestamp: 2026-10-01
---

# Qwen3.8-27B

Registry entry: `src/marli/models/qwen3_8_27b.yaml` (name `qwen3_8_27b`).

| Field | Value |
|---|---|
| HF id | `Qwen/Qwen3.8-27B` |
| Family / renderer / tool format | `qwen3_5` / `qwen3_8_medium` / `qwen3_5_xml` |
| Reasoning effort | set by the renderer: `qwen3_8_medium` (the registry default, used here), `qwen3_8_low`, `qwen3_8_nothink` (thinking off); plain `qwen3_8` uses `xhigh` (`src/marli/render/registry.py`) |
| Thinking | yes |
| Architecture | `Qwen3_5ForConditionalGeneration` |
| Context | native 262,144; served capped at 32,768 (registry `max_ctx`) |
| Default max tokens | 8,192 |
| Tinker | `tinker_id: null` (availability unverified) |
| LoRA targets | attention and MLP projections plus `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; an export key map renames layers for vLLM |
| Draft head | its own 1-layer MTP head, usable for [speculative decoding](../concepts/speculative-decoding-mtp.md) |

## Serving and training on 2×H200 [partial]

Source: [benchmark](../../sources/sacrifice-relay-throughput-bench.md). All
numbers are bf16, vLLM, single-agent code rollouts, T=1, each measured once.

- **Decode speed at 16 agents:** 48 tok/s per agent (TP1), 85 with MTP, and
  132 with TP2 + MTP.
- **Decode speed at 128 agents:** 27 tok/s per agent, with or without MTP.
- **KV capacity at 0.9 utilization (TP1):** 954k tokens, or 771k with MTP.
- **vLLM startup:** 320 s cold, 120–275 s warm.
- **Learner:**
  - the resident base copy is ≈52 GiB;
  - a 24k-context relay train step peaked at 94.8 GiB on one GPU, at ≈1,000
    tok/s;
  - data-parallel over 2 GPUs with vLLM asleep, it reached ≈3,000 tok/s (see
    [GPU time-sharing](../concepts/gpu-time-sharing.md)).
- **On-policy checks (LoRA r=32, with MTP):** over 30 RL steps,
  `kl_sample_train` stayed at or below 7.5e-4 and the mean IS ratio at
  1.0000 ± 0.0001. Adapter effect drift was 0.016–0.028 nats
  ([on-policy check](../concepts/on-policy-check.md)).

## Observed behaviour (untrained, effort `medium`) [partial]

- **DeepCoder train difficulty filter** (single agent, 12,288 tokens per
  episode, 6,144 per call, 4 attempts per problem, 800 problems):
  - 385 problems were solved 4/4, 207 in 1–3 of 4 (kept), and 208 in 0/4;
  - 22% of episodes ran out of budget, mostly through turns cut off at the
    per-call limit ([token budget binding](../concepts/token-budget-binding.md));
  - see [DeepCoder](deepcoder.md).
- **code_rules relay** (16 untrained relays, unfiltered problems): 37.5% of
  contributors never ran CI. About 8% (3/38) of those with a real choice
  revealed the rule. [pilot]

## Under RL (sacrifice relay, team reward, 30 steps, one seed) [partial]

- Team score rose +0.086 (z=2.3), mostly through reaching CI more often.
- The sacrifice rate fell from 10.6% to 7.2% (z=−1.7).
- See [the synthesis](../syntheses/does-rl-teach-sacrifice.md).

## Tensions

- The registry still says `local: unverified`, and its notes say local LoRA
  support "awaits the GPU spike". The sacrifice-relay benchmark and trial have
  since trained LoRA r=32 locally for 30+ steps with all checks passing. The
  registry entry is stale.

Sources: [experiment 1](../../sources/sacrifice-relay-experiment-1.md),
[benchmark](../../sources/sacrifice-relay-throughput-bench.md).
