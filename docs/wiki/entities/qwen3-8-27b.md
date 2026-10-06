---
type: entity
title: Qwen3.8-27B
description: "Qwen/Qwen3.8-27B, a 27B thinking model (Qwen3.5 architecture, renderer qwen3_8_medium in the sacrifice relay; its chat template defaults to xhigh effort) with its own MTP draft head; LoRA r=32 trained on the local backend on 2×H200 for up to 60 steps; team training taught its first contributor to review (17% → 80%) and transferred to new problems and a changed format, but not to standard cooperation evals; not on Tinker."
resource: src/marli/models/qwen3_8_27b.yaml
tags: [model, qwen, thinking, local-backend, vllm, lora]
timestamp: 2026-10-05
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

## Under RL (sacrifice relay, one seed per run) [partial]

- **Experiment 1** (0/1/3 scoring, team reward, 30 steps):
  - team score rose +0.086 (z=2.3), mostly through reaching CI more often;
  - the sacrifice rate fell from 10.6% to 7.2% (z=−1.7).
- **Experiment 2** (0/1 scoring, no position in the prompt, team reward, 60
  steps):
  - contributor 1's review rate rose from 17% to 80%;
  - team score rose from 0.089 to 0.267, still rising at step 59;
  - redundant reviews fell to 10%, then rose to 24% by steps 50–59.
- **Individual reward** (30 steps): flat. **Experiment 2.1** (frozen trained
  contributor 1, individual reward for the rest): followers learned to apply
  the rule (50% → 62%), not to find it.
- Sampler/trainer agreement held throughout (`kl_sample_train` ≤ 7.6e-4),
  including after resuming on a new pod.
- See [the synthesis](../syntheses/does-rl-teach-sacrifice.md).

## Evals of the trained policy (2026-10-05) [partial]

Experiment 2's team policy after step 59, against the untrained model.

- **Transfer in the relay.**
  - Held-out problems: team score 0.085 → 0.300; contributor 1 reviews 93%.
  - Fully changed surface: 0.115 → 0.275, almost all of the gain kept.
  - No format-matching habits.
  - Its followers did not get better at applying the rule (57% → 51%, n.s.).
  - See [behaviour vs format](../concepts/behaviour-transfers-across-format.md).
- **Standard cooperation evals:** no change in one-shot giving games, the
  volunteer's dilemma, HiddenBench (19.1% vs 19.2%) or a planted help request
  ([standard cooperation evals](standard-coop-evals.md)). Untrained, it is
  more generous than the A3B (dictator share 0.31; prisoner's-dilemma
  cooperation 23%).
- **Helping another agent.**
  - Inside the relay, it appended a requested line for 19–20% of
    contributors (36–38% of first contributors), the same before and after
    training.
  - In a stock Inspect agent it never did (0/120). There it read the note
    60–90% of the time and usually told the user about it
    ([harness-dependent helpfulness](../concepts/harness-dependent-helpfulness.md)).

## Serving notes (2026-10-05)

- **Chat template effort.** The chat template defaults to reasoning effort
  `xhigh`, and adds an instruction saying so to the system prompt. Training
  rendered `medium`, so chat-endpoint evals must send
  `chat_template_kwargs: {reasoning_effort: medium}`. With that set, the
  template's prompt token ids equal the training renderer's on the game
  prompts.
- **Earlier thinking.** In multi-turn chat, the template keeps earlier
  thinking, as the training buffers did.

## Tensions

- ~~The registry still says `local: unverified`, and its notes say local LoRA
  support "awaits the GPU spike". The sacrifice-relay benchmark and trial have
  since trained LoRA r=32 locally for 30+ steps with all checks passing. The
  registry entry is stale.~~ Fixed in fe745b2: the registry now says `local: "yes"`.

- **Not compared like for like with the A3B.** The A3B ran twice the batch,
  with an adapter that leaves most of its weights frozen, so "the 27B learned
  faster per game" (0.267 after about 960 games against about 2,560) is not
  a model comparison ([Qwen3.6-35B-A3B](qwen3-6-35b-a3b.md)). [open]

Sources: [experiment 1](../../sources/sacrifice-relay-experiment-1.md),
[benchmark](../../sources/sacrifice-relay-throughput-bench.md),
[study report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md).
