---
type: entity
title: Qwen3.6-35B-A3B
description: "Qwen/Qwen3.6-35B-A3B, a mixture-of-experts thinking model (35B parameters, about 3B active per token; 256 routed experts, top-8, plus a shared expert; renderer qwen3_5, no reasoning-effort switch); LoRA r=32 on attention, linear attention and the shared expert, with routed experts and router frozen, trained on the local backend on 2×H200 for up to 80 steps in the sacrifice relay; fixed-probe hot-load check needs a 0.5-nat tolerance."
resource: src/marli/models/qwen3_6_35b_a3b.yaml
tags: [model, qwen, thinking, moe, local-backend, vllm, lora, tinker]
timestamp: 2026-10-05
---

# Qwen3.6-35B-A3B

Registry entry: `src/marli/models/qwen3_6_35b_a3b.yaml` (name
`qwen3_6_35b_a3b`). It was verified for local training on 2026-10-03
(015bf57, 3b251d9).

| Field | Value |
|---|---|
| HF id | `Qwen/Qwen3.6-35B-A3B` |
| Family / renderer / tool format | `qwen3_5` / `qwen3_5` / `qwen3_5_xml` |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (vision-language mixture of experts) |
| Experts | 256 routed experts (each token uses 8) plus one shared expert that every token uses. The routed experts hold 93% of the weights and about a third of the compute per token |
| Thinking | yes; no reasoning-effort switch (unlike [Qwen3.8-27B](qwen3-8-27b.md)) |
| Context | registry `max_ctx` 32,768; `tinker_max_ctx` 65,536 |
| Default max tokens | 8,192 |
| Tinker | `tinker_id: Qwen/Qwen3.6-35B-A3B`; prices (USD per 1M tokens, as of 2026-09-23): prefill 0.54, sample 1.34, train 1.18 |
| Local (PEFT + vLLM) | `local: "yes"` |
| LoRA targets | the 27B's list: attention, linear attention and MLP projections. On this model the MLP names reach only the shared expert, because the routed experts are fused 3-D parameters, not linear layers. Routed experts and router stay frozen |

## Training it locally [partial]

Sources: [study report §5](../../sources/sacrifice-relay-experiments-1-3-and-evals.md);
`experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/README.md`.

- **Learner:** the fused experts are 3-D parameters (transformers 5.5.4), and
  the learner requires `grouped_mm` (5ea9b20).
- **Serving:** vLLM's `--lora-target-modules` must list the same modules
  (64dcf29). Otherwise vLLM silently ignores weights for modules it did not
  wrap, and wraps the experts with zero adapters.
- **Adapter size:** about 160 MB per step; trainer state about 0.5 GB.
- **Hot-load check:** the fixed-probe check reports about 0.17 nats of drift,
  against about 0.03 for the 27B. It fails at the default 0.05 tolerance. The
  likely cause is near-tie top-8 expert routing on off-policy probe text.
  - Sampled-token agreement stayed fine.
  - Runs set `local_adapter_check_tol=0.5`, and rely on the per-step abort at
    `kl_sample_train > 5e-3`.
  - See [on-policy check](../concepts/on-policy-check.md).
- **Sampler/trainer agreement:** `kl_sample_train` stayed ≤ 2.0e-3 over 80
  steps (experiment 2 team), ≤ 1.9e-3 (2.1) and ≤ 1.8e-3 (experiment 3). The
  mean IS ratio stayed at 1.000. The 27B's values are about 3× lower
  (≤ 7.6e-4).
- **Speed** on 2×H200 with the time-shared layout, vLLM at 40% of each GPU:
  10.1 min per step at 8 repos × 4 (experiment 2), and 11.6 min (experiment
  3). It sampled about 2.4× more relay games per hour than the 27B.

## Observed behaviour [partial]

All in the [code_rules](env-code-rules.md) [relay](protocol-relay.md), one
training seed, unless stated otherwise.

- **Untrained, it reviews a lot and wastefully.** Contributor 1 reviewed 38%
  of its CI runs under experiment 2's prompt (27B: 17%). Contributors who
  already knew the rule reviewed anyway 54% of the time.
- **Team training (0/1 scoring, 80 steps).**
  - Contributor 1's review rate rose from 38% to 77%.
  - Team score rose from 0.049 to 0.268.
  - Redundant reviews fell from 54% to 21%, so it learned *when* to review
    ([synthesis](../syntheses/does-rl-teach-sacrifice.md)).
- **Its followers apply a known rule poorly.**
  - Untrained, 25% on held-out problems under 0/1 scoring.
  - Under 0/1/3 scoring (experiment 3), informed followers scored 3 only 14%
    of the time. So a review lost the team points, and team training
    extinguished reviewing (21% → 3%)
    ([pays in practice](../concepts/sacrifice-pays-in-practice.md)).
- **Transfer.** The team policy keeps its gain on held-out problems (0.060 →
  0.320). Under a fully changed surface it keeps about half (0.050 → 0.140):
  its followers apply new rule kinds much less
  ([behaviour vs format](../concepts/behaviour-transfers-across-format.md)).
- **Standard harnesses**
  ([standard cooperation evals](standard-coop-evals.md)):
  - almost always selfish in one-shot giving games with thinking on;
  - volunteers in about 44% of volunteer's-dilemma decisions;
  - never acts on a planted help request, and mostly does not read it
    ([harness-dependent helpfulness](../concepts/harness-dependent-helpfulness.md));
  - the team-trained version was right more often after HiddenBench
    discussions (17.8% → 22.7%, p = 0.025), the one standard-eval effect.

## Tensions

- **The adapter does not cover the routed experts.** Comparisons with the
  27B mix model size, architecture and how much of the model the adapter can
  change. They also used twice the batch. None of the 27B vs A3B comparisons
  in the study is matched. [open]
- **The loosened hot-load tolerance weakens a guard.** At 0.5 nats, the probe
  check would miss a wrong adapter whose effect is smaller than that. A mix-up
  would then show only in `kl_sample_train`, after a batch was sampled. [open]
