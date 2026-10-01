---
type: entity
title: DeepCoder (training split)
description: "agentica-org/DeepCoder-Preview-Dataset (primeintellect, taco and lcbv5 train subsets; MIT; contains LiveCodeBench-derived problems, so never commit text), graded by hidden tests capped at 32; the sacrifice relay kept 207 of 800 problems that untrained Qwen3.8-27B solved in 1–3 of 4 attempts."
resource: src/marli/tasks/deepcoder.yaml
tags: [dataset, coding, training-data, rl, contamination-sensitive]
timestamp: 2026-10-01
---

# DeepCoder (training split)

Registry entry: `src/marli/tasks/deepcoder.yaml` (name `deepcoder`). It was
added in M5 ([PR #7](https://github.com/SidBaines/marl-investigations/pull/7))
with the `code_fn` environment.

| Field | Value |
|---|---|
| HF dataset | `agentica-org/DeepCoder-Preview-Dataset` |
| Subsets used | `primeintellect`, `taco`, `lcbv5`, train splits only |
| Kind | function-level coding; graded by tests (`test_format: deepcoder`) |
| Hidden tests | a seeded sample of at most 32 per problem; public examples untouched |
| Dedupe | first parseable row per normalized problem text |
| License | MIT |
| `commit_text` | false. Never commit problem text |

- **Contamination.** The `lcbv5` subset comes from LiveCodeBench. The
  evaluation set is `lcb_v6`, and the registry notes say to decontaminate this
  source against a built `lcb_v6` TaskSet (`data build source=deepcoder
  exclude=<lcb_v6 TaskSet manifest>`). The Codeforces and `lcbv5` test splits
  are evaluation splits, as in tinker-cookbook's code RL recipe. Never paste
  problem text, examples or transcripts into the wiki.
- **Graded by** the `code_fn` environment: base tests pass or fail, with
  `stop_on_first_failure: true` and `max_grade_s: 60` in the study. The
  [code_rules](env-code-rules.md) environment adds its house rule on top.

## Difficulty filter for the sacrifice relay [partial]

Source: [experiment 1](../../sources/sacrifice-relay-experiment-1.md) (as run
2026-09-25).

- **Candidates:** 800 problems (`marli data build source=deepcoder max_n=800
  shuffle=true seed=0`).
- **Filter:** untrained [Qwen3.8-27B](qwen3-8-27b.md) (effort `medium`), single
  agent, 12,288 tokens per episode and 6,144 per call, 4 attempts per problem.
  That is 3,200 episodes, with 0 failed. The run was paused after 100 problems
  and resumed without changing its identity.
- **Pass@4 distribution** (how many of the 4 attempts passed all base tests):

| Passes of 4 | Problems | Fate |
|---|---|---|
| 4 | 385 | dropped, too easy |
| 1–3 | 207 (26%) | **kept** |
| 0 | 208 | dropped, too hard |

- The first 100 problems projected about 320 kept (40 in band). The full run
  kept 207.
- The kept pool makes 51 relay repos of 4 problems (3 left over) and 207 solo
  repos. Over 80 steps each would be drawn about 6 times, and over the 30
  steps actually run each relay repo was drawn about 2.4 times. Only the house
  rule is resampled.
- 22% of filter episodes ran out of budget (702 of 3,200). See
  [token budget binding](../concepts/token-budget-binding.md).

The source states the band but not the reason for it. One effect: problems
the model always or never solves give every playthrough the same base result,
which leaves little reward variation to learn from (compare
[zero-variance groups](../concepts/zero-variance-groups.md)).

## Tensions

- The registry asks for decontamination against `lcb_v6`. The study's build
  command (`run.sh`) does not pass `exclude=`. That does not matter for this
  training-only study, which has no held-out set, but this pool should not be
  reused for training before an `lcb_v6` evaluation without the exclude step.
  [open]
