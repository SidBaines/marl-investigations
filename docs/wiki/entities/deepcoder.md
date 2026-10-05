---
type: entity
title: DeepCoder (training split)
description: "agentica-org/DeepCoder-Preview-Dataset (primeintellect, taco and lcbv5 train subsets; MIT; contains LiveCodeBench-derived problems, so never commit text), graded by hidden tests capped at 32; it holds reworded copies of the same problem (about 4 prompts per upstream problem in primeintellect) that exact-text dedupe misses; the sacrifice relay trained on 207 filtered problems and built its held-out set from deepcoder_no_lcb with a seven-check independence test (1,600 drawn, 295 dropped)."
resource: src/marli/tasks/deepcoder.yaml
tags: [dataset, coding, training-data, rl, contamination-sensitive, held-out, near-duplicates]
timestamp: 2026-10-05
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

## Reworded copies inside DeepCoder (found 2026-10-05) [partial]

DeepCoder contains the same problem several times in different words. The
registry's dedupe (first row per normalised problem text) catches only exact
repeats.

- **Why:** PrimeIntellect's upstream data usually holds about 4 differently
  worded prompts per source problem.
- **In the sacrifice relay's pools:**
  - experiment 1's 800 candidates held 21 copies of each other;
  - the 207 trained problems held 2 (no training repo contains two copies of
    one problem);
  - of the first 800 problems freshly drawn for a held-out set, 51 were
    reworded copies of experiment 1's candidates, 11 of them of trained
    problems.
- **Consequence:** a "new" problem drawn from DeepCoder by id or exact text
  can be one the model trained on. Any held-out set needs a near-copy check.

Source: [study report §4.3](../../sources/sacrifice-relay-experiments-1-3-and-evals.md);
`experiments/2026-09-25_sacrifice-relay/exp2_eval/README.md`.

## The held-out set for the transfer eval (2026-10-05) [partial]

Built on the dev box (CPU), checks in `src/marli/data/overlap.py` (bacbef2,
2cc9a87).

1. **Draw.** 1,600 problems from `deepcoder_no_lcb`, which is DeepCoder train
   without its LiveCodeBench subset (`src/marli/tasks/deepcoder_no_lcb.yaml`,
   20b5dca).
2. **Check.** Compare each problem, in draw order, with all 800 of experiment
   1's candidates (stricter than the 207 trained) and with every earlier
   drawn problem. Drop it if any check fires:
   - too little text to compare (under 20 words once boilerplate is removed,
     or mostly non-Latin script);
   - text overlap on rare 5-word runs, with boilerplate lines removed
     (Jaccard ≥ 0.05 or containment ≥ 0.1);
   - whole-prompt overlap, wrapper included (Jaccard ≥ 0.3);
   - shared test cases (a whole suite contained in another, or rare
     input/output pairs);
   - the same upstream source id;
   - two sentence-embedding models: bge-small-en-v1.5 at cosine ≥ 0.88 and
     all-MiniLM-L6-v2 at ≥ 0.75.
3. **Result.** 295 of 1,600 were dropped: 179 against used problems, and 116
   only as copies of an earlier drawn problem. Of the 1,305 that passed, the
   first 900 became the candidates.
   - A re-check of the 900 flagged none.
   - Both embedding models caught all 51 known copies, and the text check did
     too (lowest Jaccard 0.558).
4. **Difficulty filter.** Experiment 1's filter exactly: the untrained 27B, 4
   attempts, keep problems solved in 1–3 of 4.
   - It covered only the first 600 candidates (capped with
     `stop_after_tasks`; Sid decided 25 repos were enough).
   - It kept 156 (26%, the same rate as experiment 1); 53% were too easy and
     21% too hard.
   - Downstream filtering of a capped run needed the `allow_paused` fix
     (73c023b, ce65a2a).
5. **Repos.** 25 repos of 4, built in both the training rule forms and the
   held-out forms.

**What the checks cannot rule out** (from the source): copies in another
language; the same problem with new tests and a wholly new statement beyond
the embedded prefix; text inside images; two different problems built on one
idea; and rows without a source id.

Thresholds were calibrated by reading flagged pairs privately on the dev box.
Only counts and scores are committed (`exp2_eval/results/independence_*.txt`).

## Tensions

- The registry asks for decontamination against `lcb_v6`. The study's build
  command (`run.sh`) does not pass `exclude=`. ~~That does not matter for this
  training-only study, which has no held-out set~~ The study now has a held-out
  set, but it avoids the issue by drawing from `deepcoder_no_lcb`. The training
  pool itself still contains `lcbv5` problems. It should not be reused for
  training before an `lcb_v6` evaluation without the exclude step. [open]
- **Exact-text dedupe is not enough** (see "Reworded copies" above). Pools
  built with the registry's dedupe alone can contain one problem several
  times. [partial]
