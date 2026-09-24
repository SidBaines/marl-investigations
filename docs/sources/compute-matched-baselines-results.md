---
type: source
title: Compute-matched baselines pilot — eval-report tables (aggregates only)
description: The three `marli eval report` tables behind the compute-matched baselines pilot (AIME25 pilot, HMMT25 pilot, AIME25 multi-session rerun) — accuracy with Wilson CIs, oracle, answered rate, compute percentiles, paired lift with bootstrap CI and McNemar p, and the >10% compute-match flag.
resource: https://github.com/SidBaines/marl-investigations/tree/m3-harness/experiments/2026-09-23_compute-matched-baselines/results
source_date: 2026-09-24
status: pilot
provenance: "concatenation of three files, each copied verbatim on 2026-09-24 from branch m3-harness (PR https://github.com/SidBaines/marl-investigations/pull/4, open): experiments/2026-09-23_compute-matched-baselines/results/pilot_aime25.RESULTS.md @ 7e3d604, results/pilot_hmmt25.RESULTS.md @ 7e3d604, results/rerun_ms_aime25.RESULTS.md @ 4317902. Only the `<!-- file: ... -->` separators are added. The 3-task smoke table (results/smoke.RESULTS.md @ 7e3d604) and the machine-readable results/*.results.jsonl are not copied. Aggregate numbers only; no benchmark text."
tags: [experiment, evaluation, results-table, compute-matching, aime-2025, hmmt-feb-2025, qwen3.5-4b, tinker]
timestamp: 2026-09-24
---

<!-- file: experiments/2026-09-23_compute-matched-baselines/results/pilot_aime25.RESULTS.md @ 7e3d604 -->

# Evaluation results

| Cell | Tasks | Episodes | Accuracy [95% CI] | avg@k | maj@k | Oracle | Answered | Failed | Accuracy ok | Cost USD | Total gen mean/p50/p90 | CP tokens mean/p50/p90 | Calls mean/p50/p90 | Peak ctx mean/p50/p90 | Total uncached mean/p50/p90 | Paired lift [95% CI] | Paired n | Paired test | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coordinator | 30 | 30 | 0.567 [0.392, 0.726] | — | — | 0.567 | 1.000 | 0 | 0.567 | 0.671218 | 15804.2/15886.0/20621.8 | 13844.6/15885.5/19410.6 | 4.3/2.0/9.2 | 13793.1/16558.5/17812.3 | 2039.9/989.0/4742.4 | +0.033 [-0.067, +0.133] | 30 | mcnemar | 1 |
| debate3 | 30 | 30 | 0.133 [0.053, 0.297] | — | — | 0.200 | 0.233 | 0 | 0.133 | 1.109955 | 30541.5/31230.0/31230.0 | 10388.3/10410.0/10410.0 | 6.0/6.0/6.0 | 10785.3/10699.5/11225.0 | 1153.8/856.5/2150.4 | -0.400 [-0.567, -0.233] | 30 | mcnemar | 0.0004883 |
| ms3_compaction | 30 | 30 | 0.300 [0.167, 0.479] | — | — | 0.300 | 0.533 | 0 | 0.300 | 0.631803 | 16572.6/19242.0/28160.0 | 16572.6/19242.0/28160.0 | 3.2/4.0/5.0 | 8437.2/10425.0/11621.6 | 1976.3/2102.0/3706.9 | -0.233 [-0.400, -0.067] | 30 | mcnemar | 0.03906 |
| ms3_notes | 30 | 30 | 0.467 [0.302, 0.639] | — | — | 0.467 | 0.900 | 0 | 0.467 | 0.719197 | 19026.4/18251.0/32270.0 | 19026.4/18251.0/32270.0 | 4.0/4.0/6.1 | 9836.9/11505.5/12713.1 | 1826.4/1057.5/3561.7 | -0.067 [-0.233, +0.100] | 30 | mcnemar | 0.7266 |
| sc4 | 30 | 30 | 0.367 [0.219, 0.545] | — | — | 0.367 | 1.000 | 0 | 0.367 | 1.278620 | 30603.4/30776.0/30782.1 | 7699.3/7695.0/7704.3 | 8.0/8.0/8.0 | 8322.8/8234.0/8649.4 | 1718.8/1374.0/3021.2 | -0.167 [-0.333, -0.033] | 30 | mcnemar | 0.125 |
| single | 30 | 30 | 0.533 [0.361, 0.698] | — | — | 0.533 | 0.967 | 0 | 0.533 | 1.127959 | 25486.4/31234.0/31801.3 | 25486.4/31234.0/31801.3 | 2.6/3.0/3.0 | 26051.0/32270.0/32273.0 | 564.6/476.0/907.8 | +0.000 [+0.000, +0.000] | 30 | mcnemar | 1 |
| swarm4 | 30 | 30 | 0.333 [0.192, 0.512] | — | — | 0.533 | 1.000 | 0 | 0.333 | 1.199797 | 27176.4/30773.5/30779.3 | 8801.6/7702.0/13074.5 | 8.8/8.0/10.1 | 8036.8/8516.0/8962.0 | 2139.2/1829.5/3469.6 | -0.200 [-0.367, -0.067] | 30 | mcnemar | 0.03125 |

Comparisons are within-harness and should be read against compute: compare both total generated tokens and critical-path tokens, plus calls and context. Token counts from different tokenizers are not comparable; API token usage is separate in results.jsonl.

Accuracy uses Wilson intervals for G=1; for G>1 it averages per-task means with 2,000 seeded task bootstrap resamples for its interval. avg@k and maj@k average tasks equally; k is each task's recorded episode count. maj@k votes over verifier-equivalent answers; ties use the earliest episode.

Paired lift averages per-task accuracy differences on shared tasks with 2,000 seeded task bootstrap resamples. G=1 uses exact McNemar; G>1 uses a sign-flip permutation test on per-task differences (exact for up to 20 nonzero tasks, otherwise 10,000 seeded draws).

## Accuracy vs compute

| Cell | Accuracy [95% CI] | CI method | Mean total_gen | Mean cp_tokens |
| --- | --- | --- | --- | --- |
| coordinator | 0.567 [0.392, 0.726] | wilson | 15804.167 | 13844.600 |
| debate3 | 0.133 [0.053, 0.297] | wilson | 30541.533 | 10388.300 |
| ms3_compaction | 0.300 [0.167, 0.479] | wilson | 16572.600 | 16572.600 |
| ms3_notes | 0.467 [0.302, 0.639] | wilson | 19026.433 | 19026.433 |
| sc4 | 0.367 [0.219, 0.545] | wilson | 30603.400 | 7699.267 |
| single | 0.533 [0.361, 0.698] | wilson | 25486.433 | 25486.433 |
| swarm4 | 0.333 [0.192, 0.512] | wilson | 27176.367 | 8801.600 |

Not compute-matched (>10% mean total_gen difference from baseline): coordinator, debate3, ms3_compaction, ms3_notes, sc4.

Baseline: single.

<!-- file: experiments/2026-09-23_compute-matched-baselines/results/pilot_hmmt25.RESULTS.md @ 7e3d604 -->

# Evaluation results

| Cell | Tasks | Episodes | Accuracy [95% CI] | avg@k | maj@k | Oracle | Answered | Failed | Accuracy ok | Cost USD | Total gen mean/p50/p90 | CP tokens mean/p50/p90 | Calls mean/p50/p90 | Peak ctx mean/p50/p90 | Total uncached mean/p50/p90 | Paired lift [95% CI] | Paired n | Paired test | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coordinator | 30 | 30 | 0.333 [0.192, 0.512] | — | — | 0.333 | 1.000 | 0 | 0.333 | 0.776449 | 17683.2/15898.5/25430.4 | 15203.3/15893.0/19982.1 | 6.1/6.5/9.1 | 14544.5/16589.5/18483.3 | 2475.4/2439.5/4053.1 | -0.200 [-0.400, +0.000] | 30 | mcnemar | 0.1094 |
| debate3 | 30 | 30 | 0.033 [0.006, 0.167] | — | — | 0.067 | 0.067 | 0 | 0.033 | 1.308380 | 31188.6/31230.0/31230.0 | 10410.0/10410.0/10410.0 | 6.0/6.0/6.0 | 10697.7/10678.5/10723.4 | 851.1/793.5/928.2 | -0.500 [-0.667, -0.333] | 30 | mcnemar | 6.104e-05 |
| ms3_compaction | 30 | 30 | 0.333 [0.192, 0.512] | — | — | 0.333 | 0.700 | 0 | 0.333 | 0.558640 | 14555.6/12401.0/28160.0 | 14555.6/12401.0/28160.0 | 3.1/3.0/5.0 | 8250.3/10119.0/11444.7 | 1660.1/1673.0/3171.1 | -0.200 [-0.400, +0.000] | 30 | mcnemar | 0.1094 |
| ms3_notes | 30 | 30 | 0.333 [0.192, 0.512] | — | — | 0.333 | 1.000 | 0 | 0.333 | 0.775281 | 20306.8/24469.0/32270.0 | 20306.8/24469.0/32270.0 | 4.6/4.5/7.0 | 9780.2/11499.5/12248.8 | 1426.6/1058.0/2796.2 | -0.200 [-0.400, +0.000] | 30 | mcnemar | 0.1094 |
| sc4 | 30 | 30 | 0.200 [0.095, 0.373] | — | — | 0.367 | 1.000 | 0 | 0.200 | 1.277479 | 30782.4/30779.5/30794.2 | 7698.9/7696.0/7705.3 | 8.0/8.0/8.0 | 8241.6/8220.5/8265.2 | 1384.8/1308.0/1487.6 | -0.333 [-0.500, -0.167] | 30 | mcnemar | 0.001953 |
| single | 30 | 30 | 0.533 [0.361, 0.698] | — | — | 0.533 | 1.000 | 0 | 0.533 | 1.271610 | 28981.6/31772.0/31821.1 | 28981.6/31772.0/31821.1 | 2.7/3.0/3.0 | 29467.0/32271.0/32276.3 | 485.5/467.0/524.4 | +0.000 [+0.000, +0.000] | 30 | mcnemar | 1 |
| swarm4 | 30 | 30 | 0.233 [0.118, 0.409] | — | — | 0.333 | 1.000 | 0 | 0.233 | 1.293919 | 29178.3/30772.0/30786.9 | 9793.2/8738.0/13917.4 | 9.3/8.0/11.1 | 8486.6/8521.0/8639.0 | 1837.3/1756.5/2012.1 | -0.300 [-0.467, -0.133] | 30 | mcnemar | 0.003906 |

Comparisons are within-harness and should be read against compute: compare both total generated tokens and critical-path tokens, plus calls and context. Token counts from different tokenizers are not comparable; API token usage is separate in results.jsonl.

Accuracy uses Wilson intervals for G=1; for G>1 it averages per-task means with 2,000 seeded task bootstrap resamples for its interval. avg@k and maj@k average tasks equally; k is each task's recorded episode count. maj@k votes over verifier-equivalent answers; ties use the earliest episode.

Paired lift averages per-task accuracy differences on shared tasks with 2,000 seeded task bootstrap resamples. G=1 uses exact McNemar; G>1 uses a sign-flip permutation test on per-task differences (exact for up to 20 nonzero tasks, otherwise 10,000 seeded draws).

## Accuracy vs compute

| Cell | Accuracy [95% CI] | CI method | Mean total_gen | Mean cp_tokens |
| --- | --- | --- | --- | --- |
| coordinator | 0.333 [0.192, 0.512] | wilson | 17683.233 | 15203.300 |
| debate3 | 0.033 [0.006, 0.167] | wilson | 31188.567 | 10410.000 |
| ms3_compaction | 0.333 [0.192, 0.512] | wilson | 14555.600 | 14555.600 |
| ms3_notes | 0.333 [0.192, 0.512] | wilson | 20306.800 | 20306.800 |
| sc4 | 0.200 [0.095, 0.373] | wilson | 30782.367 | 7698.900 |
| single | 0.533 [0.361, 0.698] | wilson | 28981.567 | 28981.567 |
| swarm4 | 0.233 [0.118, 0.409] | wilson | 29178.333 | 9793.233 |

Not compute-matched (>10% mean total_gen difference from baseline): coordinator, ms3_compaction, ms3_notes.

Baseline: single.

<!-- file: experiments/2026-09-23_compute-matched-baselines/results/rerun_ms_aime25.RESULTS.md @ 4317902 -->

# Evaluation results

| Cell | Tasks | Episodes | Accuracy [95% CI] | avg@k | maj@k | Oracle | Answered | Failed | Accuracy ok | Cost USD | Total gen mean/p50/p90 | CP tokens mean/p50/p90 | Calls mean/p50/p90 | Peak ctx mean/p50/p90 | Total uncached mean/p50/p90 | Paired lift [95% CI] | Paired n | Paired test | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms3_compaction | 30 | 30 | 0.400 [0.246, 0.577] | — | — | 0.400 | 1.000 | 0 | 0.400 | 1.167303 | 13526.2/11399.5/27295.0 | 13526.2/11399.5/27295.0 | 3.2/3.0/6.0 | 7469.9/10284.0/11472.2 | 1789.4/1773.0/3404.8 | -0.133 [-0.300, +0.033] | 30 | mcnemar | 0.2891 |
| ms3_notes | 30 | 30 | 0.500 [0.332, 0.668] | — | — | 0.500 | 1.000 | 0 | 0.500 | 0.956975 | 15299.6/8924.5/32270.0 | 15299.6/8924.5/32270.0 | 3.3/3.0/5.1 | 8874.7/9954.0/11988.8 | 1186.6/889.0/2082.8 | -0.033 [-0.167, +0.067] | 30 | mcnemar | 1 |
| single | 30 | 30 | 0.533 [0.361, 0.698] | — | — | 0.533 | 0.967 | 0 | 0.533 | 1.127959 | 25486.4/31234.0/31801.3 | 25486.4/31234.0/31801.3 | 2.6/3.0/3.0 | 26051.0/32270.0/32273.0 | 564.6/476.0/907.8 | +0.000 [+0.000, +0.000] | 30 | mcnemar | 1 |

Comparisons are within-harness and should be read against compute: compare both total generated tokens and critical-path tokens, plus calls and context. Token counts from different tokenizers are not comparable; API token usage is separate in results.jsonl.

Accuracy uses Wilson intervals for G=1; for G>1 it averages per-task means with 2,000 seeded task bootstrap resamples for its interval. avg@k and maj@k average tasks equally; k is each task's recorded episode count. maj@k votes over verifier-equivalent answers; ties use the earliest episode.

Paired lift averages per-task accuracy differences on shared tasks with 2,000 seeded task bootstrap resamples. G=1 uses exact McNemar; G>1 uses a sign-flip permutation test on per-task differences (exact for up to 20 nonzero tasks, otherwise 10,000 seeded draws).

## Accuracy vs compute

| Cell | Accuracy [95% CI] | CI method | Mean total_gen | Mean cp_tokens |
| --- | --- | --- | --- | --- |
| ms3_compaction | 0.400 [0.246, 0.577] | wilson | 13526.233 | 13526.233 |
| ms3_notes | 0.500 [0.332, 0.668] | wilson | 15299.600 | 15299.600 |
| single | 0.533 [0.361, 0.698] | wilson | 25486.433 | 25486.433 |

Not compute-matched (>10% mean total_gen difference from baseline): ms3_compaction, ms3_notes.

Baseline: single.
