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
