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
