# Evaluation results

| Cell | Tasks | Episodes | Accuracy [95% CI] | avg@k | maj@k | Oracle | Answered | Failed | Accuracy ok | Cost USD | Total gen mean/p50/p90 | CP tokens mean/p50/p90 | Calls mean/p50/p90 | Peak ctx mean/p50/p90 | Total uncached mean/p50/p90 | Paired lift [95% CI] | Paired n | Paired test | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coordinator | 3 | 3 | 1.000 [0.439, 1.000] | — | — | 1.000 | 1.000 | 0 | 1.000 | 0.051053 | 12532.7/14995.0/15707.8 | 10622.7/11398.0/14988.4 | 4.7/6.0/6.0 | 10325.7/9400.0/15591.2 | 2131.7/2133.0/2833.8 | +0.000 [+0.000, +0.000] | 3 | mcnemar | 1 |
| debate3 | 3 | 3 | 0.667 [0.208, 0.939] | — | — | 0.667 | 0.667 | 0 | 0.667 | 0.105271 | 28522.0/28802.0/30744.4 | 10410.0/10410.0/10410.0 | 6.0/6.0/6.0 | 10856.3/10700.0/11141.6 | 1327.0/858.0/2182.8 | -0.333 [-1.000, +0.000] | 3 | mcnemar | 1 |
| ms3_compaction | 3 | 3 | 0.667 [0.208, 0.939] | — | — | 0.667 | 0.667 | 0 | 0.667 | 0.044399 | 11740.0/4592.0/22888.0 | 11740.0/4592.0/22888.0 | 2.3/1.0/4.2 | 6715.0/5082.0/10075.6 | 1746.3/573.0/3455.4 | -0.333 [-1.000, +0.000] | 3 | mcnemar | 1 |
| ms3_notes | 3 | 3 | 0.667 [0.208, 0.939] | — | — | 0.667 | 0.667 | 0 | 0.667 | 0.042980 | 12549.0/4055.0/26615.8 | 12549.0/4055.0/26615.8 | 2.7/3.0/3.8 | 6298.0/4844.0/10616.8 | 1397.0/789.0/2356.2 | -0.333 [-1.000, +0.000] | 3 | mcnemar | 1 |
| sc4 | 3 | 3 | 0.667 [0.208, 0.939] | — | — | 0.667 | 1.000 | 0 | 0.667 | 0.128979 | 30774.0/30772.0/30776.8 | 7693.7/7693.0/7694.6 | 8.0/8.0/8.0 | 8395.0/8238.0/8681.2 | 2019.3/1394.0/3160.4 | -0.333 [-1.000, +0.000] | 3 | mcnemar | 1 |
| single | 3 | 3 | 1.000 [0.439, 1.000] | — | — | 1.000 | 1.000 | 0 | 1.000 | 0.098276 | 21370.3/18950.0/28765.2 | 21370.3/18950.0/28765.2 | 2.7/3.0/3.0 | 22007.0/19432.0/29704.0 | 636.7/482.0/938.8 | +0.000 [+0.000, +0.000] | 3 | mcnemar | 1 |
| swarm4 | 3 | 3 | 1.000 [0.439, 1.000] | — | — | 1.000 | 1.000 | 0 | 1.000 | 0.071459 | 16404.0/15837.0/22349.8 | 6278.3/7173.0/7589.8 | 7.7/8.0/8.0 | 6881.0/6819.0/8594.2 | 2519.0/1933.0/3456.2 | +0.000 [+0.000, +0.000] | 3 | mcnemar | 1 |

Comparisons are within-harness and should be read against compute: compare both total generated tokens and critical-path tokens, plus calls and context. Token counts from different tokenizers are not comparable; API token usage is separate in results.jsonl.

Accuracy uses Wilson intervals for G=1; for G>1 it averages per-task means with 2,000 seeded task bootstrap resamples for its interval. avg@k and maj@k average tasks equally; k is each task's recorded episode count. maj@k votes over verifier-equivalent answers; ties use the earliest episode.

Paired lift averages per-task accuracy differences on shared tasks with 2,000 seeded task bootstrap resamples. G=1 uses exact McNemar; G>1 uses a sign-flip permutation test on per-task differences (exact for up to 20 nonzero tasks, otherwise 10,000 seeded draws).

## Accuracy vs compute

| Cell | Accuracy [95% CI] | CI method | Mean total_gen | Mean cp_tokens |
| --- | --- | --- | --- | --- |
| coordinator | 1.000 [0.439, 1.000] | wilson | 12532.667 | 10622.667 |
| debate3 | 0.667 [0.208, 0.939] | wilson | 28522.000 | 10410.000 |
| ms3_compaction | 0.667 [0.208, 0.939] | wilson | 11740.000 | 11740.000 |
| ms3_notes | 0.667 [0.208, 0.939] | wilson | 12549.000 | 12549.000 |
| sc4 | 0.667 [0.208, 0.939] | wilson | 30774.000 | 7693.667 |
| single | 1.000 [0.439, 1.000] | wilson | 21370.333 | 21370.333 |
| swarm4 | 1.000 [0.439, 1.000] | wilson | 16404.000 | 6278.333 |

Not compute-matched (>10% mean total_gen difference from baseline): coordinator, debate3, ms3_compaction, ms3_notes, sc4, swarm4.

Baseline: single.
