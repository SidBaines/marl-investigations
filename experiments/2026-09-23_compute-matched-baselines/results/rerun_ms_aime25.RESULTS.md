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
