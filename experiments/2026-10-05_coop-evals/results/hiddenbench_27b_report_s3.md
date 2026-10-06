# hiddenbench

HiddenBench hidden-profile group decisions: 4 agents (one model per seat), shared facts plus one private fact each, an initial vote, 15 discussion rounds and a final vote. The correct option is only identifiable when private facts are pooled; the Full Profile control gives every agent every fact. Upstream harness at a pinned commit, run by the study's run.sh with the marli patch (one model per seat; extra request fields; a failed scenario is recorded instead of aborting the run).

Source: {'repo': 'https://github.com/Yassellee/HiddenBench_ICML', 'commit': '3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1'}. Citation: Yuxuan Li, Aoi Naito and Hirokazu Shirado. Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs. ICML 2026. arXiv:2505.11556.

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | hidden | post_average | 166 | 0 | 29 | 0.191 [0.124, 0.257] | — | — |
| base | hidden | pre_average | 166 | 0 | 29 | 0.142 [0.101, 0.184] | — | — |
| base | hidden | post_majority | 166 | 0 | 29 | 0.090 [0.035, 0.152] | — | — |
| base | hidden | pre_majority | 166 | 0 | 29 | 0.006 [0.000, 0.018] | — | — |
| base | hidden/focal | post_correct | 664 | 0 | 0 | 0.191 [0.124, 0.257] | — | — |
| base | hidden/focal | pre_correct | 664 | 0 | 0 | 0.142 [0.101, 0.184] | — | — |
| base | hidden/others | post_correct | 664 | 0 | 0 | 0.191 [0.124, 0.257] | — | — |
| base | hidden/others | pre_correct | 664 | 0 | 0 | 0.142 [0.101, 0.184] | — | — |
| team_s59 | hidden | post_average | 167 | 0 | 28 | 0.192 [0.124, 0.260] | +0.001 [-0.025, +0.031] | 1 (sign_flip) |
| team_s59 | hidden | pre_average | 167 | 0 | 28 | 0.151 [0.108, 0.200] | +0.006 [-0.018, +0.035] | 0.735 (sign_flip) |
| team_s59 | hidden | post_majority | 167 | 0 | 28 | 0.108 [0.046, 0.175] | +0.020 [-0.009, +0.056] | 0.312 (sign_flip) |
| team_s59 | hidden | pre_majority | 167 | 0 | 28 | 0.024 [0.000, 0.057] | +0.018 [-0.012, +0.058] | 0.5 (sign_flip) |
| team_s59 | hidden/focal | post_correct | 668 | 0 | 0 | 0.192 [0.124, 0.260] | +0.001 [-0.025, +0.031] | 1 (sign_flip) |
| team_s59 | hidden/focal | pre_correct | 668 | 0 | 0 | 0.151 [0.108, 0.200] | +0.006 [-0.018, +0.035] | 0.735 (sign_flip) |
| team_s59 | hidden/others | post_correct | 668 | 0 | 0 | 0.192 [0.124, 0.260] | +0.001 [-0.025, +0.031] | 1 (sign_flip) |
| team_s59 | hidden/others | pre_correct | 668 | 0 | 0 | 0.151 [0.108, 0.200] | +0.006 [-0.018, +0.035] | 0.735 (sign_flip) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).

Notes:
- base: 29 of 1523 rows are harness/model errors (excluded)
- team_s59: 28 of 1531 rows are harness/model errors (excluded)
