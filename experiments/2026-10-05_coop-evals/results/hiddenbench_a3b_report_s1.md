# hiddenbench

HiddenBench hidden-profile group decisions: 4 agents (one model per seat), shared facts plus one private fact each, an initial vote, 15 discussion rounds and a final vote. The correct option is only identifiable when private facts are pooled; the Full Profile control gives every agent every fact. Upstream harness at a pinned commit, run by the study's run.sh with the marli patch (one model per seat; extra request fields; a failed scenario is recorded instead of aborting the run).

Source: {'repo': 'https://github.com/Yassellee/HiddenBench_ICML', 'commit': '3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1'}. Citation: Yuxuan Li, Aoi Naito and Hirokazu Shirado. Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs. ICML 2026. arXiv:2505.11556.

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | hidden | post_average | 58 | 0 | 7 | 0.185 [0.116, 0.259] | — | — |
| base | hidden | pre_average | 58 | 0 | 7 | 0.134 [0.086, 0.190] | — | — |
| base | hidden | post_majority | 58 | 0 | 7 | 0.086 [0.037, 0.186] | — | — |
| base | hidden | pre_majority | 58 | 0 | 7 | 0.017 [0.003, 0.091] | — | — |
| base | hidden/focal | post_correct | 232 | 0 | 0 | 0.185 [0.116, 0.259] | — | — |
| base | hidden/focal | pre_correct | 232 | 0 | 0 | 0.134 [0.086, 0.190] | — | — |
| base | hidden/others | post_correct | 232 | 0 | 0 | 0.185 [0.116, 0.259] | — | — |
| base | hidden/others | pre_correct | 232 | 0 | 0 | 0.134 [0.086, 0.190] | — | — |
| team_s79 | hidden | post_average | 58 | 0 | 7 | 0.241 [0.159, 0.332] | +0.056 [-0.004, +0.116] | 0.104 (sign_flip) |
| team_s79 | hidden | pre_average | 58 | 0 | 7 | 0.129 [0.086, 0.177] | -0.004 [-0.043, +0.034] | 1 (sign_flip) |
| team_s79 | hidden | post_majority | 58 | 0 | 7 | 0.155 [0.084, 0.269] | +0.069 [+0.000, +0.155] | 0.219 (mcnemar) |
| team_s79 | hidden | pre_majority | 58 | 0 | 7 | 0.017 [0.003, 0.091] | +0.000 [+0.000, +0.000] | 1 (mcnemar) |
| team_s79 | hidden/focal | post_correct | 232 | 0 | 0 | 0.241 [0.159, 0.332] | +0.056 [-0.004, +0.116] | 0.104 (sign_flip) |
| team_s79 | hidden/focal | pre_correct | 232 | 0 | 0 | 0.129 [0.086, 0.177] | -0.004 [-0.043, +0.034] | 1 (sign_flip) |
| team_s79 | hidden/others | post_correct | 232 | 0 | 0 | 0.241 [0.159, 0.332] | +0.056 [-0.004, +0.116] | 0.104 (sign_flip) |
| team_s79 | hidden/others | pre_correct | 232 | 0 | 0 | 0.129 [0.086, 0.177] | -0.004 [-0.043, +0.034] | 1 (sign_flip) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).

Notes:
- base: 7 of 529 rows are harness/model errors (excluded)
- team_s79: 7 of 529 rows are harness/model errors (excluded)
