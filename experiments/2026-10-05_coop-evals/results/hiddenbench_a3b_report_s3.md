# hiddenbench

HiddenBench hidden-profile group decisions: 4 agents (one model per seat), shared facts plus one private fact each, an initial vote, 15 discussion rounds and a final vote. The correct option is only identifiable when private facts are pooled; the Full Profile control gives every agent every fact. Upstream harness at a pinned commit, run by the study's run.sh with the marli patch (one model per seat; extra request fields; a failed scenario is recorded instead of aborting the run).

Source: {'repo': 'https://github.com/Yassellee/HiddenBench_ICML', 'commit': '3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1'}. Citation: Yuxuan Li, Aoi Naito and Hirokazu Shirado. Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs. ICML 2026. arXiv:2505.11556.

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | hidden | post_average | 174 | 0 | 21 | 0.178 [0.124, 0.239] | — | — |
| base | hidden | pre_average | 174 | 0 | 21 | 0.125 [0.086, 0.167] | — | — |
| base | hidden | post_majority | 174 | 0 | 21 | 0.080 [0.034, 0.138] | — | — |
| base | hidden | pre_majority | 174 | 0 | 21 | 0.006 [0.000, 0.017] | — | — |
| base | hidden/focal | post_correct | 696 | 0 | 0 | 0.178 [0.124, 0.239] | — | — |
| base | hidden/focal | pre_correct | 696 | 0 | 0 | 0.125 [0.086, 0.167] | — | — |
| base | hidden/others | post_correct | 696 | 0 | 0 | 0.178 [0.124, 0.239] | — | — |
| base | hidden/others | pre_correct | 696 | 0 | 0 | 0.125 [0.086, 0.167] | — | — |
| team_s79 | hidden | post_average | 173 | 0 | 22 | 0.227 [0.162, 0.299] | +0.048 [+0.010, +0.089] | 0.0247 (sign_flip) |
| team_s79 | hidden | pre_average | 173 | 0 | 22 | 0.129 [0.093, 0.167] | +0.004 [-0.018, +0.023] | 0.788 (sign_flip) |
| team_s79 | hidden | post_majority | 173 | 0 | 22 | 0.121 [0.057, 0.201] | +0.040 [-0.006, +0.092] | 0.194 (sign_flip) |
| team_s79 | hidden | pre_majority | 173 | 0 | 22 | 0.006 [0.000, 0.017] | +0.000 [+0.000, +0.000] | 1 (sign_flip) |
| team_s79 | hidden/focal | post_correct | 692 | 0 | 0 | 0.227 [0.162, 0.299] | +0.048 [+0.010, +0.089] | 0.0247 (sign_flip) |
| team_s79 | hidden/focal | pre_correct | 692 | 0 | 0 | 0.129 [0.093, 0.167] | +0.004 [-0.018, +0.023] | 0.788 (sign_flip) |
| team_s79 | hidden/others | post_correct | 692 | 0 | 0 | 0.227 [0.162, 0.299] | +0.048 [+0.010, +0.089] | 0.0247 (sign_flip) |
| team_s79 | hidden/others | pre_correct | 692 | 0 | 0 | 0.129 [0.093, 0.167] | +0.004 [-0.018, +0.023] | 0.788 (sign_flip) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).

Notes:
- base: 21 of 1587 rows are harness/model errors (excluded)
- team_s79: 22 of 1579 rows are harness/model errors (excluded)
