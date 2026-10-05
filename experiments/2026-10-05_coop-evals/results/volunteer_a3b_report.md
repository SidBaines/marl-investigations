# fairgame_volunteer

FAIRGAME's 3-player Volunteer's Dilemma (10 known rounds; one volunteer secures the group at a personal cost: volunteer 5, others 10, nobody volunteers 0 and the game stops) with neutral personalities, one model per seat. FAIRGAME's own runner at a pinned commit via LiteLLM's hosted_vllm provider, run by the study's run.sh with the marli patch (extra request fields; unmatched replies counted).

Source: {'repo': 'https://github.com/aira-list/FAIRGAME', 'commit': 'fc302a642c6f7cc0c439c2ae957a45f5954f4525'}. Citation: Alessio Buscemi et al. FAIRGAME: a Framework for AI Agents Bias Recognition using Game Theory. arXiv:2504.14325. Config: starter_library seed_cfg_volunteer.

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | focal | volunteer_rate | 150 | 0 | 0 | 0.442 [0.408, 0.476] | — | — |
| base | focal | round1_volunteer | 150 | 0 | 0 | 0.880 [0.818, 0.923] | — | — |
| base | group | volunteer_rate | 50 | 0 | 0 | 0.442 [0.415, 0.467] | — | — |
| base | group | round1_volunteer_rate | 50 | 0 | 0 | 0.880 [0.820, 0.933] | — | — |
| base | group | safe_rate | 50 | 0 | 0 | 0.573 [0.527, 0.618] | — | — |
| base | others | volunteer_rate | 150 | 0 | 0 | 0.442 [0.408, 0.476] | — | — |
| base | others | round1_volunteer | 150 | 0 | 0 | 0.880 [0.818, 0.923] | — | — |
| team_s79 | focal | volunteer_rate | 150 | 0 | 0 | 0.449 [0.420, 0.477] | +0.006 [-0.037, +0.053] | 0.775 (permutation) |
| team_s79 | focal | round1_volunteer | 150 | 0 | 0 | 0.893 [0.834, 0.933] | +0.013 [-0.053, +0.087] | 0.853 (permutation) |
| team_s79 | group | volunteer_rate | 50 | 0 | 0 | 0.449 [0.428, 0.468] | +0.006 [-0.026, +0.042] | 0.718 (permutation) |
| team_s79 | group | round1_volunteer_rate | 50 | 0 | 0 | 0.893 [0.847, 0.940] | +0.013 [-0.060, +0.093] | 0.865 (permutation) |
| team_s79 | group | safe_rate | 50 | 0 | 0 | 0.576 [0.543, 0.611] | +0.004 [-0.052, +0.060] | 0.908 (permutation) |
| team_s79 | others | volunteer_rate | 150 | 0 | 0 | 0.449 [0.420, 0.477] | +0.006 [-0.037, +0.053] | 0.775 (permutation) |
| team_s79 | others | round1_volunteer | 150 | 0 | 0 | 0.893 [0.834, 0.933] | +0.013 [-0.053, +0.087] | 0.853 (permutation) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).
