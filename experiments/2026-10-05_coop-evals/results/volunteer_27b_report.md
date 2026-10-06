# fairgame_volunteer

FAIRGAME's 3-player Volunteer's Dilemma (10 known rounds; one volunteer secures the group at a personal cost: volunteer 5, others 10, nobody volunteers 0 and the game stops) with neutral personalities, one model per seat. FAIRGAME's own runner at a pinned commit via LiteLLM's hosted_vllm provider, run by the study's run.sh with the marli patch (extra request fields; unmatched replies counted).

Source: {'repo': 'https://github.com/aira-list/FAIRGAME', 'commit': 'fc302a642c6f7cc0c439c2ae957a45f5954f4525'}. Citation: Alessio Buscemi et al. FAIRGAME: a Framework for AI Agents Bias Recognition using Game Theory. arXiv:2504.14325. Config: starter_library seed_cfg_volunteer.

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | focal | volunteer_rate | 150 | 0 | 0 | 0.166 [0.124, 0.213] | — | — |
| base | focal | round1_volunteer | 150 | 0 | 0 | 0.300 [0.232, 0.378] | — | — |
| base | group | volunteer_rate | 50 | 0 | 0 | 0.166 [0.128, 0.203] | — | — |
| base | group | round1_volunteer_rate | 50 | 0 | 0 | 0.300 [0.233, 0.367] | — | — |
| base | group | safe_rate | 50 | 0 | 0 | 0.394 [0.307, 0.471] | — | — |
| base | others | volunteer_rate | 150 | 0 | 0 | 0.166 [0.124, 0.213] | — | — |
| base | others | round1_volunteer | 150 | 0 | 0 | 0.300 [0.232, 0.378] | — | — |
| team_s59 | focal | volunteer_rate | 150 | 0 | 0 | 0.138 [0.101, 0.178] | -0.028 [-0.083, +0.029] | 0.342 (permutation) |
| team_s59 | focal | round1_volunteer | 150 | 0 | 0 | 0.253 [0.190, 0.328] | -0.047 [-0.140, +0.053] | 0.434 (permutation) |
| team_s59 | group | volunteer_rate | 50 | 0 | 0 | 0.138 [0.101, 0.177] | -0.028 [-0.078, +0.025] | 0.313 (permutation) |
| team_s59 | group | round1_volunteer_rate | 50 | 0 | 0 | 0.253 [0.187, 0.327] | -0.047 [-0.140, +0.053] | 0.443 (permutation) |
| team_s59 | group | safe_rate | 50 | 0 | 0 | 0.314 [0.238, 0.398] | -0.079 [-0.193, +0.042] | 0.205 (permutation) |
| team_s59 | others | volunteer_rate | 150 | 0 | 0 | 0.138 [0.101, 0.178] | -0.028 [-0.083, +0.029] | 0.342 (permutation) |
| team_s59 | others | round1_volunteer | 150 | 0 | 0 | 0.253 [0.190, 0.328] | -0.047 [-0.140, +0.053] | 0.434 (permutation) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).
