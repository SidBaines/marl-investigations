# li_shirado_games

One-shot economic games of Li & Shirado: does the model give up points so a partner or group gains? Dictator (share of 100 points given), prisoner's dilemma (give 100 doubled points vs keep) and a 4-player public-goods game (contribute vs keep), each as 100 independent single-turn trials; ultimatum and second/third-party punishment are available via task_args.games.

Source: {'package': 'inspect-ai', 'version': '0.3.276'}. Citation: Yuxuan Li and Hirokazu Shirado. Spontaneous Giving and Calculated Greed in Language Models. EMNLP 2025. arXiv:2502.17720 (prompts: Appendix A).

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | dictator | give_share | 99 | 1 | 0 | 0.219 [0.197, 0.240] | — | — |
| base | prisoners_dilemma | cooperate | 100 | 0 | 0 | 0.010 [0.002, 0.054] | — | — |
| base | public_goods | contribute | 100 | 0 | 0 | 0.020 [0.006, 0.070] | — | — |
| team_s79 | dictator | give_share | 100 | 0 | 0 | 0.235 [0.215, 0.254] | +0.016 [-0.012, +0.046] | 0.282 (permutation) |
| team_s79 | prisoners_dilemma | cooperate | 100 | 0 | 0 | 0.020 [0.006, 0.070] | +0.010 [-0.020, +0.050] | 1 (permutation) |
| team_s79 | public_goods | contribute | 100 | 0 | 0 | 0.000 [0.000, 0.037] | -0.020 [-0.050, +0.000] | 0.502 (permutation) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).
