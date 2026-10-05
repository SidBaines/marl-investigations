# li_shirado_games

One-shot economic games of Li & Shirado: does the model give up points so a partner or group gains? Dictator (share of 100 points given), prisoner's dilemma (give 100 doubled points vs keep) and a 4-player public-goods game (contribute vs keep), each as 100 independent single-turn trials; ultimatum and second/third-party punishment are available via task_args.games.

Source: {'package': 'inspect-ai', 'version': '0.3.276'}. Citation: Yuxuan Li and Hirokazu Shirado. Spontaneous Giving and Calculated Greed in Language Models. EMNLP 2025. arXiv:2502.17720 (prompts: Appendix A).

| Cell | Condition | Metric | n | Unparsed | Errors | Mean [95% CI] | Gain vs base [95% CI] | p (test) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | dictator | give_share | 100 | 0 | 0 | 0.308 [0.289, 0.329] | — | — |
| base | prisoners_dilemma | cooperate | 100 | 0 | 0 | 0.230 [0.158, 0.322] | — | — |
| base | public_goods | contribute | 100 | 0 | 0 | 0.120 [0.070, 0.198] | — | — |
| team_s59 | dictator | give_share | 100 | 0 | 0 | 0.315 [0.295, 0.334] | +0.006 [-0.022, +0.033] | 0.703 (permutation) |
| team_s59 | prisoners_dilemma | cooperate | 100 | 0 | 0 | 0.200 [0.133, 0.289] | -0.030 [-0.140, +0.090] | 0.731 (permutation) |
| team_s59 | public_goods | contribute | 100 | 0 | 0 | 0.140 [0.085, 0.221] | +0.020 [-0.070, +0.110] | 0.836 (permutation) |

n counts parsed values. Unparsed counts replies with no readable decision: excluded from n where the harness keeps them (Inspect games), re-asked where the harness re-asks (FAIRGAME). Errors counts harness/model failures (excluded). Intervals: Wilson for binary rates over independent units, otherwise a 2,000-draw seeded bootstrap (over samples when tasks repeat across cells). Gains over the baseline cell are paired on shared samples where the suite pairs them (McNemar or sign-flip p), otherwise an independent difference (two-sample bootstrap, permutation p).
