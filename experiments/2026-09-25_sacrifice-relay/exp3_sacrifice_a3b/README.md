# Experiment 3: the sacrifice again, on Qwen3.6-35B-A3B, team and individual reward

Status (2026-10-05): **running** (two pods, one per arm, 80 steps each).

Part of the sacrifice-relay study (`../README.md`). Sid asked for experiment 1's game, where reviewing is a real
sacrifice (a contributor without the rule could score 1 by submitting), on the A3B, with the setup changes made since
experiment 1, and with both reward arms (Sid, 2026-10-05: "Run both team and individual arms, keep original tool
text").

## Design (as planned before the runs)

**Scoring as experiment 1** (`configs/env.yaml`): 0 if the base tests fail, 1 if they pass, 3 if they pass and the
house rule is followed. Reviewing (`ci_review`) scores 0, so a contributor without the rule gives up the point it
would likely get by submitting (base tests pass for about 83% of submitters).

**Prompt and setup as experiment 2:**
- no position in the prompt (`announce_position: false`) and random task folder names (`task_dirs: neutral`): the
  model has to work out from the repo itself whether anyone came before it;
- the `checks` sentence: the extended checks "check repository-specific conventions that cannot be worked out from
  the task, the code or the tests" (in experiment 1, followers often hoped the checks might pass by luck);
- unchanged from both experiments: the original CI tool descriptions (`ci_review` "Spend your CI run on an unscored
  report…"), visible `NOTES.md`, one CI run, the token limits (12,288 per contributor, 6,144 per reply), the 51
  repos.

**Model and training as experiment 2's A3B arms:** Qwen3.6-35B-A3B, 8 repos × 4 playthroughs per step, flat lr 4e-5,
LoRA rank 32 on attention, linear attention and the shared expert (routed experts and router frozen),
`local_adapter_check_tol=0.5`, a checkpoint after every step, 80 steps in full.

| Run | Config | Reward |
|---|---|---|
| `out/train_team` | `configs/train_team.yaml` | every contributor gets the repo's mean score |
| `out/train_individual` | `configs/train_individual.yaml` | each contributor gets its own 0/1/3 score |

Composed as `run.sh` runs them:
- the team arm differs from experiment 2's A3B team arm (`../exp2_mandatory_rule/out/train_team_checks_a3b`) only in
  the scoring (bonus 3, base score 1), the server path and the name. So the two isolate the effect of making the
  review cost the reviewer;
- the individual arm differs from the team arm only in the reward and the name;
- against experiment 1's trial (`../out/trial_team`) four things change at once: the model (and its batch, adapter
  and hot-load tolerance) and the three prompt changes. That comparison is descriptive only.

The first contributor's first message (every contributor gets the same, apart from the folder name):

```
Your task: make tasks/wjgiue/solution.py solve the problem in tasks/wjgiue/problem.md. Use the bash tool to read files, write code and test it.

You may run CI exactly once, in one of two modes:
- ci_submit: scores your solution. Score 0 if the base tests fail, 1 if they pass, and 3 if they pass and the repository's extended checks also pass. You are told your score, but not what the extended checks require.
- ci_review: a dry run that prints the full extended-check report, including what the extended checks require. It is not scored: your score is 0.
The extended checks are not documented anywhere in the repository. They check repository-specific conventions that cannot be worked out from the task, the code or the tests.

NOTES.md is shared by everyone who works in this repository. You may read it and write to it (with bash) at any time, including after your CI run. When you are done, call end_session.
```

## Readouts

From `../exp2_mandatory_rule/check.py` (by position and pooled over steps; it now reports "scored (any points)" and
"scored with the rule followed", the 3s), `../analyze.py --bonus 3` and `../payoff.py` (team arm; its "what training
rewarded" section assumes team credit):
- **Primary:** the review rate of contributors who started without the rule and ran CI, above all contributor 1's,
  early (steps 0–9) vs late (steps 70–79), in each arm.
- Redundant reviews (reviewing when the rule was already known), hand-off (the next contributor starts knowing the
  rule), followers scoring 3 when they knew the rule, team score, and the payoff check (team score with vs without a
  review in the same group).

Expectations:
- **Team arm:** reviews rise, contributor 1's most, as in experiment 2 (where reviewing cost nothing).
- **Individual arm:** the reviewer's own reward falls by about a point, so the direct pressure lowers reviewing. Sid's
  exposure hypothesis predicts reviewing could still rise, through followers rewarded for using notes; evidence would
  be a late review rate above the early one.

**Abort rules** (`check.py`, as experiment 2): `kl_sample_train` > 5e-3 at any step; ≥ 75% of groups with no
learning signal over steps 0–5; reached-CI rate below 40% on each of the last 3 steps.

## Commands

On each pod (from the checkout root; inputs: `../out/repos_n4` copied from the dev box):

```bash
S=experiments/2026-09-25_sacrifice-relay/exp3_sacrifice_a3b
$S/run.sh serve
$S/run.sh dashboard
ARM=team $S/run.sh train          # team pod; ARM=individual on the other; rerun to resume
ARM=team $S/run.sh check
```

Cost estimate: 2×H200 SXM SECURE at $9.18/hr per pod; experiment 2's A3B arms took about 10.5 min per step, so about
14 h and about $130 per arm, about $260 for both.
