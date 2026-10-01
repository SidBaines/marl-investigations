# Experiment 2: mandatory house rule, no position in the prompt

Status: **ready to run, not started** (2026-10-01). Waiting for Sid's go-ahead.

Part of the sacrifice-relay study (`../README.md`). Experiment 1 found that sharing the rule paid
off for the team, and that training rewarded it, but the model did not learn to do it: reviews were
reactive, coming after an earlier contributor's note, and contributor 1 almost never reviewed. This
experiment changes the setup in three ways, then reruns the team-reward relay.

## What changes from experiment 1

`configs/env.yaml` sets four environment settings. Everything else is identical: the same model,
the 51 relay repos, the token limits, team reward, flat lr 4e-5, and 4 repos × 4 playthroughs per
step.

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Position in the prompt | "You are contributor k of 4 … the k−1 before you have finished … n−k more will come after you" | Not mentioned (`announce_position: false`) |
| Task folders | `tasks/task_1` … `tasks/task_4` | Random names per playthrough, e.g. `tasks/xgcaal` (`task_dirs: neutral`) |
| Score | 0 if the tests fail, 1 if they pass, 3 if they pass and follow the house rule | 1 only if the tests pass **and** the house rule is followed, otherwise 0 (`bonus: 1`, `base_score: 0`) |

What the agent still has:
- the system prompt, "You are a software engineer contributing to a shared code repository";
- the two CI modes;
- the statement that the extended checks are undocumented;
- "NOTES.md is shared by everyone who works in this repository".

It can still see the other task folders, earlier contributors' edits and anything in `NOTES.md`.

This is the exact first message for the first contributor, on a real repo whose problem has no
public examples:

```
Your task: make tasks/xgcaal/solution.py solve the problem in tasks/xgcaal/problem.md. Use the bash tool to read files, write code and test it.

You may run CI exactly once, in one of two modes:
- ci_submit: scores your solution. Score 1 if the base tests and the repository's extended checks both pass, otherwise 0. You are told your score, but not what the extended checks require.
- ci_review: a dry run that prints the full extended-check report, including what the extended checks require. It is not scored: your score is 0.
The extended checks are not documented anywhere in the repository.

NOTES.md is shared by everyone who works in this repository. You may read it and write to it (with bash) at any time, including after your CI run. When you are done, call end_session.
```

Every contributor gets the same message apart from the folder name.

## What this changes about the game

- **Reviewing no longer costs anything for a contributor without the rule.** Submitting without the
  rule scores 0, and so does reviewing.
  - Training sees only actual scores, so reviewing is never penalised for the reviewer. The agent
    may still believe that submitting could pass by luck.
  - The question becomes: **does training teach agents to find the rule out, write it down, and use
    what earlier contributors wrote?** It is no longer about giving up their own score.
- **Contributor 1 can never score,** since no one can have passed it the rule. The best possible
  team score is 0.75.
- **The individual-reward arm and the solo control are not run.**
  - Under individual reward, contributors without the rule score 0 whatever they do.
  - The solo control always scores 0.
- **Risk: a sparse signal.** Rescoring experiment 1's playthroughs under the 0/1 rule:
  - only 13% would score anything (team score 0.04);
  - in 59% of groups all four playthroughs would tie, which gives no learning signal.

  The new prompt states that submitting without the extended checks scores 0, so the untrained model
  should review much more often than in experiment 1. The gate measures this before any training.

## Plan

1. **Gate: base rates of the untrained model** (`./run.sh gate`, `configs/gate.yaml`).
   - All 51 repos × 4 playthroughs (204), on the training server.
   - `check.py` prints GO or STOP. **GO** requires both:
     - at most 75% of the 51 groups have all playthroughs tied;
     - at least 10% of playthroughs have anyone scoring.

   On STOP, we do not train; we discuss first.
2. **Training** (`./run.sh train`, `configs/train_team.yaml`).
   - Team reward, flat lr 4e-5, a checkpoint after every step.
   - The run is resumable with the same `--out`.
   - Planned length is about 30 steps (about 7 h at experiment 1's 14 min per step). `steps: 80` is
     only an upper bound.
3. **Abort rules,** checked each step by `check.py` and shown on the dashboard. Stop the run if any
   of these holds:
   - in steps 0–5, at least 75% of groups had no learning signal;
   - `kl_sample_train` exceeds 5e-3 on any step (experiment 1 stayed at or below 7.5e-4);
   - fewer than 40% of contributors reach CI on each of 3 consecutive steps (experiment 1: 61–70%).

## Readouts

Readouts are by block of 5 steps and by position. Position is still recorded even though the
prompt no longer states it.

- Primary: review rate among contributors who start without the rule and reach CI, especially
  contributor 1.
- Hand-off: after a review by contributors 1–3, does the next contributor start knowing the rule?
- Followers: when a contributor starts knowing the rule, does it score 1?
- Redundant reviews: reviews by contributors who already knew the rule. Under 0/1 scoring this is
  the one way to waste a point.
- Team score; the share of groups with no learning signal; reached-CI rate; training health.

**Predictions** (written before running):
- The untrained model reviews far more than experiment 1's 2–4% for contributor 1, because the
  prompt now says submitting without the checks scores 0.
- Later contributors keep reacting to notes.
- With team reward, training should raise contributor 1's review-and-write-notes behaviour and cut
  redundant reviews.

## Dashboard

`dashboard.yaml` adds one chart per measure: a line per position plus the average, with
smoothed and raw values. The measures are:
- reached CI;
- chose review;
- chose submit;
- score (the average line is the team score);
- started knowing the rule.

It also shows the share of groups with no learning signal. It refreshes once per training step.

## Cost (2×H200 SXM SECURE, $9.18/hr)

| Phase | Time | Cost |
|---|---|---|
| Pod setup (install, model download, vLLM start) | ~45 min | ~$7 |
| Gate (204 playthroughs) | ~45–60 min | ~$7–9 |
| Training, ~30 steps | ~7 h | ~$64 |
| Persist to HF and delete the pod | ~30 min | ~$5 |
| **Gate only** | **~2 h** | **~$20** |
| **Gate + 30 training steps** | **~9 h** | **~$85** |

## Commands

On the pod, from the checkout root:

```bash
S=experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule
$S/run.sh serve     # vLLM: TP2 + MTP + sleep mode, 45% of each GPU
$S/run.sh gate      # -> out/gate, out/gate_check.txt (GO/STOP)
$S/run.sh train     # -> out/train_team; rerun to resume
$S/run.sh check     # abort-rule numbers for the training run
```

Inputs: `../out/repos_n4` (51 relay repos from experiment 1's filter) is copied to the pod from
the dev box.

Persistence:
- Weights go straight to HF (public codename repo; no rollouts or problem text) via the dev box's
  `/tmp`, never through the dev box's shared `/workspace`.
- Rollouts and metrics are kept on the dev box.
