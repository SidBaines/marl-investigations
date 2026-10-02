# Experiment 2: mandatory house rule, no position in the prompt

Status: **done (2026-10-02)**: 30 steps of team-reward training with the `checks` prompt variant. See Results.

Part of the sacrifice-relay study (`../README.md`). Experiment 1 found that sharing the rule paid
off for the team, and that training rewarded it, but the model did not learn to do it: reviews were
reactive, coming after an earlier contributor's note, and contributor 1 almost never reviewed. This
experiment changes the setup in three ways, then reruns the team-reward relay.

## Results (2026-10-02)

Run `out/train_team_checks`: team reward, 4 repos × 4 playthroughs per step, flat lr 4e-5. The prompt is experiment 2's
plus one sentence (the `checks` variant). 30 steps (0–29), 22:48–05:29 UTC, 13.3 min per step. ± is a 95% interval.
Numbers come from `check.py --every 10` and `health.py` (`out/analysis/`, also on HF under `exp2/analysis/`).

| | Steps 0–9 | Steps 10–19 | Steps 20–29 |
|---|---|---|---|
| Team score (mean of the 4 contributors' 0/1 scores) | 0.089 ± 0.029 | 0.152 | **0.177 ± 0.034** (z = 3.9 vs steps 0–9) |
| Playthroughs where anyone scored | 22% | 38% | 46% |
| Groups with no learning signal (all 4 playthroughs tied) | 42% | 18% | 18% |
| Contributor 1 chose review (among those who ran CI) | 17% (19/112) | 40% (46/114) | **52% (52/100)** (z = 5.7) |
| All positions chose review (no rule at start, ran CI) | 26% | 37% | 53% |
| Started knowing the rule (all contributors) | 19% | 31% | 38% |
| Rule reached the next contributor after a review by 1–3 | 94% | 98% | 98% |
| Redundant reviews (knew the rule, reviewed anyway) | 18% | 14% | 12% |
| Reached CI | 60% | 64% | 63% |

- **Training taught the first mover to review.**
  - Contributor 1 never scores itself, so it only gains through the team score; its review rate tripled.
  - The team score doubled, and wasted reviews by contributors who already knew the rule fell.
  - This is the behaviour experiment 1 rewarded but did not learn.
- **What still limits the team.** Followers who start knowing the rule score in only 46–47% of cases, with no
  change during training: about 68% reach CI, and some fail the base tests. More reviewing cannot fix this; the
  budget and coding ability bind.
- **Comparison with the prompt as planned.** Steps 0–1 under the planned prompt, without the `checks` sentence, had
  2 reviews in 82 CI runs and 0 of 32 playthroughs scoring (`out/old_pod`, `analysis/baseline_prompt_steps_0_1.txt`).
  With the sentence, step 0 already had 5 reviews in 32. The sentence gives the agent a fact: the hidden checks
  cannot be passed by luck. Without it, the untrained model treats a submit as a gamble worth taking.
- **Training health:** `kl_sample_train` ≤ 7.4e-4 at every step, and the mean IS ratio stayed at 1.000 ± 0.0002.
  No errors or restarts. Only steps with no informative group skip the update.
- **Caveats:**
  - One seed and one arm.
  - The 51 repos repeat (each drawn about 2.4 times).
  - The `checks` variant was chosen from a partial, timeout-biased test (Log of the night).
  - The prompt now says the checks cannot be worked out from the task or code. This states a fact and recommends
    neither CI mode, but it makes reviewing easier to discover than in experiment 1.

Artefacts:
- **HF `sidbaines/amber-baton` under `exp2/`:**
  - adapters for all 30 steps;
  - trainer states for steps 9, 19 and 29 (enough to resume);
  - checkpoint manifests, metrics, config and analysis.
- **Dev box (private):** `out/` holds rollouts, variant runs, the old pod's run and logs.
- **Resume:** restore the run dir and rerun `VARIANT=checks run.sh train`.

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
  should review much more often than in experiment 1. Training's first steps measure this (below).

## Plan

1. **No separate gate** (Sid, 2026-10-01). Step 0 samples the untrained model (4 repos × 4
   playthroughs), and groups whose playthroughs all tie make no update, so the first steps measure
   the base rates. The early abort rule below stops the run if the signal is too sparse.
   - `./run.sh gate` (`configs/gate.yaml`: all 51 repos × 4) remains available if a full base-rate
     measurement is wanted later. Its GO rule: at most 75% of groups tied and at least 10% of
     playthroughs scoring.
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

A live page served from the pod (`./run.sh dashboard`; `dashboard.yaml` sets the charts), opened at
`https://<pod-id>-8888.proxy.runpod.net/?key=<key>` from any browser. It refreshes itself every
minute; nobody republishes it. It shows one chart per measure, with a line per position plus the
average, smoothed and raw. The measures are:
- reached CI;
- chose review;
- chose submit;
- score (the average line is the team score);
- started knowing the rule.

It also shows the share of groups with no learning signal. New numbers arrive once per training
step (about 14 min), and the current step's progress shows in between.

## Cost (2×H200 SXM SECURE, $9.18/hr)

| Phase | Time | Cost |
|---|---|---|
| Pod setup (install, model download, vLLM start) | ~45 min | ~$7 |
| Training, ~30 steps | ~7 h | ~$64 |
| Persist to HF and delete the pod | ~30 min | ~$5 |
| **Stopped early by the abort rule after step 5** | **~2.75 h** | **~$25** |
| **Full ~30-step run** | **~8.25 h** | **~$76** |
| (Optional gate, if run: 204 playthroughs) | (+1–1.5 h) | (+$9–14) |

## Commands

On the pod, from the checkout root:

```bash
S=experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule
$S/run.sh serve       # vLLM: TP2 + MTP + sleep mode, 45% of each GPU
$S/run.sh dashboard   # live page on port 8888; URL with key in out/dashboard/serve.json
$S/run.sh train       # -> out/train_team; rerun to resume
$S/run.sh check       # abort-rule numbers for the training run
```

Inputs: `../out/repos_n4` (51 relay repos from experiment 1's filter) is copied to the pod from
the dev box.

Persistence:
- Weights go straight to HF (public codename repo; no rollouts or problem text) via the dev box's
  `/tmp`, never through the dev box's shared `/workspace`.
- Rollouts and metrics are kept on the dev box.

## Log of the night (2026-10-01/02)

- **Pod swap.** The first pod (`n7gq0g825l2s5v`) had a GPU stuck in thermal slowdown: 86 °C at 345 MHz, against
  1,980 MHz on the other GPU. Sampling ran 2.6× slower, about 26 min per step. It was replaced by `su24myzficgx5o`
  and deleted (receipt in the orchestration dir).
- **Experiment 2 as planned** (`out/old_pod/train_team_hot_gpu`). Steps 0–1 of team training gave no learning signal,
  so the learner skipped both steps:
  - 2 of 82 CI runs were reviews, and 0 of 32 playthroughs scored;
  - submitters often wrote notes about the failed extended checks, but followers still submitted;
  - their reasoning: a review is a sure 0, while a submit "has a chance". They did not realise the extended checks
    cannot be passed by luck.
- **Prompt variants** (Sid: try prompts that do not steer toward either CI mode). These are new `code_rules` settings,
  each adding facts only (`configs/variants/`):
  - `tools`: symmetric CI tool descriptions. The original says ci_review "spends" the CI run.
  - `others`: "Other contributors also work in this repository, each on their own task."
  - `checks`: the extended checks "check repository-specific conventions that cannot be worked out from the task,
    the code or the tests".
  - `all`: all three together.
- **Variant test** (`out/variants/`). The untrained model ran on the first 8–16 repos × 4. The test was stopped early
  because the eval server was overloaded (KV cache full, ~2.8k preemptions), and 22 of the 34 finished playthroughs
  failed on client timeouts. The survivors lean towards short playthroughs. Among them:
  - `checks`: 4/6 CI runs were reviews, and 3/4 playthroughs scored;
  - current prompt: 2/10 and 1/4;
  - `all`: 2/7 and 1/3;
  - `tools`: 1/1 and 1/1;
  - `others`: none survived.
- **Decision:** train with `checks`, the least-changed variant with a clear signal (22:48 UTC, `out/train_team_checks`).
  Its first steps re-measure the base rates without the timeout bias.
