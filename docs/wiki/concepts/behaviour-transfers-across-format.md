---
type: concept
title: "Learned behaviour vs learned format: the transfer test"
description: "RL on a fixed environment repeats one prompt, one set of tool names and a few dozen problems, so a gain may be a habit tied to that surface; experiment 2's team-trained policies kept their gain on held-out problems (27B 0.085 → 0.300, A3B 0.060 → 0.320) and under a fully changed surface (27B 0.115 → 0.275; A3B 0.050 → 0.140), with no format-matching habits in any of 800 contributors, so the 27B learned the behaviour; the A3B's followers stumbled on new rule kinds [partial]."
resource: experiments/2026-09-25_sacrifice-relay/exp2_eval/README.md
tags: [evaluation, transfer, generalisation, memorisation, rl, relay, code-rules, held-out]
timestamp: 2026-10-05
---

# Learned behaviour vs learned format: the transfer test

**The question.** Training in one environment repeats the same problems, the
same prompt text, the same tool names, the same notes file and a few rule
forms. A trained gain could be the behaviour itself (find the rule, write it
down, use it). Or it could be a habit tied to that exact surface. The way to
tell them apart is to change the surface one layer at a time and see how much
of the trained gain survives, measured against the untrained model on the
same games. [partial]

**What repeated in sacrifice-relay training**
([report §3.8](../../sources/sacrifice-relay-experiments-1-3-and-evals.md)):

- the 51 repos, each problem fixed to a slot (each repo drawn about 2.4 times
  per 30 steps at 4 repos per step);
- the first message, the tool names and descriptions, the CI wording and
  `NOTES.md`, identical in every game;
- 10 rule forms (3 families × 3–4 names). Only the random ID was new.

## Checks inside training (no new runs) [partial]

From the saved training games, steps early vs late:

- **No memorisation of problems.** Among submitters, the base-test pass rate
  did not rise with repeats. 27B: 0.81 on a repo's first draw, 0.84 on its
  fifth. A3B: 0.79–0.89 with no trend over 12–13 draws.
- **Notes are not a template.** With IDs, keys and folders masked, the notes
  written in the last five steps are all worded differently (27B: 182 of 182;
  A3B: 393 of 394).
- **Reviews are not a reflex.** The median number of tool calls before a
  review was 9 at the start and at the end for the 27B, and 13–14 for the A3B.

These rule out the crudest shortcuts. They cannot show transfer to a new
format.

## The transfer eval (2026-10-05) [partial]

**Regime.** Untrained vs team-trained policy of each model:

- policies: [Qwen3.8-27B](../entities/qwen3-8-27b.md) after step 59 and
  [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md) after step 79 of
  experiment 2's team arm;
- game: [code_rules](../entities/env-code-rules.md) on the
  [relay](../entities/protocol-relay.md), N=4, 0/1 scoring, the `checks`
  prompt, no position, the same token limits;
- 25 held-out repos × 2 games = 50 games (200 contributors) per cell; the
  same seed for every policy, so games are paired;
- one training seed; untrained and trained share one vLLM server per model.

**Conditions run** (the full design has more):

- `heldout`: new problems, independent of every problem used before (see
  [DeepCoder](../entities/deepcoder.md)); the format is unchanged.
- `far`: every surface change at once:
  - reworded first message and system prompt;
  - CI tools renamed (`grade_solution` / `inspect_checks`) with reworded
    descriptions;
  - reworded CI replies;
  - notes moved to `docs/handoff.txt`;
  - three rule forms never used in training.

Team score is the mean 0/1 score of the four contributors (best possible
0.75). Brackets are 95% bootstrap intervals over repos. p is McNemar on
"anyone scored", matched by game.

| | 27B untrained → team | A3B untrained → team |
|---|---|---|
| `heldout` team score | 0.085 → **0.300** (+0.215 [+0.125, +0.300], p = 3e-6) | 0.060 → **0.320** (+0.26 [+0.19, +0.33], p = 1e-7) |
| `heldout` contributor 1 reviewed (of its CI runs) | 14% → 93% | 36% → 76% |
| `far` team score | 0.115 → **0.275** (+0.16 [+0.095, +0.235], p = 4e-5) | 0.050 → **0.140** (+0.09 [+0.01, +0.175], p = 0.017) |
| `far` contributor 1 reviewed | 37% → 74% | 43% → 70% |
| Trained policy, `far` minus `heldout` team score | −0.025 [−0.11, +0.05] | −0.18 [−0.26, −0.10] |

- **New problems: full transfer.** Both trained policies score at least as
  well as at the end of training (27B 0.300 vs 0.267; A3B 0.320 vs 0.268).
  So the gain was not memorised repos. [partial]
- **A changed surface: the 27B keeps almost all of its gain.** Its `far`
  team score is within noise of `heldout`. [partial]
- **The A3B keeps reviewing but keeps only about a third of its gain** over the
  untrained model (+0.09 under `far` vs +0.26 on held-out problems). Contributor 1's
  review rate barely moves (−8.5 points, interval includes 0). But followers
  who start knowing the rule apply it much less: 65% → 33%, −32 points
  [−44, −21]. [partial]
- **No format-matching habits.** In every `far` game (4 cells × 200 = 800
  contributors), no contributor called a tool that does not exist there,
  mentioned the old `NOTES.md` in a command, or submitted the rule's ID in the
  wrong form. [partial]

Sources: [study report §4.4](../../sources/sacrifice-relay-experiments-1-3-and-evals.md);
aggregate tables `experiments/2026-09-25_sacrifice-relay/exp2_eval/results/`
(in git at 3d6551f).

## Reading

- For the 27B, team training taught the behaviour, not the format. [partial]
- For the A3B, what breaks under `far` is the *follower* step: applying an
  unfamiliar rule form. The report says this is most likely the new rule
  kinds. `far` changes everything at once, and the single-change cells
  (`new_rules` alone, `tools`, `notes`, …) were not run, so the cause is not
  isolated. [open]
- The A3B's follower weakness also appears in training under 0/1/3 scoring
  ([pays in practice](sacrifice-pays-in-practice.md)). The same model step
  fails in two different settings.

## Tensions

- **Contributor count.** The report says "across 1,200 contributors in the
  `far` games" no habit errors occurred. The results file has 4 `far` cells
  of 50 games × 4 contributors, which is 800. The zero count holds either
  way. [open]
- **Only two of the ten planned conditions ran** (plus the help request), on
  25 of the planned 51 repos, with the final team policies only. The
  individual-arm, 2.1 and step-29 policies in the design were not evaluated.
  [open]
- **The 27B's followers did not improve.** Followers who started knowing the
  rule applied it 57% → 51% on `heldout` (not significant). So the 27B's
  gain is all on the finding side. [partial]

## Open questions

- Run the single-change cells to find which change hurts the A3B. [open]
- Train on varied formats (wording, tool names, notes location, rule kinds),
  holding some back for eval. [open]

See also: [the synthesis](../syntheses/does-rl-teach-sacrifice.md),
[harness-dependent helpfulness](harness-dependent-helpfulness.md) (a
different kind of transfer: to another harness altogether).
