---
type: concept
title: "A sacrifice is learned only when it pays in practice"
description: "Whether a costly hand-off pays the team depends on whether the receivers use what they are given, which is a property of the model, not of the game design; under team reward the same code_rules game paid for Qwen3.8-27B (followers applied the rule, +0.31) and did not pay for Qwen3.6-35B-A3B (informed followers scored the bonus 14% of the time, −0.12), and the A3B's reviewing fell 21% → 3%; paying off is necessary but was not sufficient (experiment 1) [partial]."
resource: experiments/2026-09-25_sacrifice-relay/payoff.py
tags: [training, rl, credit-assignment, team-reward, sacrifice, multi-agent, relay, code-rules, incentives]
timestamp: 2026-10-05
---

# A sacrifice is learned only when it pays in practice

**The phenomenon.** A game can be designed so that an agent's sacrifice pays
off for the team. Whether it *actually* pays off depends on what the other
agents do with what they receive. RL with team reward responds to the payoff
that actually happens, not the one the design intended. If the receivers do
not use the gift, training pushes the sacrifice down, and does so correctly.
[partial]

- **Designed payoff.** In [code_rules](../entities/env-code-rules.md) with N
  contributors and a bonus m, one early review is worth it for the team if
  followers apply the rule: m(N−1) > N, so 9 > 4 at N=4, m=3.
- **Realised payoff.** This is what the team actually gains. It is measured
  by comparing playthroughs of the same repo in the same training step with
  and without a review (`payoff.py`). It depends on how often a follower who
  starts knowing the rule reaches CI, passes the tests and *applies* the rule.
- **Advantage.** The weight training puts on an action: here, a
  playthrough's team score minus the mean of the other playthroughs of the
  same repo in that step. Its sign follows the realised payoff.

## Evidence: the same game, two models [partial]

Both runs: the [relay](../entities/protocol-relay.md) with 4 contributors,
0/1/3 scoring (0 if the tests fail, 1 if they pass, 3 if they pass and follow
the rule), team reward, LoRA rank 32 on the
[local backend](../entities/local-backend.md), flat lr 4e-5, the 51 repos,
one seed, steps 0–29. Brackets are 95% bootstrap intervals over groups.

| | Experiment 1: [Qwen3.8-27B](../entities/qwen3-8-27b.md) | Experiment 3: [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md) |
|---|---|---|
| Batch per step | 4 repos × 4 | 8 repos × 4 |
| Prompt | position stated, numbered folders | no position, random folders, the `checks` sentence |
| Rule reached the next contributor after a review | 94% | 96% |
| Informed followers who scored 3 | 54% of all informed followers; about 82% of those who submitted | 14%, although 83% of their submissions passed the base tests |
| Team score, review vs no review in the same group | **+0.31** [+0.21, +0.42] (72 groups) | **−0.12** [−0.17, −0.07] (145 groups) |
| A review by contributor 1 | +0.62 [+0.30, +0.97] | −0.08 [−0.20, +0.06] |
| Training's advantage, reviewed vs submitted | contributor 1: +0.57 vs +0.02 | −0.08 [−0.12, −0.03] vs +0.05 |
| Review rate (no rule at start, ran CI), steps 0–9 → 20–29 | 10.6% → 7.2% | 21% → 3% |

- **Experiment 3: the payoff was negative and the behaviour went away.** A
  review cost the reviewer about 0.7 points and gained each follower about
  0.14. The A3B's followers mostly knew the rule and did not apply it. Under
  team reward, reviewing fell from 21% to 3%; under individual reward, from
  11% to 0%. [partial]
- **Experiment 2: the review was the only way to score, and it was learned.**
  Under 0/1 scoring (1 only with the tests *and* the rule), applying the rule
  is not optional. A review costs a contributor without the rule nothing.
  Team training raised contributor 1's review rate from 17% to 80% (27B) and
  from 38% to 77% (A3B). The A3B's followers also learned to apply the rule:
  40% → 65% in experiment 2.1, and 25% → 65% on held-out problems. [partial]
- **Experiment 1: the payoff was positive, yet the behaviour was not
  learned.** So a realised payoff is necessary but not sufficient. See
  [rewarded but not learned](rewarded-choice-not-learned.md). [partial]

Sources: [study report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md)
(sections 3.1, 3.7) and [experiment 1](../../sources/sacrifice-relay-experiment-1.md).

## Why the A3B's followers skip the rule [open]

- **Hypothesis (from the source):** when passing the tests alone earns a
  point, the A3B treats the rule as optional. Transcripts have not been read.
- **Consistent with it:** under 0/1 scoring, where the rule is required, the
  untrained A3B also applied a known rule poorly: 25% on held-out problems,
  40% in experiment 2.1's first steps. Training raised both to 65%. The
  untrained 27B applied it 57% of the time in the same held-out eval.
- **Under new rule forms** the trained A3B's application fell again, to 33%
  ([behaviour vs format](behaviour-transfers-across-format.md)). Applying an
  unfamiliar rule is a weak point of this model.

## Reading

- **The incentive belongs to the pair (game, policy), not to the game.** The
  same environment gives opposite incentives on two models. It can also change
  during training as followers get better or worse at using the rule. Check the
  realised payoff per model before reading a null result as "RL cannot teach
  this". [partial]
- **The decline can feed itself.** When reviewing is rare, few followers start
  with the rule. Followers then get little practice applying it, and a review
  stays unprofitable. Experiment 3 was paused at step 30 of 80, so whether this
  locks in was not seen. [open]
- **What the receivers can do limits what the giver can learn.** In the
  program's [swarm](../entities/protocol-swarm.md) and
  [coordinator](../entities/protocol-coordinator.md) protocols, a worker's
  report or a peer's scratchpad is worth producing only if the reader uses
  it. The same logic applies there, untested. [open]

## Tensions

- **The payoff check is observational.** Who reviews is not random, so
  playthroughs with a review differ before the review happens. The
  within-group comparison cancels problem difficulty, not this selection
  (see [rewarded but not learned](rewarded-choice-not-learned.md)). [open]
- **Experiment 1 vs experiment 3 differs in more than the model.** The
  batch, adapter, hot-load tolerance and three prompt changes all differ.
  So "the 27B's followers apply the rule and the A3B's do not" may be partly
  a prompt effect. Experiment 3's own README calls the comparison
  descriptive. [open]
- **Denominators.** The experiment 3 report gives "informed followers scored
  3 only 14% of the time" next to "83% of their submissions passed the base
  tests". The first appears to count all informed followers and the second
  only those who submitted. Experiment 1's 54% (all) vs about 82%
  (submitters) shows how far apart the two bases can be. [open]

## Open questions

- Read experiment 3's transcripts: do informed A3B followers see the rule and
  decide to skip it, or fail to apply it? [open]
- Make the rule required for the 1 point too, or raise the bonus, and test
  whether a costly review is then learned. [open]

See also: [the synthesis](../syntheses/does-rl-teach-sacrifice.md),
[reactive information gathering](reactive-information-sharing.md).
