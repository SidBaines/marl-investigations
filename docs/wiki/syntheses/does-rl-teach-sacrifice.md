---
type: synthesis
title: Does RL teach agents to give up their own reward so later agents score more?
description: "Current answer: only under team reward, and only when it pays in practice. Team reward taught the first contributor to reveal the rule when that cost it nothing and paid the team (experiment 2: 27B 17% → 80%, A3B 38% → 77%), and wiped reviewing out when a costly review did not pay because followers rarely applied the rule (experiment 3, A3B: 21% → 3%); individual reward never taught it; the one costly sacrifice that did pay (experiment 1) was not learned in 30 steps; one seed throughout [partial]."
resource: docs/sources/sacrifice-relay-experiments-1-3-and-evals.md
tags: [synthesis, rl, multi-agent, sacrifice, credit-assignment, team-reward, individual-reward, relay, code-rules, transfer]
timestamp: 2026-10-05
---

# Does RL teach agents to give up their own reward so later agents score more?

## Current answer [partial]

**Only under team reward, and only when the review pays off for the team in
practice. A sacrifice that costs the reviewer a point and pays the team has
not yet been learned.** Every training run is one seed.

1. **Team reward taught the first contributor to review when reviewing paid
   off.** In experiment 2, a contributor without the rule scores 0 whether it
   reviews or submits, and the team scores only if someone finds the rule.
   - On Qwen3.8-27B, team training raised contributor 1's review rate from
     17% to 80% over 60 steps, and the team score from 0.089 to 0.267.
   - On Qwen3.6-35B-A3B it rose from 38% to 77% over 80 steps (team score
     0.049 → 0.268). [partial]
2. **Team reward wiped reviewing out when the sacrifice did not pay.** In
   experiment 3 (A3B, 0/1/3 scoring), a review costs the reviewer about a
   point.
   - Followers who were handed the rule scored 3 only 14% of the time, so
     playthroughs with a review scored 0.12 lower than the same repo's
     playthroughs without one.
   - Training's advantage pointed against reviewing, and the review rate fell
     from 21% to 3% in 30 steps. [partial]
3. **Individual reward never taught reviewing.**
   - Experiment 2's individual arm (27B) stayed flat.
   - Experiment 2.1's followers did not learn to review (27B flat; A3B fell
     from 35% to 6%).
   - Experiment 3's individual arm reached 0%. [partial]
4. **The one costly sacrifice that did pay off was not learned.** In
   experiment 1 (27B, 0/1/3 scoring), a review raised team score by +0.31 and
   training's advantage favoured it, yet the rate fell from 10.6% to 7.2% in
   30 steps. [partial]

**Reading.** Team RL follows the payoff a review actually delivers, and that
depends on whether followers use the rule, not on the payoff the game was
designed to offer ([pays in practice](../concepts/sacrifice-pays-in-practice.md)).
What has been learned so far is first-mover information gathering that costs
the gatherer nothing. Whether team RL learns a review that costs the reviewer
and pays the team is still open: experiment 1 is the only such regime, and it
ran 30 steps on one seed. [open]

**What was learned carries over to new problems and a new format, but not to
other cooperation tasks.**

- Experiment 2's team-trained policies keep their gain on held-out problems.
- The 27B also keeps it under a fully changed surface; the A3B keeps about a
  third of its gain over the untrained model (+0.09 vs +0.26)
  ([behaviour vs format](../concepts/behaviour-transfers-across-format.md)).
  [partial]
- The policies do not become more generous or helpful in standard cooperation
  evals. The one exception is a small HiddenBench gain for the A3B
  ([standard cooperation evals](../entities/standard-coop-evals.md)). [partial]

## Evidence by regime

All runs: [code_rules](../entities/env-code-rules.md) on
[relay_n4](../entities/protocol-relay.md) (4 contributors in turn, one CI run
each, shared notes file visible), the same 51 repos of 4
[DeepCoder](../entities/deepcoder.md) problems, LoRA rank 32 on the
[local backend](../entities/local-backend.md), flat lr 4e-5, a group
leave-one-out baseline, a token-sum loss, one seed. "Review rate" is
contributor 1's reviews out of its CI runs (experiments 2 and 2.1) or, where
marked, the review rate of all contributors who started without the rule and
ran CI. Early is steps 0–9; late is the last 10 steps run.

| Run | Model, batch | Scoring | Reward | Steps | Review rate, early → late | Team score, early → late | Did a review pay the team? |
|---|---|---|---|---|---|---|---|
| Exp 1 | 27B, 4 × 4 | 0 / 1 / 3 | team | 30 | 10.6% → 7.2% (all) | 0.545 → 0.631 | yes, +0.31 |
| Exp 2 team | 27B, 4 × 4 | 0 / 1 (rule required) | team | 60 | 17% → 80% | 0.089 → 0.267 | yes: no one scores without it |
| Exp 2 individual | 27B, 4 × 4 | 0 / 1 | individual | 30 | 23% → 30% | 0.078 → 0.081 | as above, but never to the reviewer |
| Exp 2.1 | 27B, 4 × 4; frozen trained contributor 1 | 0 / 1 | individual, contributors 2–4 | 30 | followers without the rule, after an earlier CI run: 34% → 32% | 0.114 → 0.152 | as above |
| Exp 2 team | A3B, 8 × 4 | 0 / 1 | team | 80 | 38% → 77% | 0.049 → 0.268 | as above |
| Exp 2.1 | A3B, 8 × 4; frozen trained contributor 1 | 0 / 1 | individual, contributors 2–4 | 31 of 80 (paused) | followers without the rule, after an earlier CI run: 35% → 6% | 0.190 → 0.259 | as above |
| Exp 3 team | A3B, 8 × 4 | 0 / 1 / 3 | team | 31 of 80 (paused) | 21% → 3% (all); contributor 1 19% → 2% | 0.481 → 0.597 | no, −0.12 |
| Exp 3 individual | A3B, 8 × 4 | 0 / 1 / 3 | individual | 31 of 80 (paused) | 11% → 0% (all) | 0.508 → 0.655 | not computed |

- Under 0/1 scoring contributor 1 can never score, so the best possible team
  score is 0.75. Under 0/1/3 the team score is on a 0–3 scale. Do not compare
  team scores across the two scorings.
- In experiments 1 and 3, the team score rose mainly because more
  contributors reached CI and submitted, the 1-point route
  ([token budget binding](../concepts/token-budget-binding.md)). [partial]
- Experiment 2 also changed the prompt (see [Tensions](#tensions)), so
  experiment 1 vs experiment 2 is not a one-factor comparison. Experiment 3's
  team arm vs experiment 2's A3B team arm differs only in the scoring (plus
  the server path and run name). That pair is the cleanest test of
  "costless vs costly review" on one model. [partial]

Sources: [the study report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md)
(sections 2–3) and [experiment 1](../../sources/sacrifice-relay-experiment-1.md).

## Pre-registered hypotheses: status

| Hypothesis | Status |
|---|---|
| H-team: under team reward the first-mover review rate rises, with team score | **Supported when the review costs the reviewer nothing** (experiment 2, both models). **Reversed when a costly review does not pay** (experiment 3). **Not seen in 30 steps when a costly review did pay** (experiment 1). [partial] |
| H-individual (Sid's exposure hypothesis): the rate also rises under individual reward, because followers are rewarded for using notes | **Not supported.** Tested four ways: experiment 2's individual arm, 2.1 on the 27B, 2.1 on the A3B, and experiment 3's individual arm. Followers learned to *use* the rule (27B 50% → 62%, A3B 40% → 65%), but never to find it. [partial] |
| Control: in `solo`, the review rate falls toward 0 | **Untested** (not run in any experiment). [open] |
| Understanding vs habit: hand-off stays reliable and redundant reviews fall | Hand-off stayed at 94–98% in every run. Redundant reviews fell for the A3B (54% → 21%). For the 27B they fell (18% → 10%), then rose again to 24% by steps 50–59. The transfer eval found no format-matching habits ([behaviour vs format](../concepts/behaviour-transfers-across-format.md)). [partial] |

"Redundant review": a contributor who already knew the rule reviewed anyway
and scored 0.

## Why experiment 2 learned and experiment 1 did not [open]

Experiment 2 changed four things at once, so which one mattered is not known.

- **The review stopped costing the reviewer anything.** Under 0/1 scoring a
  contributor without the rule scores 0 either way, so training never
  penalised a review for the reviewer.
- **One prompt sentence made reviewing easier to find.** The sentence: the
  extended checks "cannot be worked out from the task, the code or the tests".
  - Without it, the untrained 27B reviewed in 2 of 82 CI runs; it reasoned
    that a submit "has a chance".
  - With it, contributor 1 reviewed 17% of the time from step 0. That is far
    more choice events per step than experiment 1's 3–4.
  - See [rewarded but not learned](../concepts/rewarded-choice-not-learned.md).
- **No position in the prompt, and random folder names.** Contributor 1 can no
  longer read off that it is first.
- **A denser learning signal at the start, then a richer one.** 42% of groups
  had no learning signal at steps 0–9, falling to 8% by steps 50–59
  ([zero-variance groups](../concepts/zero-variance-groups.md)).

## Why experiment 3 unlearned it [partial]

The A3B's followers rarely applied a rule they had been given when passing the
tests alone earned a point.

- The hand-off itself worked: 96% of reviews by contributors 1–3 reached the
  next contributor.
- But informed followers scored 3 only 14% of the time, although 83% of their
  submissions passed the base tests.
- So a review cost the reviewer about 0.7 points and gained each follower
  about 0.14.
- Training's advantage was −0.08 for reviewing against +0.05 for submitting.
- Under 0/1 scoring (experiment 2), the same untrained A3B also applied a
  known rule poorly (25% on held-out problems). Training raised that to 65%,
  because there the rule was the only way to score.

See [a sacrifice is learned only when it pays in practice](../concepts/sacrifice-pays-in-practice.md).
Why the A3B treats the rule as optional is a hypothesis to check in
transcripts. [open]

## Regime notes

- **Models:** [Qwen3.8-27B](../entities/qwen3-8-27b.md) (dense, effort
  `medium`) and [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md)
  (mixture of experts, about 3B parameters active per token; the adapter
  leaves the routed experts frozen). One learner is shared by all trained
  seats; experiment 2.1's contributor 1 is a frozen adapter.
- **Budgets:** 12,288 generated tokens per contributor, at most 6,144 per
  call.
- **Not matched across models:** the A3B ran twice the batch (8 repos × 4 per
  step), with an adapter that excludes the routed experts. Per game seen, the
  27B learned faster: 0.267 after about 960 games, against about 2,560 for the
  A3B. [partial]
- **Cost:** about $834 for the whole study (09-25 to 10-05), including the
  eval session.

## What would change the answer

- **A second seed** of the key comparisons (experiment 2 team vs individual;
  experiment 3 team vs experiment 2 A3B team).
- **Experiment 3 with followers who apply the rule.** If informed A3B
  followers treat the rule as optional, a larger bonus, or the rule required
  for the 1 point, tests whether a costly sacrifice is then learned.
- **Experiment 1's regime run longer** (the planned 80 steps), the only regime
  where a costly sacrifice paid.
- **Experiment 2.1 with team reward for followers:** does reviewing appear
  when followers can see that earlier reviews helped?
- **The solo control,** never run.

## History

- **2026-10-01 (experiment 1 only):** ~~Not seen yet. In one 30-step
  team-reward run (Qwen3.8-27B, local, code_rules relay N=4, one seed),
  sacrificing paid off and was rewarded, but its rate fell (10.6% → 7.2%)
  while team score rose by reaching CI more often; other arms untested
  [pilot].~~ Replaced on 2026-10-05 by the answer above, after experiments 2,
  2.1 and 3 and the eval session. Experiment 1's numbers stand. What changed is
  that other regimes now show when the behaviour is and is not learned.
- ~~So far the null result points at learning a rare, cue-driven choice from a
  thin, diluted signal, not at a missing incentive.~~ Still a candidate for
  experiment 1, but experiment 3 shows a second route to a null result: a
  missing *realised* incentive.

## Tensions

- **Is experiment 2 a sacrifice?** The report's summary counts experiment 2
  as "yes": team reward taught an agent "to give up its own CI run". But
  experiment 2's own design notes say that under 0/1 scoring "reviewing no
  longer costs anything for a contributor without the rule", and that the
  question "is no longer about giving up their own score". This page follows
  the design notes. Experiment 2 shows costless information gathering for the
  team, not a sacrifice. [open]
- **Experiment 3 vs experiment 1 is not one factor.** The model, batch,
  adapter, hot-load tolerance and three prompt changes all differ.
  Experiment 3's own README calls the comparison descriptive only. [open]
- **Experiment 3 was paused at step 30 of 80.** Once reviewing is near 0%, few
  followers start with the rule, so followers have little chance to learn to
  apply it. The decline may therefore be self-reinforcing. This is an
  inference, not measured. [open]
- **The 27B's late overshoot.** Redundant reviews climbed from 10% to 24% at
  steps 50–59, so reviewing was starting to become a habit not tied to need.
  Learning had not levelled off at step 59 (27B) or step 79 (A3B). [open]
- **The individual arm cannot reach contributor 1 under 0/1 scoring.**
  Contributor 1 never scores (0 of 480), so under individual reward it gets
  no learning signal at all. The null for contributor 1 in that arm is by
  construction, not evidence against the exposure hypothesis. Experiment 3's
  individual arm, where contributor 1 can score, is a fairer test, and it
  also failed. But there too the exposure signal was thin: only 1% of
  contributors scored 3 at steps 0–9, so followers were rarely rewarded for
  using a note. [partial]
- **The pre-registered gate (from experiment 1).** The source never reports a
  base-rate eval of the untrained model on the training repos (the plan was
  to stop if the first-mover rate was below 3%). Experiment 2 dropped the
  separate gate by design; step 0 measured the base rates instead. [open]
- **Who is counted (from experiment 1).** Experiment 1's sacrifice rate is
  measured among contributors who ran CI, a pool that grew from 359 to 419.
  Experiment 3 has the same issue: reaching CI rose from 65% to 74%. [open]
