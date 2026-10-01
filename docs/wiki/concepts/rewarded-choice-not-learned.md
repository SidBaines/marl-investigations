---
type: concept
title: "Rewarded but not learned: a rare choice RL did not pick up"
description: "In the sacrifice-relay trial, the costly choice (reveal the hidden rule for later contributors) paid off for the team and got a positive advantage from training, yet it became rarer over 30 steps; a real incentive and a correctly signed gradient were not enough for a rare choice whose tokens are a sliver of each update [partial]."
resource: experiments/2026-09-25_sacrifice-relay/payoff.py
tags: [training, rl, credit-assignment, sparse-signal, multi-agent, relay, sacrifice, team-reward]
timestamp: 2026-10-01
---

# Rewarded but not learned: a rare choice RL did not pick up

**The phenomenon.** A choice can (a) pay off under the reward, and (b) get a
positive advantage from training every time it is taken, and still (c) become
*less* common under RL. So a null result for "did the policy learn X?" does
not show that X was unrewarded. Check (a) and (b) separately before blaming
the incentive. [partial]

Two terms used below:

- **Advantage**: the weight training puts on a sampled action. Positive means
  "make this more likely", negative means "less likely". In this run it is a
  playthrough's team score minus the mean team score of the other
  playthroughs of the same repo in the same step (a *leave-one-out group
  baseline*), with no further scaling (`norm=mean`).
- **Playthrough / group**: one episode of a repo, with 4 contributors working
  in turn. A group is the G=4 playthroughs of one repo in one training step.

## The case: sacrificing in the code_rules relay

**Regime.** [Qwen3.8-27B](../entities/qwen3-8-27b.md) with LoRA r=32 on the
[local backend](../entities/local-backend.md). One learner is shared by all
contributors. Flat lr 4e-5 (twice the pre-registered value), an
importance-sampling loss (sum-reduced), team reward, the leave-one-out group
baseline, `norm=mean` and token-sum aggregation. The environment is
[code_rules](../entities/env-code-rules.md) on the
[relay protocol](../entities/protocol-relay.md): N=4, bonus m=3, one CI run per
contributor, shared NOTES.md visible. Each step has 4 repos × G=4 = 16
playthroughs (64 contributor decisions). 30 steps, one seed. The 51 repos are
built from 207 filtered [DeepCoder](../entities/deepcoder.md) problems.
Budgets: 12,288 generated tokens per contributor, at most 6,144 per call,
context at most 24,576. Source:
[experiment 1](../../sources/sacrifice-relay-experiment-1.md).

**The choice.** Each contributor has one CI run. `ci_submit` scores its own
solution (0, 1, or 3 if it also follows the hidden house rule). `ci_review`
reveals the rule but scores the contributor 0. This is the *sacrifice*. Later
contributors who learn the rule (in practice, from NOTES.md) can score 3.

**(a) It paid off for the team.** The comparison is within a group: the team
score of playthroughs with a sacrifice minus the same group's playthroughs
without one, so problem difficulty cancels. Brackets are 95% bootstrap
intervals. [partial]

| First sacrifice by | Team-score difference | Groups |
|---|---|---|
| any contributor | +0.31 [+0.21, +0.42] | 72 |
| contributor 1 | +0.62 [+0.30, +0.97] | 11 |
| contributor 2 | +0.46 [+0.30, +0.60] | 40 |
| contributor 3 | +0.12 [+0.01, +0.24] | 26 |
| contributor 4 (can help no one) | −0.13 [−0.22, −0.04] | 10 |

The sacrificer loses 0.5–0.75 points and each later contributor gains 0.9–1.4.
That matches a back-of-envelope estimate from observed rates (+0.74, +0.49,
+0.20, −0.11 by position).

**(b) Training rewarded it.** The advantages were recomputed from the saved
rollouts, and they match the logged mean |advantage| exactly. Among
contributors who started without the rule and ran CI, the mean advantage was:
[partial]

| | Reviewed (sacrificed) | Submitted |
|---|---|---|
| contributor 1 | +0.57 [+0.24, +0.94] (n=11) | +0.02 |
| contributor 2 | +0.35 [+0.19, +0.53] (n=51) | −0.02 |
| contributor 3 | +0.06 [−0.05, +0.16] (n=32) | −0.03 |
| contributor 4 | −0.12 [−0.20, −0.03] (n=10) | −0.02 |

The signs match the payoffs in (a): training pushed toward reviewing for
contributors 1–2, was near zero for 3, and pushed against it for 4.

**(c) The behaviour moved the other way.**

- Among contributors with a real choice (no rule at the start, ran CI), the
  sacrifice rate fell from 10.6% ± 3.2% (38/359, steps 0–9) to 7.2% ± 2.5%
  (30/419, steps 20–29). That is −3.4 points, z=−1.7, meaning the change is 1.7
  standard errors below zero. It is suggestive, not conclusive. [partial]
- After an earlier contributor's note about the extended checks, the review
  rate fell from 20% (steps 0–9) to 18% (10–19) to 11% (20–29). This split was
  chosen after looking at the data. [pilot]
- Over the same steps, team score rose +0.086 (z=2.3). Most of that came from
  contributors reaching CI more often (see
  [token budget binding](token-budget-binding.md)). [partial]

**What this rules out.** The training machinery delivered the update it
computed: `kl_sample_train` stayed at or below 7.5e-4 and the mean IS ratio at
1.0000 ± 0.0001 at every step (see [on-policy check](on-policy-check.md)). So
the gap is between a correct gradient and the behaviour it produced, not a
sampler/trainer mismatch. [partial]

## Candidate explanations (all untested) [open]

These are from the source:

1. **The signal is thin.** There were 3–4 reviews per step, and only 11 by
   contributor 1 in the whole run.
2. **The signal is diluted.** The choice is a few tokens among about 550k
   action tokens per step. Under team reward, one advantage covers every
   token of every contributor in a playthrough. One step's gradient mixes 16
   playthroughs.
3. **Spill-over from the dominant learned behaviour.** RL's main gain was
   reaching CI and submitting sooner. The same update may make reviewing less
   likely.

## Caveats

- **Observational, not randomized.** Who sacrifices is not random. Almost all
  reviews by contributors 2–4 follow an earlier contributor's note (see
  [reactive information gathering](reactive-information-sharing.md)), so
  playthroughs with a sacrifice already differed before it happened.
  `payoff.py` itself flags this selection for earlier positions. The
  within-group comparison cancels problem difficulty, not this.
- One seed and one arm. No individual-reward or solo control was run, and only
  30 of the 80 planned steps.
- The 51 repos repeat (each drawn about 2.4 times over 30 steps). Only the rule
  is resampled.

## Tensions

- The pre-registered readout compared steps 0–9 with steps 70–79. The trial
  stopped at 30 steps, so the source compares 0–9 with 20–29. "Not supported so
  far" is not the same as "refuted". [open]
- The denominator for the sacrifice rate conditions on reaching CI, and that
  share grew (61.4% → 70.2%). Contributors who newly reach CI may differ from
  the rest, so part of the fall could be a change in who is counted. This
  confound was not checked. [open]

## Open questions

- Which explanation is it? Run longer (the planned 80 steps). Get more choice
  events per step (larger B or G). Credit the decision rather than the whole
  playthrough. Try a per-slot LoRA. The source also lists hidden-notes and
  oracle-notes ablations, m and N sweeps, and a swarm variant. [open]
- The next experiment changes the setup (no position in the prompt, 0/1
  scoring), so it will not be a direct replication. [open]

See also: [the synthesis](../syntheses/does-rl-teach-sacrifice.md),
[zero-variance groups](zero-variance-groups.md) (another way a behaviour's
gradient can vanish).
