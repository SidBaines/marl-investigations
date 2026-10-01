---
type: synthesis
title: Does RL teach agents to give up their own reward so later agents score more?
description: "Current answer: not seen yet. In one 30-step team-reward run (Qwen3.8-27B, local backend, code_rules relay N=4, one seed), sacrificing paid off for the team and training's advantage favoured it, but the sacrifice rate fell (10.6% → 7.2%, z=−1.7) while team score rose by reaching CI more often; the individual-reward and solo arms are untested [pilot]."
resource: docs/sources/sacrifice-relay-experiment-1.md
tags: [synthesis, rl, multi-agent, sacrifice, credit-assignment, team-reward, relay, code-rules]
timestamp: 2026-10-01
---

# Does RL teach agents to give up their own reward so later agents score more?

## Current answer [pilot]

**Not seen yet, in the one regime run.** The setup made sacrifice worthwhile,
and training rewarded it, but the policy did not move toward it in 30 steps.

1. **The incentive is real.** Playthroughs with a sacrifice scored +0.31
   [+0.21, +0.42] higher in team score than the same repo's playthroughs
   without one (72 groups). The gain was largest when contributor 1 sacrificed
   (+0.62). [partial]
2. **The hand-off works.** 94% of sacrifices by contributors 1–3 passed the
   rule to the next contributor, and informed followers who submitted scored
   the bonus about 82% of the time. [partial]
3. **Training pushed the right way.** The advantage for reviewing was +0.57
   (contributor 1) and +0.35 (contributor 2), against about 0 for submitting.
   [partial]
4. **But the behaviour moved the other way.** The sacrifice rate among
   contributors with a real choice fell from 10.6% to 7.2% (z=−1.7). Team
   score rose (+0.086, z=2.3) for another reason: more contributors reached CI.
   [partial]
5. **Gathering is reactive.** Contributors review almost only after an earlier
   contributor's note reports failed extended checks. Contributor 1, whose
   review is worth the most, reviews 2–4% of the time. [partial]

So far the null result points at **learning a rare, cue-driven choice from a
thin, diluted signal**, not at a missing incentive. Sid's individual-reward
hypothesis (that exposure to others' revealed rules would raise sacrifice even
under individual reward) and the solo control are **untested**. [open]

## Regime

- **Model and backend:** [Qwen3.8-27B](../entities/qwen3-8-27b.md) (renderer
  `qwen3_8_medium`, T=1), LoRA r=32 on the
  [local backend](../entities/local-backend.md), 2×H200 with the
  [time-shared layout](../concepts/gpu-time-sharing.md) and
  [MTP sampling](../concepts/speculative-decoding-mtp.md). One learner shared
  by all contributors.
- **Environment and protocol:** [code_rules](../entities/env-code-rules.md) on
  [relay_n4](../entities/protocol-relay.md): N=4, m=3, one CI run per
  contributor, NOTES.md visible, 51 repos from 207 filtered
  [DeepCoder](../entities/deepcoder.md) problems.
- **Training:**
  - team reward (every contributor gets the repo's mean score);
  - leave-one-out group baseline, `norm=mean`, token-sum aggregation,
    importance-sampling loss;
  - flat lr 4e-5 (twice the pre-registered 2e-5);
  - 4 repos × G=4 = 16 playthroughs and 64 decisions per step;
  - 30 of the 80 planned steps, one seed.
- **Budgets:** 12,288 generated tokens per contributor, at most 6,144 per call,
  context at most 24,576.
- **Statistics:** ± is a 95% normal interval; brackets are 95% bootstrap
  intervals over groups; z is the change divided by its standard error. Early
  means steps 0–9 and late means steps 20–29 (160 relays and 640 contributors
  each).
- **Cost:** about $153 for the whole 2×H200 pod session, including the
  benchmark and filter.
- **Sources:** [experiment 1](../../sources/sacrifice-relay-experiment-1.md)
  and the [benchmark](../../sources/sacrifice-relay-throughput-bench.md).

## Pre-registered hypotheses: status

| Hypothesis | Status |
|---|---|
| H-team: in `relay_team`, the first-mover probe rate rises, with team return | **Not supported so far.** The rate fell in 30 of 80 steps; team return rose for another reason. [partial] |
| H-individual (Sid's): the rate also rises under individual reward, via exposure | **Untested** (arm not run). [open] |
| Control: in `solo`, the probe rate falls toward 0 | **Untested** (arm not run). [open] |
| Understanding vs habit: hand-off fidelity rises, redundant probing falls | Hand-off was already high (94%) [partial]; the trend was not reported. [open] |

The "first-mover probe rate" is the share of contributors who did not know the
rule when they started, ran CI, and chose `ci_review`.

## Why it might not have been learned

See [rewarded but not learned](../concepts/rewarded-choice-not-learned.md).
These are candidates from the source, all untested. [open]

- **Thin signal:** 3–4 reviews per step, 11 by contributor 1 in the whole run.
- **Dilution:** the choice is a few tokens among about 550k action tokens per
  step, and one team advantage covers all of them.
- **Spill-over:** the dominant learned behaviour is to reach CI and submit
  sooner ([token budget binding](../concepts/token-budget-binding.md)).
- **Ruled out:** a sampler/trainer mismatch. `kl_sample_train` stayed at or
  below 7.5e-4 every step ([on-policy check](../concepts/on-policy-check.md)).

## What would change the answer

- **Run the missing arms.** These are individual reward and solo, plus seeds,
  ideally to the planned 80 steps.
- **More choice events per step,** or credit aimed at the decision rather than
  the whole playthrough.
- **Ablations listed in the source:** hidden notes, oracle notes, a per-slot
  LoRA, m and N sweeps, and a swarm variant.
- **The next experiment** removes the position from the prompt and uses 0/1
  scoring. Its result will not be a direct replication of this one.

## Tensions

- **The pre-registered gate.** The plan was a base-rate eval of the untrained
  model on the training repos, stopping if the first-mover probe rate was
  below 3%. The source does not report it. The nearest numbers are about 8%
  (3/38) in 16 untrained relays on unfiltered problems, and 10.6% in training
  steps 0–9. Both clear the 3% bar. [open]
- **Who is counted.** The sacrifice rate is measured among contributors who
  ran CI, a pool that grew from 359 to 419. A change in who is counted could
  explain part of the fall. Not checked. [open]
