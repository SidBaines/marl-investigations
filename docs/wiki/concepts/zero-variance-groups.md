---
type: concept
title: Zero-variance groups at small G
description: "With a group baseline and small groups, many groups have identical rewards, so advantages are zero and the group is dropped; at B=2×G=2 the Tinker RL smoke barely stepped, and in the sacrifice relay's sparse 0/1 scoring 42–49% of G=4 groups carried no signal at first, falling to 8% as team training succeeded, while the individual arm stayed at 40–45% [partial]."
resource: src/marli/train/credit.py
tags: [training, rl, credit-assignment, grpo, group-size, diagnostics, sparse-reward]
timestamp: 2026-10-05
---

# Zero-variance groups at small G

**Mechanism.** Advantages are rewards minus a baseline computed within a group
of G episodes of the same task. When every episode in a group gets the same
reward (all 0 or all 1), every advantage is 0. The credit pipeline drops those
groups (step 5 of the pipeline in `src/marli/train/types.py`;
`drop_zero_variance=True` by default). The loss is sum-reduced, so a
zero-advantage datum contributes no gradient.

**Back-of-envelope (derivation, not a measurement).** Assume episodes are
independent with per-task pass rate p. A group is then zero-variance with
probability pᴳ + (1−p)ᴳ. At p=0.5 that is 0.50 for G=2, 0.125 for G=4 and 0.008
for G=8. It is much higher for tasks the policy almost always solves or almost
never solves.

## Measured in the Tinker RL smoke [pilot]

**Regime:** [Qwen3.5-4B](../entities/qwen3-5-4b.md) with LoRA r=32 on
[Tinker](../entities/tinker.md), lr 1e-5, and an importance-sampling loss. 3
steps, B=2 tasks × G=2 episodes, on [POLARIS-53K](../entities/polaris-53k.md)
(32 shuffled tasks, seed 0). Credit was team reward, a role leave-one-out
baseline and `norm=mean`. Source: [smoke README](../../sources/tinker-rl-smoke.md).

- Most groups had zero reward variance and were correctly skipped.
- `ms_all` ([multi-session notes](../entities/protocol-multi-session.md),
  `segment_credit=all`) **never stepped** in 3 steps.
- `ms_last` (`segment_credit=last`) stepped **once**, and only on last-session
  datums.
- So the smoke **cannot compare credit schemes**. The README says a pilot needs
  B≥8 and G≥4 per step.

## Team reward on the relay: the drop logic checked [partial]

In the sacrifice-relay trial ([Qwen3.8-27B](../entities/qwen3-8-27b.md) on the
[local backend](../entities/local-backend.md), [relay](../entities/protocol-relay.md)
N=4, team reward, 4 repos × G=4, 30 steps, one seed), `payoff.py`
recomputed every advantage from the saved rollouts. It used the same rule:
team score minus the leave-one-out group mean, with zero-variance groups
dropped. The result matched the logged mean |advantage| exactly. This is an
independent check of the credit pipeline under team reward
([experiment 1](../../sources/sacrifice-relay-experiment-1.md)).

- The share of groups dropped was not reported. [open]
- Groups that survive can still carry too little signal for a rare choice.
  There were 3–4 sacrifices per step, each sharing one advantage with every
  token of its playthrough. That is a different way for a behaviour's gradient
  to vanish (see [rewarded but not learned](rewarded-choice-not-learned.md)).
- The study filtered problems to those solved in 1–3 of 4 attempts
  ([DeepCoder](../entities/deepcoder.md)). The source gives no reason, but the
  filter also keeps group rewards varied.

## Sparse 0/1 scoring on the relay: the share of silent groups [partial]

Experiment 2 made the score sparse: 1 only if the tests pass *and* the hidden
rule is followed. That made zero-variance groups the main risk. Rescoring
experiment 1's games under that rule predicted that 59% of groups would tie.
The run had an abort rule for ≥ 75% tied groups over steps 0–5.

The setting: G=4 playthroughs per repo, team or individual reward, the
leave-one-out baseline, one seed. Source:
[study report §3](../../sources/sacrifice-relay-experiments-1-3-and-evals.md);
`experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/README.md`.

| Run | Groups with no learning signal, by 10-step block |
|---|---|
| Exp 2 team, [Qwen3.8-27B](../entities/qwen3-8-27b.md), 4 repos per step | 42%, 18%, 18%, 15%, 10%, 8% (steps 0–59) |
| Exp 2 team, [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md), 8 repos per step | 49% (0–9), 31% (20–29), 12% (40–49), 6% (60–69), 8% (70–79) |
| Exp 2 individual, 27B | 40%, 45%, 40% (steps 0–29) |

- **The share falls when training works.** Once some playthroughs score, the
  ties break, and the signal gets richer as the policy improves. [partial]
- **Under individual reward it does not.** At step 14 every group tied, so
  the run made no update; that arm has 29 adapters for 30 steps. Contributor 1
  never scored (0 of 480), so all of its groups were silent. [partial]
- **The prompt sentence that raised the base review rate** (see
  [rewarded but not learned](rewarded-choice-not-learned.md)) kept the
  early share below the abort line. Under experiment 2's planned prompt,
  steps 0–1 had no learning signal at all, and the learner skipped both.
  [partial]
- For experiment 1 the share of dropped groups is still not reported. [open]

## Bugs this exposed (fixed in M3-5-fix, 70d7cd1)

- Zero-advantage datums were still billed and still stepped learners, for
  example `coord_perrole`'s `work` learner at step 0. Now they are skipped.
- A step where every episode failed passed silently. Now it fails loudly.

## Prior practice (scimt)

scimt's TRL GRPO backend guards against the same failure. It aborts after
`zero_gradient_abort_logs: 3` zero-gradient logs, and has a
`zero_std_warmup_fraction` of 0.10. Its comment notes that zero-gradient
batches "can be legitimate when a mature policy's usable generation groups are
reward-uniform" ([GRPOOptions](../../sources/scimt-grpo-options.md)). marli's
equivalent after M3-5-fix is to skip zero-advantage datums and fail loudly on
all-failed steps. It has no abort on a run of zero-gradient steps. [open]

## Tensions

- The smoke checklist includes "both learners step in `coord_perrole`", and
  the README reports it passed at step 0. But the training-loop review names
  that same `work` step as a zero-advantage step. The check passed partly for
  the wrong reason, so there is no evidence yet that a per-role worker LoRA
  receives a non-zero gradient. The smoke ran before the fix. [open]

## Open questions

- Pick G and task difficulty together. POLARIS-53K records a 7B pass rate per
  task, which could be used to filter out tasks that are almost always or
  almost never solved. [open]

See also: [on-policy check](on-policy-check.md),
[multi-session carry](multi-session-carry.md).
