---
type: concept
title: Zero-variance groups at small G
description: With a group baseline and small groups, most groups have identical rewards, so advantages are zero and the group is dropped; at B=2×G=2 the Tinker RL smoke's multi-session runs barely stepped and could not compare credit schemes.
resource: src/marli/train/credit.py
tags: [training, rl, credit-assignment, grpo, group-size, diagnostics]
timestamp: 2026-09-24
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
