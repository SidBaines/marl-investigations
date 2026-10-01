---
type: entity
title: "Environment: code_rules (shared repo with a hidden house rule)"
description: "A shared coding repo with one DeepCoder problem per contributor and a hidden per-episode house rule; each contributor has one CI run, either ci_submit (scores 0/1/m) or ci_review (reveals the rule, scores 0: the sacrifice); a shared NOTES.md carries information to later contributors."
resource: src/marli/envs/code_rules.py
tags: [environment, coding, multi-agent, relay, sacrifice, notes, deepcoder]
timestamp: 2026-10-01
---

# Environment: `code_rules`

Registry name `code_rules` (`src/marli/envs/code_rules.py`). It is built on
`code_fn`, the function-level coding environment from M5
([PR #7](https://github.com/SidBaines/marl-investigations/pull/7)), which does
the base-test grading. It is designed for the
[relay protocol](protocol-relay.md) and was introduced for the sacrifice-relay
study ([experiment 1](../../sources/sacrifice-relay-experiment-1.md)).

## The repo

- **One problem per contributor slot**, drawn from
  [DeepCoder](deepcoder.md) train. Slot k works on
  `tasks/task_k/problem.md` and `tasks/task_k/solution.py`, with public
  examples in `tasks/task_k/examples/`.
- **`NOTES.md`** starts empty and is shared. Any contributor can read it and
  write to it with bash at any time, including after its CI run.
- **One hidden house rule per episode**, sampled from the episode seed, so it
  cannot be memorized across repeats of the same repo. The rule requires a
  random ID to appear in a set form in `solution.py`. There are three families:
  - *header*: an exact first line, such as `# release: TCK-4821`;
  - *constant*: a module-level string constant;
  - *docstring*: a tagged token in the module docstring.

  The ID is 3 capital letters plus 4 digits (about 1.8 × 10⁸ possibilities),
  so it cannot be guessed, and it is trivial to follow once known. A repo
  bundle lists which families are allowed (`rule_families`) and whether it has
  a rule at all (`has_rule`).

## One CI run per contributor (`ci_runs: 1`)

| Action | Contributor's own score |
|---|---|
| `ci_review`: dry run that reveals the rule, unscored (the *sacrifice*) | 0 |
| `ci_submit`, base tests fail | 0 |
| `ci_submit`, base tests pass, rule not followed | 1 |
| `ci_submit`, base tests pass, rule followed | m (`bonus`, default 3) |
| no CI run | 0 |

`ci_submit` reports the number of base tests passed, PASSED or FAILED for the
extended checks, and the score. It never reveals the rule. The grader sees
only the CI payload (a snapshot of the source plus the rule), never the notes.

## The prompt

Each contributor is told its position (contributor k of N), how many
contributors came before and will come after, the two CI modes and their
scores, that the extended checks are not documented in the repo, and that
NOTES.md is shared. It is never asked to help anyone, and never told whether
its reward is individual or team.

## Payoff arithmetic (from the source)

- For the contributor itself, reviewing always costs its expected pass
  (p ≈ 0.5 on the filtered problems).
- For the team, with one early reviewer and followers who apply the rule from
  the notes, reviewing is worth it only if m(N−1) > N. At N=4 and m=3 that is
  9 > 4, so a review by contributor 1 is strongly team-optimal. At N=1 it never
  pays.
- In the trial, playthroughs whose first review came from contributor 1, 2, 3
  or 4 scored +0.62, +0.46, +0.12 and −0.13 higher than the same repo's
  playthroughs without a review. This is observational, not randomized
  ([rewarded but not learned](../concepts/rewarded-choice-not-learned.md)).
  [partial]

## Config (`CodeRulesEnvConfig`, as of 0bd4647)

| Key | Values | Meaning |
|---|---|---|
| `bonus` | positive float (3.0) | m, the score for passing and following the rule |
| `notes` | `visible` / `hidden` | `hidden` empties NOTES.md as each contributor starts |
| `ci_runs` | positive int (1) | CI runs per contributor |
| `code` | mapping | passed to `code_fn` (e.g. `stop_on_first_failure: true`, `max_grade_s: 60`) |

## Per-contributor grade fields

The grade fields feed analysis, and `score` is the reward:

- `score`, `base_pass`, `pass_frac`, `rule_met`;
- `probed` (ran `ci_review`), `submitted`, `ran_ci`;
- `rule_known_at_start`: the rule's ID appeared anywhere in the repo when the
  contributor started. This is the study's definition of "knew the rule";
- `rule_known_at_ci`, `notes_had_rule`, `solution_edited_at_start`.

The episode-level grade averages each field over slots, so `score` there is
the **team score**. It also counts `n_probed` and `n_rule_met`. If grading
fails, the current contributor ends, the relay continues, and the episode is
marked not-ok for the training failure guard.

## Used in

- Sacrifice relay, experiment 1: N=4 (`relay_n4`), m=3, notes visible, 51
  repos of 4 filtered problems. See
  [the synthesis](../syntheses/does-rl-teach-sacrifice.md). [pilot]
