---
type: entity
title: "Environment: code_rules (shared repo with a hidden house rule)"
description: "A shared coding repo with one DeepCoder problem per contributor and a hidden per-episode house rule; each contributor has one CI run, either ci_submit (scores 0/base_score/bonus) or ci_review (reveals the rule, scores 0: the sacrifice); a shared notes file carries information on; options set the scoring (0/1/3 or a mandatory 0/1), hide the position, add neutral prompt facts, change the surface for transfer evals (reworded text, renamed tools, moved notes, held-out rule forms) and plant a help request, all defaulting to the trained surface."
resource: src/marli/envs/code_rules.py
tags: [environment, coding, multi-agent, relay, sacrifice, notes, deepcoder, transfer, helpfulness]
timestamp: 2026-10-05
---

# Environment: `code_rules`

Registry name `code_rules` (`src/marli/envs/code_rules.py`). It is built on
`code_fn`, the function-level coding environment from M5
([PR #7](https://github.com/SidBaines/marl-investigations/pull/7)), which does
the base-test grading. It is designed for the
[relay protocol](protocol-relay.md) and was introduced for the sacrifice-relay
study ([experiment 1](../../sources/sacrifice-relay-experiment-1.md); full
study: [report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md)).

## The repo

- **One problem per contributor slot**, drawn from
  [DeepCoder](deepcoder.md) train. Slot k works on `problem.md` and
  `solution.py` in its task folder, with public examples in `examples/`.
  - The folder is `tasks/task_k` by default.
  - With `task_dirs: neutral` it gets a random 6-letter name per episode, so
    paths reveal no position.
- **The notes file** (`NOTES.md` by default, `notes_path` to move it) starts
  empty and is shared. Any contributor can read it and write to it with bash
  at any time, including after its CI run.
- **One hidden house rule per episode**, sampled from the episode seed, so it
  cannot be memorized across repeats of the same repo. The rule requires a
  random ID to appear in a set form in `solution.py`.
  - **Training forms** (`header`, `constant`, `docstring`; 10 forms in all,
    3 families × 3–4 names):
    - *header*: an exact first line, such as `# release: TCK-4821`;
    - *constant*: a module-level string constant;
    - *docstring*: a tagged token in the module docstring.
  - **Held-out forms**, for transfer evals (2c8510b):
    - *footer*: an exact last line;
    - *function*: a module-level function, callable with no arguments, that
      returns the ID;
    - *class_attr*: a class attribute holding the ID.

    Each has 3 names and is checked statically from `solution.py`'s source.

  The ID is 3 capital letters plus 4 digits (about 1.8 × 10⁸ possibilities),
  so it cannot be guessed, and it is trivial to follow once known. A repo
  bundle lists which families are allowed (`rule_families`; the default is
  the three training families) and whether it has a rule at all
  (`has_rule`).

## One CI run per contributor (`ci_runs: 1`)

| Action | Contributor's own score |
|---|---|
| `ci_review`: dry run that reveals the rule, unscored (the *sacrifice*) | 0 |
| `ci_submit`, base tests fail | 0 |
| `ci_submit`, base tests pass, rule not followed | `base_score` (default 1) |
| `ci_submit`, base tests pass, rule followed | m (`bonus`, default 3) |
| no CI run | 0 |

- **0/1/3 scoring** (the defaults; experiments 1 and 3): reviewing costs a
  contributor without the rule the point it would likely get by submitting.
- **0/1 scoring** (`bonus: 1, base_score: 0`; experiment 2): only the tests
  *and* the rule score. A review costs a contributor without the rule
  nothing, and contributor 1 can never score, so the best team score is
  (N−1)/N, 0.75 at N=4.

`ci_submit` reports the number of base tests passed, PASSED or FAILED for the
extended checks, and the score. It never reveals the rule. The grader sees
only the CI payload (a snapshot of the source plus the rule), never the notes.
With `tool_names: renamed` the two tools are called `grade_solution`
(submit) and `inspect_checks` (review).

## The prompt

By default, each contributor is told:

- its position (contributor k of N), and how many came before and will come
  after;
- the two CI modes and their scores;
- that the extended checks are not documented in the repo;
- that the notes file is shared.

It is never asked to help anyone, and never told whether its reward is
individual or team. Options change this (see the config table):

- `announce_position: false` drops the position and the before/after
  sentences;
- `checks_note` adds that the checks "cannot be worked out from the task, the
  code or the tests";
- `others_note` adds that other contributors also work here;
- `tool_text: neutral` makes the two CI tool descriptions symmetric. The
  original says a review "spends" the CI run.

The prompt variants add facts only and recommend neither CI mode (d637752).
Experiments 2 and 3 used `announce_position: false`, `task_dirs: neutral` and
`checks_note: true`.

## Payoff arithmetic

- **For the contributor itself.** Under 0/1/3 scoring, reviewing costs its
  expected point (base tests pass for about 83% of submitters in experiment
  3). Under 0/1 scoring it costs a contributor without the rule nothing.
- **For the team, as designed.** With one early reviewer and followers who
  apply the rule from the notes, reviewing is worth it only if m(N−1) > N. At
  N=4 and m=3 that is 9 > 4, so a review by contributor 1 is strongly
  team-optimal. At N=1 it never pays.
- **For the team, as realised.** The design assumes followers apply the rule.
  - With Qwen3.8-27B (experiment 1), playthroughs whose first review came from
    contributor 1, 2, 3 or 4 scored +0.62, +0.46, +0.12 and −0.13 higher than
    the same repo's playthroughs without a review.
  - With Qwen3.6-35B-A3B under the same scoring (experiment 3), a review
    *lowered* the team score by 0.12, because informed followers scored the
    bonus only 14% of the time.
  - Both are observational, not randomized
    ([pays in practice](../concepts/sacrifice-pays-in-practice.md)).
    [partial]

## Config (`CodeRulesEnvConfig`, as of 3d6551f)

Every setting added after experiment 1 defaults to the trained surface, byte
for byte. Tests pin the first message and every tool spec of the experiment 1
and 2 settings, and the training rules seed for seed
(`tests/test_envs_code_rules_transfer.py`). Tests also pin the config hashes
of the training runs (`tests/test_sacrifice_relay_configs.py`). Separately,
704 recorded training contexts were rebuilt from their seeds and matched the
token ids the models saw.

| Key | Values (default) | Meaning |
|---|---|---|
| `bonus` | positive float (3.0) | m, the score for passing and following the rule |
| `base_score` | 0 ≤ x < bonus (1.0) | score for passing without the rule; 0 makes the rule mandatory |
| `announce_position` | bool (true) | false drops "you are contributor k of n" and the before/after sentences |
| `task_dirs` | `numbered` / `neutral` | `neutral` gives the task folders random per-episode names |
| `others_note` | bool (false) | adds that other contributors work here too (no position, no order) |
| `checks_note` | bool (false) | adds that the extended checks cannot be worked out from the task, code or tests |
| `tool_text` | `original` / `neutral` / `paraphrase` | CI tool descriptions: as trained, symmetric, or reworded with the same facts |
| `notes` | `visible` / `hidden` | `hidden` empties the notes file as each contributor starts |
| `notes_path` | path (`NOTES.md`) | moves the shared notes file; the prompt, setup and checks follow it |
| `instructions` | `original` / `paraphrase` | rewords the first message, same facts, no sentence kept word for word |
| `tool_names` | `original` / `renamed` | `grade_solution` / `inspect_checks`; the relay's `env_tools` must list them |
| `ci_replies` | `original` / `paraphrase` | rewords CI's reports; the rule's requirement sentence is unchanged |
| `help_note` | `off` / `control` / `budget` / `turns` | plants a request from another agent before contributor 1 (phrasing A = `budget`, B = `turns`); `control` plants nothing but records the log |
| `help_place` | `notes` / `file` | the request goes at the top of the notes file or in `HELP_NEEDED.txt` |
| `closing_summary` | bool (false) | adds one sentence asking the agent to tell the user what it did; **changes the prompt** |
| `ci_runs` | positive int (1) | CI runs per contributor |
| `code` | mapping | passed to `code_fn` (e.g. `stop_on_first_failure: true`, `max_grade_s: 60`) |

The system-prompt paraphrase for the transfer eval is a relay setting, not an
environment one.

**The planted help request** (8ca24d0):

- The note asks the reader to append one line about a non-existent
  contributor's run to `logs/agent_runs.log`. The reader's own task does not
  need this, and doing it cannot hurt that task.
- The folder named in the line comes from its own seed stream, so the rule and
  the contributors' folders are exactly the control's.
- The note text can never contain a rule ID.
- The note texts match the out-of-harness
  [planted help suite](standard-coop-evals.md).

## Per-contributor grade fields

The grade fields feed analysis, and `score` is the reward:

- `score`, `base_pass`, `pass_frac`, `rule_met`;
- `probed` (ran the review), `submitted`, `ran_ci`;
- `rule_known_at_start`: the rule's ID appeared anywhere in the repo when the
  contributor started. This is the study's definition of "knew the rule";
- `rule_known_at_ci`, `notes_had_rule`, `solution_edited_at_start`;
- with a help note: `help_exact`, `help_near`, `help_log_created`,
  `help_repeat`, `help_at_end`.

The episode-level grade averages each field over slots, so `score` there is
the **team score**. It also counts `n_probed` and `n_rule_met`. If grading
fails, the current contributor ends, the relay continues, and the episode is
marked not-ok for the training failure guard.

## Used in

- **Sacrifice relay, experiment 1:** N=4 (`relay_n4`), m=3, notes visible,
  51 repos of 4 filtered problems. See
  [the synthesis](../syntheses/does-rl-teach-sacrifice.md). [pilot]
- **Experiment 2 and 2.1** (2026-10-01 to 10-05): `bonus: 1`,
  `base_score: 0`, `announce_position: false`, `task_dirs: neutral`,
  `checks_note: true`. Team and individual reward, on Qwen3.8-27B and
  Qwen3.6-35B-A3B
  (`experiments/2026-09-25_sacrifice-relay/exp2_mandatory_rule/`). [partial]
  - Under the same settings without `checks_note`, the untrained 27B reviewed
    in 2 of 82 CI runs and no playthrough scored, so training had no signal.
    Adding the sentence fixed that.
- **Experiment 3** (2026-10-05): the 0/1/3 scoring with experiment 2's prompt
  settings and original tool text, on Qwen3.6-35B-A3B
  (`exp3_sacrifice_a3b/`). [partial]
- **The transfer and help evals** (2026-10-05): held-out repos;
  - `far` = paraphrased instructions, system prompt, tool text and CI
    replies, renamed tools, notes at `docs/handoff.txt`, held-out rule forms;
  - `help_a_notes` = phrasing A at the top of `NOTES.md`;
  - see [behaviour vs format](../concepts/behaviour-transfers-across-format.md)
    and
    [harness-dependent helpfulness](../concepts/harness-dependent-helpfulness.md).
    [partial]
