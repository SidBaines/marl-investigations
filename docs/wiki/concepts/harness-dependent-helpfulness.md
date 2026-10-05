---
type: concept
title: "Agent helpfulness depends on the harness"
description: "The same model and the same note asking it to append a line for another agent gave very different results in two harnesses: Qwen3.8-27B did it for 19–20% of contributors (36–38% of first contributors) inside the code_rules relay, where the note sat in the shared NOTES.md it is told to use, but 0 of 120 times in a stock Inspect agent, though it read and reported the note most of the time; Qwen3.6-35B-A3B almost never helped in either; team training changed neither [partial]."
resource: experiments/2026-10-05_coop-evals/README.md
tags: [evaluation, helpfulness, cooperation, harness, multi-agent, inspect, relay, notes, generalisation]
timestamp: 2026-10-05
---

# Agent helpfulness depends on the harness

**The phenomenon.** Whether an agent acts on a request from another agent is
not a fixed trait of the model. With the same request text, the same model
helped often in one harness and never in another. So a single helpfulness
number, measured in one setting, does not describe the model. Compare
within one harness, as this wiki's schema already requires for scores.
[partial]

## The planted help request

A note from "another agent" asks the reader to append one line about that
agent's run to `logs/agent_runs.log`. The reader's own task does not need
this, and doing it cannot hurt that task. There are two phrasings, A and B.
The note texts are identical in both harnesses. Measures:

- **helped:** the requested line appears in the log;
- **read the note:** a command opened the note's file, or a tool result showed
  the request;
- **told:** the agent mentioned the note or the log in text visible to the
  user, by a fixed keyword rule. A no-note control gives the rule's
  false-alarm rate.

## Results [partial]

**Inside the relay** ([code_rules](../entities/env-code-rules.md), phrasing A
at the top of `NOTES.md`):

- held-out repos, 50 games, so 200 contributors per cell;
- same game settings as the [transfer eval](behaviour-transfers-across-format.md).

| | 27B untrained | 27B team | A3B untrained | A3B team |
|---|---|---|---|---|
| Appended the requested line | 19% (38/200) | 20% (40/200) | 0% | 0.5% (1/200) |
| ... of first contributors | 36% | 38% | 0% | 0% |
| Saw the request in a tool result | 98% | 97% | 90% | 98% |
| Mentioned it in visible text | 42% | 39% | 0% | 2% |

**In a standard harness** ([planted help suite](../entities/standard-coop-evals.md)):

- agent: Inspect's stock `react()` agent with bash and submit;
- tasks: one of 30 small original coding tasks;
- note conditions: phrasing A or B, in `NOTES.md` or `HELP_NEEDED.txt`, so
  120 episodes per policy (plus 30 no-note controls).

| | 27B untrained → team | A3B untrained → team |
|---|---|---|
| Helped | 0/120 → 0/120 | 0/120 → 0/120 |
| Read the note (range over the 4 note conditions) | 60–83% → 63–90% | 23–43% → 23–47% |
| Told the user | 53–80% → 63–90% | 0% → 0% |
| "Told" in the no-note control (false alarms) | 10% → 7% | 0% → 0% |

Models: [Qwen3.8-27B](../entities/qwen3-8-27b.md) and
[Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md), untrained and after
experiment 2's team training (one training seed). Sources:
[study report §4.4](../../sources/sacrifice-relay-experiments-1-3-and-evals.md);
aggregate tables in `experiments/2026-09-25_sacrifice-relay/exp2_eval/results/`
and `experiments/2026-10-05_coop-evals/results/` (in git at 3d6551f).

- **The 27B helps inside the relay and never outside it.** In the standard
  harness it usually reads the note and tells the user about it, but does not
  do the task. [partial]
- **The A3B hardly helps anywhere.** In the standard harness it mostly does
  not look at the note. [partial]
- **Team training changed neither model's helping in either harness.**
  [partial]

## What differs between the two settings [open]

The report attributes the gap to the harness. "Harness" here bundles
several differences, and none has been varied alone:

- **How shared files are framed.** The relay prompt says `NOTES.md` is shared
  by everyone working in the repository and may be read and written at any
  time. The Inspect agent gets one neutral user message.
- **Where the note sits.** In the relay it is in the file the agent is told to
  use, and almost every contributor saw it. In the Inspect harness, read rates
  were 23–90%.
- **The main task.** Relay contributors work on hard, filtered DeepCoder
  problems under a 12,288-token budget, and many never reach CI. The Inspect
  tasks are small: every policy passed the hidden tests almost every time, in
  about 500–1,300 output tokens per episode.
- **The reporting channel.** The Inspect agent's submission is its report to
  the user. The relay has no user report.
- **Sampling and prompting.** The relay uses training's sampling (T=1, no
  truncation) and the training renderer. The Inspect harness uses the chat
  endpoint with model-card sampling (top_p 0.95, top_k 20; A3B presence
  penalty 1.5). The chat template was checked to match the renderer token for
  token on game prompts.

## Cost of the note inside the relay [partial]

The note was not free for the team-trained policies. Compare `help_a_notes`
with the no-note control, using the same policy, repos and seeds (from the
committed help and transfer reports):

- **27B team:**
  - contributors scored 8.5 points less often [−16.5, 0.0];
  - they generated 849 more tokens each [+358, +1,351];
  - the team score fell by 0.085 [−0.165, 0.000];
  - contributor 1 reviewed 19 points less often [−35, −5].
- **A3B team:** scored 8.0 points less often [−16.5, +1.0], and reviewed
  9.5 points more often [+3.5, +15.5].
- **Untrained models:** no clear cost.

The report does not discuss this cost.

## Reading

- **Helpfulness evals need the harness held fixed.** A model's rate of acting
  on another agent's request is a property of (model, harness, framing). This
  is the case for testing generality in standard harnesses, and against
  reading one harness's number as a trait. [partial]
- **Reporting is not helping.** The 27B told the user about the request most
  of the time without doing it. A "told the user" measure and a "did it"
  measure answer different questions. [partial]

## Tensions

- **Keyword measures.** "Told" is a keyword rule, not a judge. In the
  standard harness its false-alarm rate on the 27B is 7–10%. Inside the relay
  it is 2% (5 of 200 untrained-27B contributors in the no-note games). The two
  rates are not comparable across harnesses. [open]
- **Small cells.** Each standard-harness condition is 30 episodes, so a 0/30
  cell has a 95% upper bound of about 11%. The pooled 0/120 is firmer. [partial]

## Open questions

- A direct test: the same note and task in both harnesses, varying only where
  the note sits and how the instructions frame shared files. [open]

See also: [standard cooperation evals](../entities/standard-coop-evals.md),
[reactive information gathering](reactive-information-sharing.md),
[the synthesis](../syntheses/does-rl-teach-sacrifice.md).
