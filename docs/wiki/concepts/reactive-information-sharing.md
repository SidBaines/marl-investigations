---
type: concept
title: "Reactive vs proactive information gathering between agents"
description: "Untrained, agents in the code_rules relay gather information reactively: Qwen3.8-27B reveals the hidden rule almost only after an earlier contributor's note reports failed checks (90 of 93 reviews by contributors 2–4), and the first contributor rarely reviews (2–4%); team RL can make it proactive: in experiment 2 the first contributor's review rate rose 17% → 80% (27B) and 38% → 77% (A3B), and held on new problems (93% / 76%); passing the rule on stays reliable (94–100%) [partial]."
resource: experiments/2026-09-25_sacrifice-relay/payoff.py
tags: [multi-agent, information-sharing, notes, relay, sacrifice, behaviour, code-rules, team-reward]
timestamp: 2026-10-05
---

# Reactive vs proactive information gathering between agents

**Definitions.** An agent gathers information **proactively** when it pays for
the information before any sign that it is needed, because others will
benefit. It gathers **reactively** when it pays only after a cue, such as
another agent reporting a problem. *Sharing* is the separate step of writing
down what was learned for later agents.

**The finding.** In the [code_rules relay](../entities/env-code-rules.md),
gathering is reactive: contributors pay to reveal the hidden rule almost only
after an earlier contributor's note says the extended checks failed. Sharing,
once they have paid, is reliable. The team-optimal policy is close to the
opposite: the *first* contributor should review, because its review helps the
most people. [partial]

**Regime.** [Qwen3.8-27B](../entities/qwen3-8-27b.md) (renderer reasoning
effort `medium`), LoRA r=32 on the [local backend](../entities/local-backend.md),
team reward, 30 steps of 4 repos × G=4, one seed.
[Relay](../entities/protocol-relay.md) with N=4, m=3, one CI run per
contributor, NOTES.md visible, 12,288 tokens per contributor. The prompt
states the contributor's position (k of N), the two CI modes and their
scores, and that later contributors can read NOTES.md. It never asks the agent
to help anyone, and never says whether the reward is individual or team.
Source: [experiment 1](../../sources/sacrifice-relay-experiment-1.md).

"Knew the rule" (`rule_known_at_start`) means the rule's ID appeared anywhere
in the repo when the contributor started, not only in NOTES.md.

## Evidence

**Who reviews, by position** (sacrifice rate among contributors with a real
choice: no rule at the start, ran CI). [partial]

| | Contributor 1 | Contributor 2 | Contributor 3 | Contributor 4 (can help no one) |
|---|---|---|---|---|
| Steps 0–9 | 4/125 = 3% | 17/98 = 17% | 14/73 = 19% | 3/63 = 5% |
| Steps 20–29 | 2/133 = 2% | 16/124 = 13% | 9/80 = 11% | 3/82 = 4% |

For the team, a review by contributor 1 is worth the most (+0.62 team score,
see [rewarded but not learned](rewarded-choice-not-learned.md)). Yet
contributor 1 reviews least.

**What triggers a review.** 90 of 93 reviews by contributors 2–4 came after an
earlier contributor submitted and wrote about the extended checks in
NOTES.md. Without such a note, the review rate was 2% (3/194). Contributor 1
never sees such a note and reviewed 2–4% of the time. [partial]

- How a "note" is detected: `payoff.py` looks for an earlier contributor who
  submitted and ran a bash command that mentions NOTES.md and "extended" and
  writes to a file (`>`, `tee` or `write`). It is a keyword heuristic, and the
  transcripts have not been read.
- Before training, the base rate looked similar: in 16 untrained relays on
  unfiltered problems, 3 of 38 contributors with a real choice reviewed
  (about 8%, plausibly 3–21%)
  ([benchmark](../../sources/sacrifice-relay-throughput-bench.md)). [pilot]

**Sharing works.** [partial]

- For 87 of 93 sacrifices by contributors 1–3 (94%), the rule reached the next
  contributor. The other 6 ran out of tokens before writing notes.
- Followers who started knowing the rule averaged a score of 1.64 and scored 3
  in 54% of cases. Their main limit was reaching CI at all (66% did). Among
  followers who knew the rule *and* submitted, about 82% scored 3 (81.8% of
  n=33 in steps 0–9, 82.1% of n=28 in steps 20–29). The two rates differ
  because the second counts only those who reached CI and submitted.
- For comparison, contributors 2–4 in playthroughs with no sacrifice averaged
  0.47.

**Under training.** After a note, the review rate fell: 20% (steps 0–9), 18%
(10–19), 11% (20–29). The split was chosen after looking at the data. [pilot]

## Proactive gathering can be learned (experiment 2, 2026-10-02/03) [partial]

Experiment 2 changed the setup in three ways:

- no position in the prompt, and random task-folder names, so a contributor
  has to infer from the repo whether anyone came before it;
- 0/1 scoring: 1 only with the tests *and* the rule, so a review costs a
  contributor without the rule nothing;
- a prompt sentence saying the extended checks cannot be worked out from the
  task, the code or the tests.

Under team reward, the first contributor learned to review before any sign
that it was needed. One seed per model;
[local backend](../entities/local-backend.md), LoRA rank 32. Source:
[study report §3.2–3.6](../../sources/sacrifice-relay-experiments-1-3-and-evals.md).

| | Qwen3.8-27B, 4 × 4 per step | [Qwen3.6-35B-A3B](../entities/qwen3-6-35b-a3b.md), 8 × 4 per step |
|---|---|---|
| Contributor 1 reviewed (of its CI runs), steps 0–9 → last 10 | 17% → 80% (60 steps) | 38% → 77% (80 steps) |
| Started knowing the rule (all contributors) | 19% → 59% | 32% → 61% |
| Rule reached the next contributor after a review | 94% → 98% (steps 0–29) | not reported in the source |
| Redundant reviews (knew the rule, reviewed anyway) | 18% → 10% → 24% | 54% → 21% |

- **Proactive gathering for others can be learned** when it is free for the
  gatherer and pays the team. [partial]
- **It holds on new problems.** Contributor 1 reviewed 93% (27B) and 76%
  (A3B) of its CI runs on held-out repos, and 74% / 70% under a fully changed
  surface ([behaviour vs format](behaviour-transfers-across-format.md)).
  [partial]
- **Only under team reward.**
  - Under individual reward contributor 1 never scores (0 of 480), so it gets
    no signal, and its review rate drifted (23% → 30%).
  - In experiment 2.1, followers without the rule did not learn to review
    either. [partial]
- **The followers' cue is a base-model habit, not a learned one.** In
  experiment 2.1 (27B), followers without the rule reviewed about twice as
  often after an earlier contributor's CI run (about 33%) as without one
  (10–18%). That gap was there from step 0 and did not grow. [partial]
- **When the review costs the reviewer and does not pay** (experiment 3,
  A3B), team RL removed reviewing altogether
  ([pays in practice](sacrifice-pays-in-practice.md)). [partial]

## Interpretation [open]

- Untrained, the model treats the review as a fix for a reported failure, not
  as an investment for later contributors. Why is untested. One possibility: the
  note makes the hidden checks salient and shows that a plain submit misses
  the bonus, while contributor 1 has no such evidence.
- This matters beyond this environment. The program's
  [swarm](../entities/protocol-swarm.md) and
  [coordinator](../entities/protocol-coordinator.md) protocols rely on agents
  sharing through files. This is the first measurement in this wiki of whether
  agents *pay* to gather information for others. It now covers one
  environment and two models.
- Team training can turn reactive gathering into proactive gathering when
  the gathering is free for the gatherer. Whether the same holds when it
  costs the gatherer is open (see
  [the synthesis](../syntheses/does-rl-teach-sacrifice.md)).

## Open questions

- ~~Does removing the position from the prompt change who reviews? The next
  experiment does this. [open]~~ Experiment 2 removed it, but together with
  two other changes, and contributor 1 started reviewing far more (3% → 17%
  untrained). The position's own effect is not isolated. [open]
- Does a hidden-notes ablation (each contributor starts with an empty
  NOTES.md; `notes: hidden`) remove the trigger? It would also cut the main
  hand-off route, though the rule could still pass through other files.
  [open]
- Read a sample of transcripts to check the note heuristic. [open]

See also: [the synthesis](../syntheses/does-rl-teach-sacrifice.md),
[multi-session carry](multi-session-carry.md) (notes-to-self, the
single-agent analogue of a shared notes file).
