---
type: concept
title: "Reactive vs proactive information gathering between agents"
description: "In the code_rules relay, Qwen3.8-27B pays to reveal the hidden rule almost only after an earlier contributor's note reports failed extended checks (90 of 93 reviews by contributors 2–4; 2% without such a note), and the first contributor, whose review is worth the most to the team, rarely reviews (2–4%); once the rule is known it is passed on reliably (94%) [partial]."
resource: experiments/2026-09-25_sacrifice-relay/payoff.py
tags: [multi-agent, information-sharing, notes, relay, sacrifice, behaviour, code-rules]
timestamp: 2026-10-01
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

## Interpretation [open]

- The model treats the review as a fix for a reported failure, not as an
  investment for later contributors. Why is untested. One possibility: the
  note makes the hidden checks salient and shows that a plain submit misses
  the bonus, while contributor 1 has no such evidence.
- This matters beyond this environment. The program's
  [swarm](../entities/protocol-swarm.md) and
  [coordinator](../entities/protocol-coordinator.md) protocols rely on agents
  sharing through files. This is the first measurement in this wiki of whether
  agents *pay* to gather information for others. It covers one environment
  and one model.

## Open questions

- Does removing the position from the prompt change who reviews? The next
  experiment does this. [open]
- Does a hidden-notes ablation (each contributor starts with an empty
  NOTES.md; `notes: hidden`) remove the trigger? It would also cut the main
  hand-off route, though the rule could still pass through other files.
  [open]
- Read a sample of transcripts to check the note heuristic. [open]

See also: [the synthesis](../syntheses/does-rl-teach-sacrifice.md),
[multi-session carry](multi-session-carry.md) (notes-to-self, the
single-agent analogue of a shared notes file).
