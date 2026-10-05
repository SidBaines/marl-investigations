---
type: entity
title: "Standard cooperation evals (marli eval external)"
description: "The external-eval layer (verb marli eval external, a file-backed suite registry pinned to upstream commits) and its four suites: Li & Shirado's one-shot giving games, HiddenBench (pooling private facts), FAIRGAME's volunteer's dilemma and a planted help request in a stock Inspect agent; first run 2026-10-05 on experiment 2's policies, where only HiddenBench moved (A3B, +4.8 points) [partial]."
resource: src/marli/eval/external/
tags: [evaluation, benchmark, cooperation, multi-agent, inspect, hiddenbench, fairgame, external, sandbox]
timestamp: 2026-10-05
---

# Standard cooperation evals (`marli eval external`)

**Why it exists.** These evals run standard benchmarks in their own
harnesses, not in marli's interaction layer. Running them in our own layer
would bring back the cues under test: our prompt style, tool and turn
structure, and message delivery. It would also break comparability with
published numbers. Sid's call (2026-10-05). It was built on branch
`evals-coop` (8f9f692, b54635f, 0da14bb) and merged into `evals-session-1005`.
Study dir: `experiments/2026-10-05_coop-evals/`.

## The layer

- **Verb:** `marli eval external`, code in `src/marli/eval/external/`.
- **Registry:** one YAML per suite in `src/marli/eval/external/suites/`. Each
  is pinned to an upstream commit or an `inspect-ai` version (0.3.276), and
  sets only its own settings.
- **Serving:** policies are served by vLLM's OpenAI-compatible chat endpoint.
  - The model's own chat template builds the prompt.
  - The reasoning parser keeps thinking out of the reply text that parsers
    and other agents see.
  - Before sampling, a check confirms the server actually serves every
    adapter.
- **Settings:**
  - thinking is on for every model;
  - sampling follows the model card: T 1.0, top_p 0.95, top_k 20, max_tokens
    16,384, and presence penalty 1.5 for the A3B;
  - the upstream benchmarks fix no sampling settings;
  - outputs sampled with other settings are refused.
- **Report:** for each cell, condition and metric:
  - n, unparsed decisions and harness errors, counted separately;
  - the mean with a 95% interval;
  - the gain over the untrained model, paired where the suite allows it
    (McNemar or sign-flip p), otherwise a two-sample bootstrap with a
    permutation p.
- **Mixed cells:** the multi-seat suites can seat one trained model with
  untrained teammates, or the reverse. Results are split into the odd seat
  (*focal*) and the rest (*others*). Mixed cells were not in the 2026-10-05
  run.

## The four suites

| Suite | Harness | Measures | Seats | Pinned to |
|---|---|---|---|---|
| `li_shirado_games` | Inspect task in marli; prompts transcribed from Li & Shirado (EMNLP 2025, arXiv:2502.17720) | giving at a personal cost in one-shot games, 100 trials each: dictator (share of 100 points given), prisoner's dilemma (give 100 doubled points), 4-player public goods (contribute) | 1 | inspect-ai 0.3.276 |
| `hiddenbench` | upstream code plus a small patch (one model per seat, extra request fields, a failing scenario recorded rather than aborting) | pooling private facts: 4 agents, one private fact each, 15 discussion rounds; share of agents right before and after discussion (Li, Naito & Shirado, ICML 2026, arXiv:2505.11556) | 4 | `3be6ca1` |
| `fairgame_volunteer` | FAIRGAME's own runner plus a small patch, via LiteLLM | volunteer's dilemma, 3 players, 10 known rounds, neutral personalities: volunteering rate, round-1 volunteering, share of rounds the group is safe (Buscemi et al., arXiv:2504.14325) | 3 | `fc302a6` |
| `planted_help` | Inspect's stock `react()` agent (bash + submit) in the `marli` sandbox | acting on a note from another agent asking for an unneeded, harmless log line; reading it; telling the user; cost to its own task. 30 original coding tasks × 5 conditions (4 note conditions + control) | 1 | inspect-ai 0.3.276 |

- **Checks on CPU.** The decision parser reproduces Li & Shirado's released
  Table 1 to within 2 points for all 10 model variants. The HiddenBench reader
  matches upstream scoring. All four suites ran end to end against a fake
  vLLM server with scripted players.
- **The `marli` sandbox.** RunPod pods have no Docker, and Inspect's own
  `local` sandbox exposes the host (secrets, network, disk). The `marli`
  provider (`inspect_sandbox.py`) wraps the subprocess sandbox the coding
  environments use:
  - a dedicated uid, with Landlock limiting the filesystem;
  - seccomp blocking network, ptrace and mount;
  - resource limits and a scrubbed environment.

  It needs root and Landlock ABI ≥ 4.
- **Data.** HiddenBench and FAIRGAME tasks are read from the pinned checkouts
  at run time. No benchmark text or transcript is committed, and `results/`
  holds aggregates only. HiddenBench tasks have been public since 2025, but
  every comparison is against the same base model.

## Results, 2026-10-05 [partial]

**Regime.**

- Policies: untrained vs experiment 2's team-trained policy
  ([Qwen3.8-27B](qwen3-8-27b.md) after step 59;
  [Qwen3.6-35B-A3B](qwen3-6-35b-a3b.md) after step 79), one training seed.
- Serving: one data-parallel vLLM server per model on 2×H200, shared with the
  relay transfer eval.
- Scale:
  - games: 100 trials each;
  - HiddenBench: 65 tasks × 3 discussions (7–11 tasks per discussion raised
    upstream errors and are excluded);
  - FAIRGAME: 50 games × 3 players;
  - planted help: 30 tasks × 5 conditions.

Source: [study report §4.4](../../sources/sacrifice-relay-experiments-1-3-and-evals.md);
tables in `experiments/2026-10-05_coop-evals/results/` (in git at 3d6551f).

| | 27B untrained → team | A3B untrained → team |
|---|---|---|
| Dictator: share given | 0.31 → 0.32 | 0.22 → 0.24 |
| Prisoner's dilemma: cooperate | 23% → 20% | 1% → 2% |
| Public goods: contribute | 12% → 14% | 2% → 0% |
| Volunteer's dilemma: volunteered | 17% → 14% (n.s.) | 44% → 45% |
| Volunteer's dilemma: group safe | 39% → 31% (p = 0.2) | 57% → 58% |
| HiddenBench: agents right after discussion | 19.1% → 19.2% (+0.1 [−2.5, +3.1], p = 1.0) | 17.8% → **22.7%** (+4.8 [+1.0, +8.9], p = 0.025) |
| HiddenBench: agents right before discussion | 14.2% → 15.1% | 12.5% → 12.9% |
| Planted help: helped | 0/120 → 0/120 | 0/120 → 0/120 |

- **No effect of team training** on the one-shot games, the volunteer's
  dilemma or the planted help request, for either model. [partial]
- **With thinking on, the A3B is almost always selfish in the one-shot
  games.** This is a floor, matching Li & Shirado's finding for Qwen with
  thinking (0/100 prisoner's-dilemma cooperation with thinking, against
  100/100 without). The 27B is more generous. [partial]
- **HiddenBench is the one suite that moved, for the A3B only.** The gain
  appears after discussion, with no difference before it, which is the
  pattern expected if agents pool facts better. [partial]
  - The first discussion per task gave +5.6, p = 0.10. That motivated two
    more discussions, which alone gave +4.5 [0.0, +9.3], p = 0.077.
  - The group's majority vote moved less: 8.0% → 12.1%, p = 0.19.
  - The 27B showed no effect.
- **Planted help:** see
  [harness-dependent helpfulness](../concepts/harness-dependent-helpfulness.md).

## Tensions

- **Many tests, one hit.** Four suites, several metrics each, two models. One
  result reached p < 0.05, and its sample was extended after a promising
  first look (1 discussion, then 3). The replication on the new discussions
  alone was p = 0.077. Treat the A3B HiddenBench gain as a lead, not a
  finding. [open]
- **Fewer discussions than the paper.** HiddenBench's paper used 10
  discussions per task; this run used 3. [open]
- **Only the final team policies were run.** The individual-arm and 2.1
  policies, which would show whether any effect is specific to team reward,
  were configured but not run. [open]

## Open questions

- Repeat HiddenBench with 10 discussions per task, on the individual-reward
  and 2.1 policies too. [open]

See also: [the synthesis](../syntheses/does-rl-teach-sacrifice.md),
[behaviour vs format](../concepts/behaviour-transfers-across-format.md).
