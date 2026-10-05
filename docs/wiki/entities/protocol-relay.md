---
type: entity
title: "Protocol: relay (sequential contributors over a shared workspace)"
description: "N contributors work one after another, each in a fresh context with its own grade and credit, over a workspace that persists between them, so information passes only through files; configs relay_n1 (solo control), relay_n3, relay_n4 and relay_n5 for code_rules; opener_role seats contributor 1 on its own (e.g. frozen) policy; used for every sacrifice-relay run and its transfer eval."
resource: src/marli/interact/protocols/relay.py
tags: [protocol, relay, sequential, multi-agent, shared-workspace, notes]
timestamp: 2026-10-05
---

# Protocol: `relay`

Implementation: `src/marli/interact/protocols/relay.py`.

- **Shape.** There is one role, `contributor`, with `count = n_agents`. Each
  contributor is graded. Contributors run **in turn**, never in parallel. Each
  starts in a fresh context (a new agent and a new training segment), but the
  environment's workspace persists. The only way to pass information forward
  is to leave it in files.
- **Place in the taxonomy.** It sits between
  [multi-session](protocol-multi-session.md) (sequential fresh contexts, but
  one agent and one reward) and the [swarm](protocol-swarm.md) (several agents
  sharing a workspace, but in parallel).
- **Tools.** `end_session`, plus the environment's tools listed in
  `env_tools`. Slotted environments such as [code_rules](env-code-rules.md)
  provide graded submissions through slot hooks, so they need no `submit`.
  Non-slotted environments must list `submit`.
- **Limits.** The episode budget must be at least `n_agents ×` the per-agent
  budget, or the config is rejected rather than silently changed. Each
  contributor gets its full budget, with `final_reserve` 0, one session, and
  `on_exhaust: none`, so nothing forces a submission.
- **Seating.** `seat_key = (contributor, slot)`. The study seats every slot on
  one shared LoRA (`contributor: learner:policy`). A per-slot LoRA is a planned
  ablation.
- **A separate opener (`opener_role`, e3f376a).** This gives slot 0 its own
  role name, with the same tools and prompt, so it can sit on a different
  policy while later slots train. Experiment 2.1 seated a frozen, team-trained
  adapter there (a vLLM adapter seat). Agent ids stay `contrib<slot>` either
  way.
- **System prompt.** A relay setting (default: "You are a software engineer
  contributing to a shared code repository…"). The transfer eval's `far` and
  `reworded` conditions paraphrase it.
- **Compute.** Because contributors run in sequence, the critical path is the
  sum of all contributors. At N=4 × 12,288 tokens it is up to about 48k tokens,
  and that relay sets the sampling time of a training step (see
  [GPU time-sharing](../concepts/gpu-time-sharing.md)).

## Configs

| Config | `n_agents` | `env_tools` | Use |
|---|---|---|---|
| `relay_n4` | 4 | bash, ci_submit, ci_review | the relay arms |
| `relay_n1` | 1 | bash, ci_submit, ci_review | the solo control (reviewing never pays); never run |
| `relay_n3`, `relay_n5` | 3, 5 | bash, ci_submit, ci_review | contributor-count transfer conditions (75592a8); built, not yet run |

With N contributors and a mandatory rule (0/1 scoring), contributor 1 can
never score, so the best team score is (N−1)/N. Compare the n3 and n5 cells
through rates and lift over the untrained model, not raw team scores.

## Used in

All sacrifice-relay runs use `relay_n4`, one seed each. Sources:
[experiment 1](../../sources/sacrifice-relay-experiment-1.md) and the
[study report](../../sources/sacrifice-relay-experiments-1-3-and-evals.md).

- **Experiment 1.** Team reward, Qwen3.8-27B on the local backend, 30 steps
  of 4 repos × G=4. Team score rose +0.086 (z=2.3), and the sacrifice rate
  did not rise ([synthesis](../syntheses/does-rl-teach-sacrifice.md)).
  [partial]
- **Experiment 2 and 2.1** (Qwen3.8-27B and Qwen3.6-35B-A3B; team,
  individual, and a frozen opener with individual reward for the rest).
  Under team reward the first contributor learned to review. [partial]
- **Experiment 3** (Qwen3.6-35B-A3B, team and individual). Reviewing was
  extinguished. [partial]
- **The transfer eval** (25 held-out repos × 2 games per cell): see
  [behaviour vs format](../concepts/behaviour-transfers-across-format.md).
  [partial]
- The pre-registered `solo` arm (`relay_n1`) has still not been run. The
  individual arm ran in experiments 2 and 3.
- **Ops.** Relay games are latency-bound: each is a chain of dependent calls
  across contributors. Eval throughput comes from games in flight, and about
  30–40 relay agents per vLLM engine is the safe limit
  ([local backend](local-backend.md)).
