---
type: entity
title: "Protocol: relay (sequential contributors over a shared workspace)"
description: "N contributors work one after another, each in a fresh context with its own grade and credit, over a workspace that persists between them, so information passes only through files; relay_n4 and relay_n1 (solo control) are the configs used with code_rules."
resource: src/marli/interact/protocols/relay.py
tags: [protocol, relay, sequential, multi-agent, shared-workspace, notes]
timestamp: 2026-10-01
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
- **Compute.** Because contributors run in sequence, the critical path is the
  sum of all contributors. At N=4 × 12,288 tokens it is up to about 48k tokens,
  and that relay sets the sampling time of a training step (see
  [GPU time-sharing](../concepts/gpu-time-sharing.md)).

## Configs

| Config | `n_agents` | `env_tools` | Use |
|---|---|---|---|
| `relay_n4` | 4 | bash, ci_submit, ci_review | the relay arms |
| `relay_n1` | 1 | bash, ci_submit, ci_review | the solo control (reviewing never pays) |

## Used in

- **Sacrifice relay, experiment 1.** `relay_n4` with team reward, Qwen3.8-27B
  on the local backend, 30 steps of 4 repos × G=4, one seed. Team score rose
  +0.086 (z=2.3), and the sacrifice rate did not rise
  ([synthesis](../syntheses/does-rl-teach-sacrifice.md)). [partial] The
  pre-registered `relay_individual` and `solo` arms were not run.
