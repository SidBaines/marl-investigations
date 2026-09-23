# marl-investigations

Private research repo for **RL on multi-agent LLM systems**. We study three
setups:
- **coordinator-free swarms** that share scratchpads;
- **a coordinator with dynamically spawned workers**;
- **multi-session agents** that carry state through compaction or
  notes-to-self.

We evaluate them on hard math and agentic coding at matched compute, and train
them with RL. Training can use one shared LoRA or one per role, and we choose
which agents and sessions receive credit.

The `marli` library provides:
- a **token-level interaction layer**: tool-using agent loops, a shared
  workspace, lockstep or async scheduling, compaction/notes context managers,
  and exact per-call token records;
- **environments** (math, function-level coding, SWE) with local sandboxes
  (Docker or a restricted subprocess);
- **evaluation** over policy × protocol × benchmark grids with compute
  accounting (total and critical-path tokens) and paired statistics;
- **RL training** with multiple learners and configurable credit assignment,
  sharing one algorithm loop between a local learner (PEFT + vLLM on RunPod)
  and Tinker.

Every component is also a `marli` CLI verb. Agents (Claude, Codex) chain the
verbs through the JSON manifests each one writes.

```bash
uv sync --extra dev
uv run marli --help
uv run --extra dev pytest -q
```

## Where things live

- `src/marli/` — the library.
- `experiments/` — one directory per study.
- `examples/` — the on-ramp, kept green.
- `docs/wiki/` — curated findings.
- `docs/plans/` — design docs; start with `2026-09-23-agent-systems.md`.

Conventions for humans and agents are in [CLAUDE.md](CLAUDE.md).
