# marl-investigations

Private research repo for **multi-agent RL on LLMs**. The `marli` library
provides:

- an **interaction layer** in which several LLM agents take turns under a
  protocol (debate, solver/critic, round-robin, self-consistency, …), with
  per-agent visibility and token-level records;
- **evaluation** of any policy × protocol × benchmark grid, with
  compute-matched baselines and paired statistics;
- **RL training** that shares one algorithm loop between a local learner
  (LoRA + vLLM on RunPod) and Tinker;
- **serving** for vLLM with LoRA hot-swap, and **dataset building** from
  benchmark registries.

Every component is also a `marli` CLI verb. Agents (Claude, Codex) chain the
verbs through the JSON manifests each one writes.

```bash
uv sync --extra dev
uv run marli --help
uv run --extra dev pytest -q
```

## Where things live

| Path | Contents |
|---|---|
| `src/marli/` | the library |
| `experiments/` | one directory per study |
| `examples/` | the kept-green on-ramp |
| `docs/wiki/` | curated findings |
| `docs/plans/` | design docs |

Conventions for humans and agents are in [CLAUDE.md](CLAUDE.md).
