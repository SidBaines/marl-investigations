"""Continue saved episodes whose agents ran out of budget, under larger limits.

``eval continue`` reads an ``eval rollout`` EpisodeSet and its run config. It
replays each selected episode's recorded prefix exactly (interact/replay.py) and
samples live from the cut, so a continued episode is distributed as if the
source run had used the new limits from the start. The output is a complete
EpisodeSet over the same TaskSet, with one row per source episode and the same
episode ids, so ``data filter``, ``eval score`` and study scripts read it
unchanged:

* **continued** — at least one agent's turn ended on a limit (select=truncated),
  or every ok episode (select=all, a replay-fidelity check);
* **copied** — verbatim source rows: not selected, not truncated, or not ok in
  the source (continuation cannot repair a failed source episode).

``continuations.jsonl`` records the provenance of every row: source episode id,
cut (agent id, call index, reason, extended?), replayed calls, replayed tool
calls and result mismatches, and any divergence. The manifest carries the
counts and references both the source EpisodeSet and its TaskSet.

``limits`` are overrides merged onto the source's limits, e.g.
``limits.agent.max_gen_tokens=20480``. Raising only limits that bound an agent's
*end* keeps the prefix equivalent. A change that alters a replayed call (a
system prompt that states a budget, a session count, or a larger per-call cap
where an earlier completion was cut) is rejected before sampling, or detected as
a divergence during replay.

Resumable per episode. A divergence is permanent for that episode (``fail``:
row not-ok, not retried); backend errors are retried on resume.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marli.budget import SpendGuard
from marli.config import compose, from_mappings, input_field, runtime_field, to_dict
from marli.envs.registry import make_env
from marli.errors import BudgetExceededError, ConfigError
from marli.eval.policies import PolicySpec, build_policies, resolve_spec
from marli.eval.rollout import EpisodeSet, RolloutConfig
from marli.eval.store import compact
from marli.handles import InputRef
from marli.interact import records
from marli.interact.agent import SamplingOverrides
from marli.interact.configs import build_protocol, resolve_protocol
from marli.interact.limits import Limits
from marli.interact.replay import SEQUENTIAL_PROTOCOLS, ReplayPlan, find_cut
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.types import Episode
from marli.policy.refs import parse_ref
from marli.rundir import RunDir
from marli.seeds import derive_seed
from marli.tasks.taskset import TaskSet, read_tasks

PROVENANCE = "continuations.jsonl"


@dataclass
class ContinueConfig:
    episodes: str | None = input_field(None, help="source EpisodeSet (an eval rollout run)")
    limits: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, PolicySpec] = field(default_factory=dict)
    select: str = "truncated"
    episode_ids: list[str] = field(default_factory=list)
    on_divergence: str = "fail"
    seed: int = 0
    record_tokens: bool = False
    retry_failed: bool = runtime_field(True, help="retry backend-failed continuations on resume")
    concurrency: int = runtime_field(8, help="concurrent episodes")
    max_usd: float | None = runtime_field(None, help="spend guard for this run")

    def __post_init__(self) -> None:
        if self.select not in {"truncated", "all"}:
            raise ConfigError("select must be truncated or all")
        if self.on_divergence not in {"fail", "live"}:
            raise ConfigError("on_divergence must be fail or live")
        if type(self.concurrency) is not int or self.concurrency < 1:
            raise ConfigError("concurrency must be a positive integer")
        if not isinstance(self.limits, dict):
            raise ConfigError("limits must be a mapping of overrides")
        SpendGuard(self.max_usd)


def should_resume(handle: EpisodeSet, cfg: ContinueConfig) -> bool:
    """Reopen while backend-failed continuations remain retryable."""
    counts = handle.meta.get("continuation", {}).get("counts", {})
    return cfg.retry_failed and counts.get("error", 0) > 0


def _system_prompts(protocol: Any, limits: Limits) -> dict[str, str]:
    """Role prompts as agents render them; any budget they mention must stay unchanged."""
    return {
        role.role: role.system_prompt.format(
            agent_id=f"{role.role}0",
            role=role.role,
            n_agents=role.count or 0,
            max_workers_per_call=limits.spawn.max_per_call,
            max_workers_total=limits.spawn.max_total,
            worker_tokens=limits.worker.max_gen_tokens,
            session_tokens=limits.session.max_gen_tokens,
        )
        for role in protocol.roles()
    }


def _latest(root: Path, *, with_tokens: bool) -> dict[str, tuple[Episode, dict[str, list[int]]]]:
    latest: dict[str, tuple[Episode, dict[str, list[int]]]] = {}
    for episode, buffers in records.read_episodes(root, with_tokens=with_tokens):
        latest[episode.episode_id] = (episode, buffers)
    return latest


async def continue_episodes(cfg: ContinueConfig, run: RunDir) -> EpisodeSet:
    if cfg.episodes is None:
        raise ConfigError("continue requires episodes")
    source = EpisodeSet.load(cfg.episodes)
    if source.meta.get("paused"):
        raise ConfigError("source rollout is paused; finish it before continuing its episodes")
    source_cfg = compose(RolloutConfig, source.root / "config.yaml")
    if cfg.record_tokens and source.tokens is None:
        raise ConfigError("record_tokens needs a source that recorded tokens (copies need them)")
    task_refs = [ref for ref in source.inputs if ref.kind == "taskset"]
    if len(task_refs) != 1:
        raise ConfigError("source EpisodeSet must reference exactly one TaskSet")
    taskset = TaskSet.load(task_refs[0].resolve())
    if taskset.sha256() != task_refs[0].sha256:
        raise ConfigError("source TaskSet changed since the rollout")
    tasks = {task.task_id: task for task in read_tasks(taskset)}

    protocol = build_protocol(source_cfg.protocol, source_cfg.protocol_config)
    name, _, protocol_config = resolve_protocol(source_cfg.protocol, source_cfg.protocol_config)
    if protocol.name not in SEQUENTIAL_PROTOCOLS or source_cfg.schedule != "lockstep":
        raise ConfigError(
            f"continuation supports lockstep {sorted(SEQUENTIAL_PROTOCOLS)} protocols, "
            f"not {protocol.name!r} ({source_cfg.schedule})"
        )
    old_limits = source_cfg.limits
    limits = from_mappings(Limits, to_dict(old_limits), cfg.limits)
    old_effective = protocol.adjust_limits(old_limits)
    effective = protocol.adjust_limits(limits)
    if _system_prompts(protocol, old_effective) != _system_prompts(protocol, effective):
        raise ConfigError("the new limits change a system prompt; continuation is not equivalent")
    if effective.session.max_sessions != old_effective.session.max_sessions:
        raise ConfigError("session.max_sessions is part of every prompt; it cannot change")
    if effective.tool_output_chars != old_effective.tool_output_chars:
        warnings.warn(
            "tool_output_chars changed: replayed results keep their recorded text",
            stacklevel=2,
        )

    specs = cfg.policies or source_cfg.policies
    if set(specs) != set(source_cfg.policies):
        raise ConfigError(f"policies must use the source's keys {sorted(source_cfg.policies)}")
    recorded_specs = source.meta.get("policy_specs", {})
    for key, spec in specs.items():
        _, model, renderer = resolve_spec(spec)
        before = recorded_specs.get(key, {})
        if before.get("model") and model is not None and before["model"] != model.name:
            raise ConfigError(f"policy {key!r}: model {model.name!r} != source {before['model']!r}")
        if before.get("renderer") and renderer and before["renderer"] != renderer:
            raise ConfigError(
                f"policy {key!r}: renderer {renderer!r} != source {before['renderer']!r}"
            )

    latest = _latest(source.root, with_tokens=cfg.record_tokens)
    wanted = set(cfg.episode_ids)
    unknown = wanted - latest.keys()
    if unknown:
        raise ConfigError(f"episode_ids not in the source: {sorted(unknown)[:5]}")

    saved = compact(run, record_tokens=cfg.record_tokens)
    done_rows = {row["episode_id"]: row for row in saved}
    provenance: dict[str, dict[str, Any]] = {}
    for row in run.read_rows(PROVENANCE):
        provenance[row["episode_id"]] = row
    # Copies and divergences are final; backend errors are retried.
    done = {
        key
        for key, row in done_rows.items()
        if row["ok"]
        or not cfg.retry_failed
        or provenance.get(key, {}).get("action") in {"copied", "diverged"}
    }
    spend = SpendGuard(cfg.max_usd)
    spend.charge(
        sum(call["usage"]["cost_usd"] for row in saved for call in row["calls"]),
        "previous attempts",
    )
    policies, renderers = await build_policies(specs, spend=spend)
    sampling = {
        role: SamplingOverrides(**to_dict(specs[policy].sampling))
        for role, policy in source_cfg.seating.items()
    }
    backend = ",".join(sorted({parse_ref(spec.ref).kind for spec in specs.values()}))
    semaphore = asyncio.Semaphore(cfg.concurrency)
    stopped: Exception | None = None

    def record(
        episode: Episode, buffers: dict[str, list[int]], action: str, extra: dict[str, Any]
    ) -> None:
        records.write_episode(run.append_row, episode, buffers, record_tokens=cfg.record_tokens)
        row = {
            "episode_id": episode.episode_id,
            "source_episode_id": episode.episode_id,
            "action": action,
            **extra,
        }
        run.append_row(PROVENANCE, row)
        provenance[episode.episode_id] = row

    async def handle(key: str) -> None:
        nonlocal stopped
        episode, buffers = latest[key]
        if not episode.ok:
            record(episode, buffers, "copied", {"why": "source episode not ok"})
            return
        if wanted and key not in wanted:
            record(episode, buffers, "copied", {"why": "not selected"})
            return
        cut = find_cut(episode)
        if cut is None and cfg.select == "truncated":
            record(episode, buffers, "copied", {"why": "no agent ended on a limit"})
            return
        plan = ReplayPlan(
            episode,
            cut,
            seed=derive_seed(cfg.seed, "continue", key),
            on_divergence=cfg.on_divergence,
        )
        async with semaphore:
            spend.check(0, "next episode")
            continued, new_buffers = await run_episode(
                EpisodeSpec(
                    protocol=build_protocol(source_cfg.protocol, source_cfg.protocol_config),
                    env=make_env(source_cfg.env, source_cfg.env_config, tasks[episode.task_id]),
                    task=tasks[episode.task_id],
                    seating=source_cfg.seating,
                    policies=policies,
                    renderers=renderers,
                    limits=limits,
                    schedule=source_cfg.schedule,
                    run_seed=source_cfg.run_seed,
                    episode_idx=episode.episode_idx,
                    group_id=episode.group_id,
                    config_hash=run.config_hash,
                    protocol_name=episode.protocol,
                    backend=backend,
                    sampling=sampling,
                    replay=plan,
                )
            )
        if continued.episode_id != key:
            raise ConfigError(f"continued episode id {continued.episode_id!r} != source {key!r}")
        if not continued.ok and any(
            "BudgetExceededError" in error or "budget limit $" in error
            for error in continued.errors
        ):
            raise BudgetExceededError("; ".join(continued.errors))
        summary = plan.summary()
        if plan.divergence is not None and cfg.on_divergence == "fail":
            action = "diverged"
        elif continued.ok:
            action = "continued"
        else:
            action = "error"
        record(continued, new_buffers, action, summary)
        spend.check(0, "completed episode")

    pending = iter(sorted(key for key in latest if key not in done))

    async def worker() -> None:
        nonlocal stopped
        while stopped is None:
            key = next(pending, None)
            if key is None:
                return
            try:
                await handle(key)
            except Exception as exc:
                if stopped is None or isinstance(exc, BudgetExceededError):
                    stopped = exc
                return

    workers = [asyncio.create_task(worker()) for _ in range(min(cfg.concurrency, len(latest)) or 1)]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    if stopped is not None:
        raise stopped

    saved = compact(run, record_tokens=cfg.record_tokens)
    failures = sum(not row["ok"] for row in saved)
    counts = {"copied": 0, "continued": 0, "diverged": 0, "error": 0, "extended": 0}
    for row in saved:
        entry = provenance.get(row["episode_id"], {})
        counts[entry.get("action", "error")] = counts.get(entry.get("action", "error"), 0) + 1
        counts["extended"] += bool((entry.get("cut") or {}).get("extended"))
    counts["tool_mismatches"] = sum(
        entry.get("tool_mismatches", 0) for entry in provenance.values()
    )
    notes = []
    if counts["diverged"]:
        notes.append(f"{counts['diverged']} continuations diverged from their recording")
    if counts["error"]:
        notes.append(f"{counts['error']} continuations failed; resume to retry")
    for note in notes:
        warnings.warn(note, stacklevel=2)
    usage = {
        key: 0.0
        for key in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cost_usd")
    }
    for row in saved:
        for call in row["calls"]:
            for key in usage:
                usage[key] += call["usage"].get(key) or 0
    resolved_specs = {}
    for key, spec in specs.items():
        ref, model, renderer = resolve_spec(spec)
        resolved_specs[key] = {
            **to_dict(spec),
            "ref": str(ref),
            "model": model.name if model else ref.target,
            "renderer": renderer or getattr(policies[key], "renderer_name", None),
        }
    return EpisodeSet(
        root=run.out,
        inputs=(InputRef.of(source), InputRef.of(taskset)),
        episodes="episodes.jsonl",
        tokens="tokens.jsonl" if cfg.record_tokens else None,
        n=len(saved),
        n_failed=failures,
        protocol=name,
        protocol_config=protocol_config,
        env=source_cfg.env,
        env_config=source_cfg.env_config,
        taskset=str(taskset.manifest_path),
        policies={key: str(parse_ref(spec.ref, resolve_paths=True)) for key, spec in specs.items()},
        usage=usage,
        cost_usd=max(spend.spent, usage["cost_usd"]),
        meta={
            "seating": source_cfg.seating,
            "policy_specs": resolved_specs,
            "limits": to_dict(effective),
            "continuation": {
                "source": str(source.manifest_path),
                "limit_overrides": json.loads(json.dumps(cfg.limits)),
                "select": cfg.select,
                "on_divergence": cfg.on_divergence,
                "counts": counts,
            },
            "warnings": notes,
        },
    )
