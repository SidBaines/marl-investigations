"""An episode row is the durable commit point; incomplete runs resume by identity."""

from __future__ import annotations

import asyncio
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar

from marli.budget import SpendGuard
from marli.config import input_field, runtime_field, to_dict
from marli.envs.registry import make_env
from marli.errors import BudgetExceededError, ConfigError
from marli.eval.policies import PolicySpec, build_policies, resolve_spec
from marli.eval.store import compact, episode_id, read_episodes
from marli.handles import Handle, InputRef, register_handle
from marli.interact import records
from marli.interact.agent import SamplingOverrides
from marli.interact.configs import build_protocol, resolve_protocol
from marli.interact.limits import Limits
from marli.interact.run import EpisodeSpec, run_episode
from marli.policy.refs import parse_ref
from marli.rundir import RunDir
from marli.tasks.taskset import TaskSet, read_tasks


@dataclass
class RolloutConfig:
    tasks: str | None = input_field(None, help="TaskSet manifest or dir")
    protocol: str = "single"
    protocol_config: dict[str, Any] = field(default_factory=dict)
    env: str = "math"
    env_config: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, PolicySpec] = field(default_factory=dict)
    seating: dict[str, str] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    schedule: str = "lockstep"
    episodes_per_task: int = 1
    run_seed: int = 0
    max_tasks: int | None = None
    record_tokens: bool = False
    retry_failed: bool = runtime_field(True, help="rerun non-ok episodes on resume")
    concurrency: int = runtime_field(8, help="concurrent episodes")
    max_usd: float | None = runtime_field(None, help="spend guard for this run")

    def __post_init__(self) -> None:
        for name in ("episodes_per_task", "concurrency"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if self.max_tasks is not None and (type(self.max_tasks) is not int or self.max_tasks < 0):
            raise ConfigError("max_tasks must be a non-negative integer or None")
        if self.schedule not in {"lockstep", "async"}:
            raise ConfigError("schedule must be lockstep or async")
        self.limits.__post_init__()
        SpendGuard(self.max_usd)


@register_handle
@dataclass(frozen=True)
class EpisodeSet(Handle):
    KIND: ClassVar[str] = "episodes"
    MANIFEST: ClassVar[str] = "episodes.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("episodes", "tokens")

    episodes: str
    tokens: str | None
    n: int
    n_failed: int
    protocol: str
    protocol_config: dict[str, Any]
    env: str
    env_config: dict[str, Any]
    taskset: str
    policies: dict[str, str]
    usage: dict[str, float]
    cost_usd: float

    def summary(self) -> dict[str, Any]:
        return {"n": self.n, "n_failed": self.n_failed, "cost_usd": self.cost_usd}


def should_resume(handle: EpisodeSet, cfg: RolloutConfig) -> bool:
    """Completed sampling remains retryable while transient failures remain."""
    return cfg.retry_failed and handle.n_failed > 0


async def rollout(cfg: RolloutConfig, run: RunDir) -> EpisodeSet:
    if cfg.tasks is None:
        raise ConfigError("rollout requires tasks")
    taskset = TaskSet.load(cfg.tasks)
    tasks = read_tasks(taskset)[: cfg.max_tasks]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ConfigError("TaskSet task_id values must be unique")
    name, _, protocol_config = resolve_protocol(cfg.protocol, cfg.protocol_config)
    protocol = build_protocol(cfg.protocol, cfg.protocol_config)
    effective_limits = protocol.adjust_limits(cfg.limits)
    unknown_policies = set(cfg.seating.values()) - cfg.policies.keys()
    if unknown_policies:
        raise ConfigError(f"seating references unknown policies: {sorted(unknown_policies)}")
    for role in protocol.roles():
        if cfg.seating.get(role.role) not in cfg.policies:
            raise ConfigError(f"unseated role or unknown policy for {role.role!r}")
    for spec in cfg.policies.values():
        _, model, _ = resolve_spec(spec)
        if model and effective_limits.ctx.max_ctx > model.max_ctx:
            raise ConfigError(f"ctx.max_ctx exceeds model {model.name!r} max_ctx")

    saved = compact(run, record_tokens=cfg.record_tokens)
    latest = {row["episode_id"]: row for row in saved}
    done = {key for key, row in latest.items() if row["ok"] or not cfg.retry_failed}
    recorded_cost = sum(call["usage"]["cost_usd"] for row in saved for call in row["calls"])
    previous_spend = recorded_cost
    if run.path("progress.json").exists():
        previous_spend = max(
            previous_spend, json.loads(run.path("progress.json").read_text())["spend"]["spent_usd"]
        )
    spend = SpendGuard(cfg.max_usd)
    spend.charge(previous_spend, "previous attempts")
    policies, renderers = await build_policies(cfg.policies, spend=spend)
    resolved_specs = {}
    for key, spec in cfg.policies.items():
        ref, model, renderer = resolve_spec(spec)
        resolved_specs[key] = {
            **to_dict(spec),
            "ref": str(ref),
            "model": model.name if model else ref.target,
            "renderer": renderer or getattr(policies[key], "renderer_name", None),
        }
    sampling = {
        role: SamplingOverrides(**to_dict(cfg.policies[policy].sampling))
        for role, policy in cfg.seating.items()
    }
    total = len(tasks) * cfg.episodes_per_task
    backend = ",".join(sorted({parse_ref(spec.ref).kind for spec in cfg.policies.values()}))
    pending = iter(
        (task, index)
        for task in tasks
        for index in range(cfg.episodes_per_task)
        if episode_id(task.task_id, run.config_hash, index) not in done
    )
    semaphore = asyncio.Semaphore(cfg.concurrency)
    stopped: Exception | None = None

    def progress() -> None:
        run.write_progress(
            {
                "done": len(done),
                "total": total,
                "failures": sum(not row["ok"] for row in latest.values()),
                "spend": spend.summary(),
            }
        )

    async def worker() -> None:
        nonlocal stopped
        while stopped is None:
            item = next(pending, None)
            if item is None:
                return
            task, index = item
            try:
                async with semaphore:
                    spend.check(0, "next episode")
                    episode, buffers = await run_episode(
                        EpisodeSpec(
                            protocol=build_protocol(cfg.protocol, cfg.protocol_config),
                            env=make_env(cfg.env, cfg.env_config, task),
                            task=task,
                            seating=cfg.seating,
                            policies=policies,
                            renderers=renderers,
                            limits=cfg.limits,
                            schedule=cfg.schedule,
                            run_seed=cfg.run_seed,
                            episode_idx=index,
                            group_id=f"{task.task_id}/{run.config_hash[:8]}",
                            config_hash=run.config_hash,
                            protocol_name=name,
                            backend=backend,
                            sampling=sampling,
                        )
                    )
                    if not episode.ok and any(
                        "BudgetExceededError" in error or "budget limit $" in error
                        for error in episode.errors
                    ):
                        raise BudgetExceededError("; ".join(episode.errors))
                    # Preserve charged retries even if the process dies during the append.
                    progress()
                    records.write_episode(
                        run.append_row, episode, buffers, record_tokens=cfg.record_tokens
                    )
                    done.add(episode.episode_id)
                    latest[episode.episode_id] = {"ok": episode.ok}
                    progress()
                    spend.check(0, "completed episode")
            except Exception as exc:
                # Stop dequeuing immediately, but keep every already-running episode's result.
                if stopped is None or isinstance(exc, BudgetExceededError):
                    stopped = exc
                progress()
                return

    workers = [asyncio.create_task(worker()) for _ in range(min(cfg.concurrency, total))]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    progress()
    if stopped is not None:
        raise stopped
    saved = compact(run, record_tokens=cfg.record_tokens)
    failures = sum(not row["ok"] for row in saved)
    if failures:
        warnings.warn(f"rollout has {failures} failed episodes; resume to retry", stacklevel=2)
    usage: dict[str, float] = {
        key: 0.0
        for key in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cost_usd")
    }
    for episode in read_episodes(run.out):
        for call in episode.calls:
            for key in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cost_usd"):
                usage[key] = usage.get(key, 0.0) + (getattr(call.usage, key) or 0)
    return EpisodeSet(
        root=run.out,
        inputs=(InputRef.of(taskset),),
        episodes="episodes.jsonl",
        tokens="tokens.jsonl" if cfg.record_tokens else None,
        n=len(saved),
        n_failed=failures,
        protocol=name,
        protocol_config=protocol_config,
        env=cfg.env,
        env_config=cfg.env_config,
        taskset=str(taskset.manifest_path),
        policies={
            key: str(parse_ref(spec.ref, resolve_paths=True)) for key, spec in cfg.policies.items()
        },
        usage=usage,
        cost_usd=max(spend.spent, usage.get("cost_usd", 0.0)),
        meta={
            "seating": cfg.seating,
            "policy_specs": resolved_specs,
            "limits": to_dict(effective_limits),
            "warnings": [f"rollout has {failures} failed episodes; resume to retry"]
            if failures
            else [],
        },
    )
