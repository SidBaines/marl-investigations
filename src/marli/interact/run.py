"""Run one episode of one protocol on one task — the single entry point used by
eval rollouts, RL rollouts and tests.

::

    @dataclass
    class EpisodeSpec:
        protocol: Protocol
        env: Env                        # already constructed for this task
        task: Task
        seating: dict[str, str]         # role -> policy_id
        policies: dict[str, Policy]     # policy_id -> TokenPolicy | ChatPolicy
        # policy_id -> renderer FACTORY (token policies only): renderers are per-agent-lineage
        # objects (they remember the tool specs they rendered), so each agent gets its own.
        renderers: dict[str, Callable[[], DeltaRenderer]]
        limits: Limits
        schedule: str = "lockstep"      # lockstep | async
        delivery: DeliverySpec = DeliverySpec()
        run_seed: int = 0
        episode_idx: int = 0
        group_id: str = ""
        config_hash: str = ""
        protocol_name: str = ""
        backend: str = ""
        clock: Clock | None = None

    async def run_episode(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
        # 1. env.setup(); build Workspace, Scheduler, Ledger, Recorder, SystemIO
        # 2. outcome = await protocol.run(io)   (bounded by limits.episode.max_wall_s)
        # 3. grades: per-agent (grade_individual) + "_system" for outcome.final_answer
        # 4. metrics = compute_metrics(...); assemble Episode; env.teardown() in finally
        # BackendError ends only the failing agent; spend/config/programming errors propagate.
        # Raises ConfigError for invalid specs (unseated role, renderer/policy mismatch,
        # trainable seat with non-raw sampling, ctx > backend max_seq_len).

Lockstep records use logical (tick, seat_key, in-agent) order, independently
of generation latency. Sequence references are normalized together before
metrics are computed; async records retain their observed order. Replayability
requires seated policies to declare deterministic=True (absent means False).
System prompt templates use the role's count for n_agents, or 0 for dynamic roles.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from marli.envs.base import Env, Task
from marli.errors import ConfigError
from marli.interact.agent import SamplingOverrides
from marli.interact.context import make_context_manager
from marli.interact.limits import Ledger, Limits
from marli.interact.records import Recorder, compute_metrics
from marli.interact.scheduler import AsyncScheduler, Clock, LockstepScheduler, SystemClock
from marli.interact.system import EpisodeSystem, Protocol
from marli.interact.tools import TOOLS
from marli.interact.types import Episode, EventKind, Outcome
from marli.interact.workspace import DeliverySpec, Permissions, Workspace
from marli.policy.base import Policy, SamplingSpec, TokenPolicy, check_trainable_sampling
from marli.render.base import DeltaRenderer


@dataclass
class EpisodeSpec:
    protocol: Protocol
    env: Env
    task: Task
    seating: dict[str, str]
    policies: dict[str, Policy]
    renderers: dict[str, Callable[[], DeltaRenderer]]
    limits: Limits
    schedule: str = "lockstep"
    delivery: DeliverySpec = DeliverySpec()
    run_seed: int = 0
    episode_idx: int = 0
    group_id: str = ""
    config_hash: str = ""
    protocol_name: str = ""
    backend: str = ""
    clock: Clock | None = None
    # Role -> seat sampling controls (the contract's SamplingOverrides).
    sampling: dict[str, SamplingOverrides] = field(default_factory=dict)


def _validate(spec: EpisodeSpec) -> None:
    spec.limits.__post_init__()
    if spec.schedule not in {"lockstep", "async"}:
        raise ConfigError("schedule must be lockstep or async")
    if spec.delivery.mode not in {"notify", "push", "pull"}:
        raise ConfigError("delivery.mode must be notify, push or pull")
    if spec.delivery.view not in {"latest", "full"}:
        raise ConfigError("delivery.view must be latest or full")
    for name in ("push_max_chars", "index_first_line_chars"):
        if getattr(spec.delivery, name) < 0:
            raise ConfigError(f"delivery.{name} must be non-negative")
    roles = spec.protocol.roles()
    if len({role.role for role in roles}) != len(roles):
        raise ConfigError("protocol.roles must have unique role names")
    for role in roles:
        try:
            role.system_prompt.format(
                agent_id=f"{role.role}0", role=role.role, n_agents=role.count or 0
            )
        except (KeyError, ValueError, IndexError, AttributeError) as exc:
            raise ConfigError(f"role {role.role!r}: invalid system_prompt template: {exc}") from exc
        if role.role not in spec.seating:
            raise ConfigError(f"unseated role {role.role!r}")
        policy_id = spec.seating[role.role]
        if policy_id not in spec.policies:
            raise ConfigError(f"role {role.role!r}: unknown policy {policy_id!r}")
        policy = spec.policies[policy_id]
        if isinstance(policy, TokenPolicy):
            if policy_id not in spec.renderers:
                raise ConfigError(f"policy {policy_id!r}: missing renderer factory")
            renderer = spec.renderers[policy_id]()
            if renderer.name != policy.renderer_name:
                raise ConfigError(f"policy {policy_id!r}: renderer name mismatch")
        elif policy.trainable:
            raise ConfigError(f"trainable chat policy {policy_id!r} is not supported")
        max_seq_len = getattr(policy, "max_seq_len", None)
        if max_seq_len is not None and spec.limits.ctx.max_ctx > max_seq_len:
            raise ConfigError(f"ctx.max_ctx exceeds policy {policy_id!r} max_seq_len")
        sampling = spec.sampling.get(role.role, SamplingOverrides())
        if policy.trainable:
            check_trainable_sampling(
                SamplingSpec(
                    spec.limits.call.max_tokens,
                    sampling.temperature,
                    sampling.top_p,
                    sampling.top_k,
                ),
                policy_id=policy_id,
            )
        if role.limits_key not in {"agent", "worker"}:
            raise ConfigError(f"role {role.role!r}: limits_key must be agent or worker")
        make_context_manager(role.context, spec.limits)
        available = set(TOOLS.names()) | {tool.spec.name for tool in spec.env.tools(role.role)}
        for name in role.tools:
            if name not in available:
                raise ConfigError(f"role {role.role!r}: unknown tool {name!r}")


async def run_episode(spec: EpisodeSpec) -> tuple[Episode, dict[str, list[int]]]:
    """Validate before spending, isolate backend failures, and always close the env."""
    _validate(spec)
    clock = spec.clock if spec.clock is not None else SystemClock()
    group_id = spec.group_id or f"{spec.task.task_id}/{spec.config_hash[:8]}"
    recorder = Recorder(f"{group_id}/e{spec.episode_idx}", clock=clock)
    roles = spec.protocol.roles()
    workspace = Workspace(
        roles={},
        permissions={
            role.role: replace(
                role.permissions or Permissions(), notes=role.context.kind in {"notes", "both"}
            )
            for role in roles
        },
        delivery=spec.delivery,
        staged=spec.schedule == "lockstep",
        notes_cap_chars=max((role.context.notes_cap_chars for role in roles), default=4000),
    )
    scheduler_type = LockstepScheduler if spec.schedule == "lockstep" else AsyncScheduler
    scheduler = scheduler_type(workspace, recorder, clock=clock)
    ledger = Ledger(
        spec.limits,
        schedule=spec.schedule,
        expected_agents=max(1, sum(role.count or 0 for role in roles)),
    )
    io = EpisodeSystem(
        spec, workspace=workspace, scheduler=scheduler, ledger=ledger, recorder=recorder
    )
    runner: asyncio.Task[Outcome] | None = None
    timer: asyncio.Task[None] | None = None
    errors: list[str] = []

    async def drive() -> Outcome:
        outcome = await spec.protocol.run(io)
        await io.wait(io.handles)
        return outcome

    try:
        await spec.env.setup()
        deadline = clock.now() + spec.limits.episode.max_wall_s

        async def expire() -> None:
            await clock.sleep(max(0.0, deadline - clock.now()))

        runner = asyncio.create_task(drive())
        timer = asyncio.create_task(expire())
        done, _ = await asyncio.wait({runner, timer}, return_when=asyncio.FIRST_COMPLETED)
        if runner in done:
            outcome = runner.result()
        else:
            errors.append("episode wall-clock limit")
            ledger.limit_hit("_episode", "episode.max_wall_s")
            runner.cancel()
            for handle in io.handles:
                handle.task.cancel()
            await asyncio.gather(
                runner, *(handle.task for handle in io.handles), return_exceptions=True
            )
            outcome = Outcome(
                None,
                {key: runtime.submission for key, runtime in io.runtimes.items()},
                spec.protocol.name,
            )
        errors.extend(
            runtime.error for runtime in io.runtimes.values() if runtime.error is not None
        )
        grades = {
            agent_id: await spec.env.grade(runtime.submission)
            for agent_id, runtime in io.runtimes.items()
            if runtime.submission is not None
        }
        grades["_system"] = await spec.env.grade(outcome.final_answer)
        for key, hits in ledger.limits_hit().items():
            for limit in hits:
                recorder.limit_hit(key, limit)
        episode, buffers = recorder.build(
            group_id=group_id,
            episode_idx=spec.episode_idx,
            task_id=spec.task.task_id,
            protocol=spec.protocol_name or spec.protocol.name,
            config_hash=spec.config_hash,
            backend=spec.backend,
            outcome=outcome,
            grades=grades,
            metrics={},
            replayable=spec.schedule == "lockstep"
            and all(
                getattr(
                    spec.policies[spec.seating[role.role]],
                    "deterministic",
                    False,
                )
                for role in roles
            ),
            ok=not errors,
            errors=tuple(errors),
        )
        if spec.schedule == "lockstep":
            episode = _lockstep_order(episode)
        return replace(episode, metrics=compute_metrics(episode, buffers)), buffers
    finally:
        tasks = [handle.task for handle in io.handles]
        tasks.extend(task for task in (runner, timer) if task is not None)
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await spec.env.teardown()


def _lockstep_order(episode: Episode) -> Episode:
    """Assign logical sequence numbers without serializing concurrent sampling."""
    seats = {agent.agent_id: agent.seat_key for agent in episode.agents}
    ordered = sorted(
        episode.events,
        key=lambda event: (
            event.tick if event.tick is not None else -1,
            event.kind == EventKind.COMMIT,  # staged writes commit at tick close
            seats[event.agent_id],
            event.seq,
        ),
    )
    seqs = {event.seq: seq for seq, event in enumerate(ordered)}
    calls = tuple(
        sorted(
            (replace(call, seq=seqs[call.seq]) for call in episode.calls),
            key=lambda call: call.seq,
        )
    )
    segment_order = {call.segment_id: call.seq for call in reversed(calls)}
    return replace(
        episode,
        calls=calls,
        events=tuple(replace(event, seq=seqs[event.seq]) for event in ordered),
        segments=tuple(sorted(episode.segments, key=lambda seg: segment_order[seg.segment_id])),
        workspace_log=tuple(replace(write, seq=seqs[write.seq]) for write in episode.workspace_log),
    )
