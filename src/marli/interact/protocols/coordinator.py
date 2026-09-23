"""Delegation joins fresh worker contexts before the coordinator resumes solving.

The blocking spawn tool leaves scheduling, token accounting and worker cleanup
to the interaction runtime, preserving the same call records as other protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

from marli.errors import ConfigError
from marli.interact.system import PROTOCOLS, Protocol, RoleSpec, SystemIO
from marli.interact.tools import TOOLS, ToolCtx, ToolError, ToolResult
from marli.interact.types import Outcome
from marli.interact.workspace import Permissions
from marli.render.base import ToolSpec

_COORDINATOR_SEAT = ("coord", 0)
_DELEGATION_PREAMBLE = (
    "You may delegate parallel subtasks with spawn_workers. Workers start with a fresh "
    "context, so give each a self-contained subtask and any context it needs. They return "
    "a report; you resume after all workers have finished. Use their findings to solve "
    "the original task and call submit with your final answer."
)


@dataclass(frozen=True)
class CoordinatorConfig:
    coordinator_tools: tuple[str, ...] = ("spawn_workers", "submit")
    worker_tools: tuple[str, ...] = ("return_report",)
    env_tools: tuple[str, ...] = ()
    worker_scratchpads: bool = False
    coordinator_system_prompt: str = (
        "You are coordinator {agent_id}. Solve the task, delegating useful subtasks "
        "with spawn_workers. Use submit to provide your final answer."
    )
    worker_system_prompt: str = (
        "You are worker {agent_id}. Solve your assigned subtask independently. "
        "Use return_report to send your findings to the coordinator."
    )

    def __post_init__(self) -> None:
        if "spawn_workers" in (*self.worker_tools, *self.env_tools):
            raise ConfigError("workers cannot use spawn_workers (spawn.max_depth=1)")


@TOOLS.register("spawn_workers")
class _SpawnWorkers:
    shared = False
    control = False
    spec = ToolSpec(
        "spawn_workers",
        "Run parallel workers on self-contained subtasks and wait for all their reports.",
        {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": (
                        "A non-empty list of objects, each with task (string) and "
                        "optional context (string)."
                    ),
                },
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        # The runtime constructs tools per agent, so this counter lasts one episode.
        self._spawned = 0

    async def __call__(self, ctx: ToolCtx, *, tasks: list[dict[str, str]]) -> ToolResult:
        limits = ctx.ledger.limits.spawn
        count = len(tasks)
        if not 1 <= count <= limits.max_per_call or self._spawned + count > limits.max_total:
            raise ToolError(
                f"spawn_workers requires 1 <= len(tasks) <= {limits.max_per_call} "
                f"(spawn.max_per_call) and a running total <= {limits.max_total} "
                f"(spawn.max_total); already spawned {self._spawned}"
            )
        # The tool schema contract supports only shallow array validation.
        for index, task in enumerate(tasks):
            if (
                not isinstance(task, dict)
                or not isinstance(task.get("task"), str)
                or ("context" in task and not isinstance(task["context"], str))
                or task.keys() - {"task", "context"}
            ):
                raise ToolError(
                    f"tasks[{index}] must contain task (string) and optional context (string)"
                )
        affordable = ctx.ledger.reserve_workers(ctx.agent_id, count)
        if affordable < count:
            raise ToolError(f"only {affordable} workers are affordable; requested {count}")

        handles = []
        for task in tasks:
            number = self._spawned
            first_message = f"{task['task']}\n\nYou are helping solve: {ctx.system.task.prompt}"
            if "context" in task:
                first_message += f"\n\nContext: {task['context']}"
            first_message += "\n\nEnd by calling return_report with your findings"
            handle = await ctx.system.start_agent(
                "worker",
                agent_id=f"{ctx.agent_id}/w{number}",
                seat_key=(*_COORDINATOR_SEAT, "w", number),
                first_message=first_message,
                parent=ctx.agent_id,
            )
            handles.append(handle)
            self._spawned += 1

        ctx.scheduler.block(ctx.agent_id)
        try:
            results = await ctx.system.wait(handles)
        finally:
            ctx.scheduler.unblock(ctx.agent_id)
        sections = []
        for result in results:
            report = result.report
            if report is None:
                report = f"[worker {result.agent_id}: no report]"
            sections.append(f"[worker {result.agent_id}] {report}")
        return ToolResult("\n\n".join(sections), control={})


@PROTOCOLS.register("coordinator")
class CoordinatorProtocol(Protocol):
    name = "coordinator"

    def __init__(self, config: CoordinatorConfig | None = None) -> None:
        self.config = config if config is not None else CoordinatorConfig()

    def roles(self) -> list[RoleSpec]:
        config = self.config
        coordinator_tools = (*config.coordinator_tools, *config.env_tools)
        worker_tools = (*config.worker_tools, *config.env_tools)
        if config.worker_scratchpads:
            coordinator_tools += ("read_scratchpad", "list_scratchpads")
            worker_tools += ("write_scratchpad",)
        return [
            RoleSpec(
                "coordinator",
                coordinator_tools,
                config.coordinator_system_prompt,
                count=1,
                limits_key="agent",
                permissions=Permissions(
                    read_others=config.worker_scratchpads,
                    readable_roles=("worker",),
                    write_scratchpad=False,
                ),
            ),
            RoleSpec(
                "worker",
                worker_tools,
                config.worker_system_prompt,
                count=None,
                limits_key="worker",
                permissions=Permissions(
                    read_others=False,
                    write_scratchpad=config.worker_scratchpads,
                ),
            ),
        ]

    async def run(self, io: SystemIO) -> Outcome:
        handle = await io.start_agent(
            "coordinator",
            agent_id="coord0",
            seat_key=_COORDINATOR_SEAT,
            first_message=f"{io.env.task_message('coordinator')}\n\n{_DELEGATION_PREAMBLE}",
        )
        result = await handle
        return Outcome(result.submission, {"coord0": result.submission}, "coordinator")
