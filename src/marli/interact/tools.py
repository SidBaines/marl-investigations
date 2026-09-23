"""Tools: specs the model sees + async handlers the harness runs.

A tool is registered by name (``TOOLS``) as a factory ``(ToolCtx) -> Tool``;
roles list tool names (``RoleSpec.tools``). Environments contribute their own
tools (``python``, ``bash``) through ``Env.tools(role)``.

Tool calls in one turn run **in the order sampled**. A *control* tool
(``submit``, ``end_session``, ``return_report``) ends the agent's turn: any
later calls in the same turn receive an error result and are not executed.
Tools with ``shared=True`` (anything touching the shared sandbox or workspace
writes) run inside ``scheduler.tool_phase`` so lockstep ordering is
deterministic; pure tools (reads) may run concurrently.

Results are truncated (head + tail, ``Limits.tool_output_chars``) before they
enter the agent's context; the truncated text is what is recorded.

Built-in tools (M1-6): ``write_scratchpad(content, mode="append"|"overwrite")``,
``read_scratchpad(agent_id, version=None)``, ``list_scratchpads()``,
``read_notes()``, ``write_notes(content)``, ``submit(answer)``,
``end_session()``, ``return_report(report)``, ``wait_for_update(timeout_s)``
(async only). ``spawn_workers(tasks=[{task, context}])`` is registered by the
coordinator protocol (M2-3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from marli.registry import FnRegistry
from marli.render.base import ToolSpec


class ToolError(Exception):
    """A user-visible tool failure: becomes an error result for the agent, not a crash."""


@dataclass(frozen=True)
class ToolResult:
    content: str
    error: str | None = None
    # Control effects the runtime applies after the tool returns, e.g.
    # {"submit": "42"} | {"end_session": True} | {"report": "..."}.
    control: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCtx:
    """Everything a handler may touch. Graders never get one."""

    agent_id: str
    role: str
    tick: int | None
    seq: int
    workspace: Any  # interact.workspace.Workspace
    sandbox: Any | None  # envs.sandbox.base.Sandbox (None for math-only envs)
    scheduler: Any  # interact.scheduler.Scheduler (for block/unblock in spawn)
    system: Any  # interact.system.SystemIO (spawn_workers starts agents through it)
    ledger: Any  # interact.limits.Ledger


class Tool(Protocol):
    spec: ToolSpec
    shared: bool
    control: bool  # ends the turn (submit / end_session / return_report)

    async def __call__(self, ctx: ToolCtx, **arguments: Any) -> ToolResult: ...


TOOLS: FnRegistry = FnRegistry("tools")  # name -> factory() -> Tool
