"""Math grading depends only on the submitted answer and the task reference."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

from marli.envs.base import Env, Task
from marli.envs.registry import ENVS
from marli.envs.sandbox.subprocess import SubprocessSandbox
from marli.interact.tools import Tool, ToolCtx, ToolError, ToolResult
from marli.render.base import ToolSpec

if TYPE_CHECKING:
    from marli.tasks.verifiers import MathVerifier

_VERIFIER: MathVerifier | None = None
_VERIFIER_LOCK = threading.Lock()


def shared_verifier() -> MathVerifier:
    """Share the verifier's worker pool across all math episodes in this process."""
    global _VERIFIER
    with _VERIFIER_LOCK:
        if _VERIFIER is None:
            from marli.tasks.verifiers import MathVerifier

            _VERIFIER = MathVerifier()
        return _VERIFIER


@dataclass(frozen=True)
class MathEnvConfig:
    python_tool: bool = False
    python_timeout_s: float = 30.0
    instruction: str = (
        "Solve the problem. When you are confident, call the `submit` "
        "tool with your final answer only (no working, no units). For answers that "
        "are not integers, use LaTeX."
    )
    answer_format: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.python_timeout_s) or self.python_timeout_s <= 0:
            raise ValueError("python_timeout_s must be positive and finite")
        if self.answer_format not in (None, "integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")


class _PythonTool:
    shared = True
    control = False
    spec = ToolSpec(
        "python",
        "Run Python code in the episode's shared sandbox.",
        {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    )

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s

    async def __call__(self, ctx: ToolCtx, *, code: str) -> ToolResult:
        if ctx.sandbox is None:
            raise ToolError("python requires an active episode sandbox")
        try:
            result = await ctx.sandbox.exec(["python3", "-"], timeout_s=self.timeout_s, stdin=code)
        except (ValueError, OSError) as exc:
            raise ToolError(f"python execution failed: {exc}") from exc
        return ToolResult(json.dumps(asdict(result)))


@ENVS.register("math")
class MathEnv(Env):
    name = "math"

    def __init__(self, config: dict[str, Any] | MathEnvConfig, task: Task) -> None:
        if isinstance(config, dict):
            unknown = config.keys() - {field.name for field in fields(MathEnvConfig)}
            if unknown:
                raise ValueError(f"unknown math environment config keys: {sorted(unknown)}")
            config = MathEnvConfig(**config)
        self.config = config
        self.task = task
        self.answer_format = (
            config.answer_format
            if config.answer_format is not None
            else task.meta.get("answer_format")
        )
        if self.answer_format not in ("integer", "latex"):
            raise ValueError("answer_format must be 'integer' or 'latex'")
        self._sandbox: SubprocessSandbox | None = None

    async def setup(self) -> None:
        if self.config.python_tool and self._sandbox is None:
            sandbox = SubprocessSandbox()
            await sandbox.start()
            self._sandbox = sandbox

    async def teardown(self) -> None:
        if self._sandbox is not None:
            await self._sandbox.close()
            self._sandbox = None

    @property
    def sandbox(self) -> SubprocessSandbox | None:
        return self._sandbox

    def task_message(self, role: str) -> str:
        return f"{self.task.prompt}\n\n{self.config.instruction}"

    def tools(self, role: str) -> list[Tool]:
        return [_PythonTool(self.config.python_timeout_s)] if self.config.python_tool else []

    async def grade(self, submission: str | None) -> dict[str, float]:
        from marli.tasks.verifiers import extract_boxed

        pred = submission
        if pred is not None:
            boxed = extract_boxed(pred)
            if boxed is not None:
                pred = boxed
        correct = await shared_verifier().verify(pred, str(self.task.answer), self.answer_format)
        return {"correct": float(correct), "answered": float(submission is not None)}

    def canonical(self, submission: str | None) -> str | None:
        if submission is None:
            return None
        from marli.tasks.verifiers import _integer_key, normalize_answer

        answer = normalize_answer(submission)
        if not answer:
            return None
        return (_integer_key(answer) or answer) if self.answer_format == "integer" else answer

    async def same_answer(self, a: str | None, b: str | None) -> bool:
        if a is None or b is None:
            return False
        return await shared_verifier().verify(a, b, self.answer_format)
