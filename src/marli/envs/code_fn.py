"""Grade workspace solutions in isolation, with expected answers held by the parent.

The submission is a completion note, never executable code or a claimed answer.
Only solution.py crosses into a fresh grader uid/workdir. Each test starts from
that same snapshot so one candidate invocation cannot alter later tests. Inputs
arrive on stdin; expected outputs never enter either sandbox.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from typing import Any

from marli.envs.base import Env, Task
from marli.envs.registry import ENVS
from marli.envs.sandbox.subprocess import ResourceLimits, SubprocessSandbox
from marli.errors import ConfigError
from marli.interact.tools import Tool, ToolCtx, ToolError, ToolResult, validate_tool_spec
from marli.render.base import ToolSpec

logger = logging.getLogger(__name__)

_FUNCTIONAL_HARNESS = """\
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
import sys

args = [json.loads(line) for line in sys.stdin.read().splitlines()]
with contextlib.redirect_stdout(io.StringIO()):
    spec = importlib.util.spec_from_file_location("solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = module.Solution() if hasattr(module, "Solution") else module
    result = getattr(target, sys.argv[1])(*args)
print(json.dumps(result))
"""

_COMPILE = (
    "from pathlib import Path; compile(Path('solution.py').read_text(), 'solution.py', 'exec')"
)


@dataclass(frozen=True)
class CodeFnEnvConfig:
    timeout_per_test_s: float = 6.0
    max_tests: int | None = None
    bash_timeout_s: float = 30.0
    instruction: str = (
        "The shared workspace contains problem.md, examples/NN.in and examples/NN.out, "
        "and solution.py. Write your Python solution in solution.py: read stdin and "
        "print stdout for stdin tasks, or define the named function (or Solution method) "
        "for functional tasks. Use bash to test with the public examples. "
        "When done, call submit with a short note. The files are graded, not the note."
    )
    memory_mb: int = 2048

    def __post_init__(self) -> None:
        for name in ("timeout_per_test_s", "bash_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_tests is not None and (type(self.max_tests) is not int or self.max_tests <= 0):
            raise ValueError("max_tests must be a positive integer or None")
        if type(self.memory_mb) is not int or self.memory_mb <= 0:
            raise ValueError("memory_mb must be a positive integer")


class _BashTool:
    shared = True
    control = False
    spec = ToolSpec(
        "bash",
        "Run a bash command in the episode's shared workspace.",
        {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    )

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        validate_tool_spec(self.spec)

    async def __call__(self, ctx: ToolCtx, *, command: str) -> ToolResult:
        if ctx.sandbox is None:
            raise ToolError("bash requires an active episode sandbox")
        try:
            result = await ctx.sandbox.exec(command, timeout_s=self.timeout_s)
        except (ValueError, OSError) as exc:
            raise ToolError(f"bash execution failed: {exc}") from exc
        return ToolResult(json.dumps(asdict(result)))


def _stdout_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


@ENVS.register("code_fn")
class CodeFnEnv(Env):
    name = "code_fn"

    def __init__(self, config: dict[str, Any] | CodeFnEnvConfig, task: Task) -> None:
        if isinstance(config, dict):
            unknown = config.keys() - {field.name for field in fields(CodeFnEnvConfig)}
            if unknown:
                raise ValueError(f"unknown code_fn environment config keys: {sorted(unknown)}")
            config = CodeFnEnvConfig(**config)
        bundle = task.answer
        if not isinstance(bundle, dict) or bundle.get("kind") not in ("stdin", "functional"):
            raise ConfigError("code_fn requires a stdin or functional test bundle")
        if bundle["kind"] == "functional":
            name = bundle.get("fn_name")
            if not isinstance(name, str) or not name.isidentifier():
                raise ConfigError("functional test bundles require fn_name")
        elif bundle.get("fn_name") is not None:
            raise ConfigError("stdin test bundles cannot specify fn_name")
        for key in ("tests", "public"):
            cases = bundle.get(key)
            if not isinstance(cases, list) or (key == "tests" and not cases):
                raise ConfigError(
                    f"code_fn requires a {'nonempty ' if key == 'tests' else ''}{key} list"
                )
            if any(
                not isinstance(case, dict)
                or not isinstance(case.get("input"), str)
                or not isinstance(case.get("output"), str)
                for case in cases
            ):
                raise ConfigError("test inputs and outputs must be strings")
            if bundle["kind"] == "functional":
                try:
                    for case in cases:
                        for line in case["input"].splitlines():
                            json.loads(line)
                        json.loads(case["output"])
                except (ValueError, RecursionError):
                    raise ConfigError(
                        "functional tests require JSON arguments and results"
                    ) from None
        self.config = config
        self.task = task
        self._sandbox: SubprocessSandbox | None = None
        self._graders: set[SubprocessSandbox] = set()
        indices = list(range(len(bundle["tests"])))
        if config.max_tests is not None and len(indices) > config.max_tests:
            indices = sorted(random.Random(task.task_id).sample(indices, config.max_tests))
            logger.info(
                "code_fn task %s: subsampled %d/%d tests; indices=%s",
                task.task_id,
                len(indices),
                len(bundle["tests"]),
                indices,
            )
        self._test_indices = tuple(indices)

    def _new_sandbox(self) -> SubprocessSandbox:
        memory_bytes = self.config.memory_mb * 1024**2
        return SubprocessSandbox(
            limits=ResourceLimits(address_space_bytes=memory_bytes),
            max_memory_bytes=memory_bytes,
        )

    async def setup(self) -> None:
        if self._sandbox is not None:
            return
        sandbox = self._new_sandbox()
        try:
            await sandbox.start()
            starter = self.task.meta.get("starter_code") or ""
            problem = self.task.prompt
            if starter:
                problem += f"\n\nStarter code:\n```python\n{starter}\n```\n"
            await sandbox.write_file("problem.md", problem)
            await sandbox.write_file("solution.py", starter)
            # Empty examples directories are still part of the workspace contract.
            (sandbox.workdir / "examples").mkdir(mode=0o755)
            if sandbox.uid is not None:
                os.chown(sandbox.workdir / "examples", sandbox.uid, sandbox.uid)
            for i, test in enumerate(self.task.answer["public"]):
                await sandbox.write_file(f"examples/{i:02d}.in", test["input"])
                await sandbox.write_file(f"examples/{i:02d}.out", test["output"])
        except BaseException:
            await sandbox.close()
            raise
        self._sandbox = sandbox

    async def teardown(self) -> None:
        for grader in tuple(self._graders):
            await grader.close()
            self._graders.discard(grader)
        if self._sandbox is not None:
            await self._sandbox.close()
            self._sandbox = None

    @property
    def sandbox(self) -> SubprocessSandbox | None:
        return self._sandbox

    def task_message(self, role: str) -> str:
        interface = (
            f"Functional task: define {self.task.answer['fn_name']} "
            "as a function or Solution method."
            if self.task.answer["kind"] == "functional"
            else "Stdin task: read standard input and print the answer."
        )
        return f"{self.task.prompt}\n\n{interface}\n\n{self.config.instruction}"

    def tools(self, role: str) -> list[Tool]:
        return [_BashTool(self.config.bash_timeout_s)]

    async def grade(self, submission: str | None) -> dict[str, float]:
        """Count all selected tests even after failure; compiled means syntax-valid.

        Missing/unreadable/invalid source fails the whole selected suite. n_tests
        records the suite size, including tests rejected by a compilation failure.
        """
        if self._sandbox is None:
            raise RuntimeError("code_fn requires setup before grading")
        grades = {
            "pass_all": 0.0,
            "pass_frac": 0.0,
            "compiled": 0.0,
            "n_tests": float(len(self._test_indices)),
            "timeouts": 0.0,
            "correct": 0.0,
        }
        try:
            solution = await self._sandbox.read_file("solution.py")
        except (OSError, ValueError, UnicodeError):
            return grades
        grader = self._new_sandbox()
        self._graders.add(grader)
        try:
            await grader.start()
            await grader.write_file("solution.py", solution)
            compiled = await grader.exec(
                ["python3", "-I", "-S", "-c", _COMPILE],
                timeout_s=self.config.timeout_per_test_s,
            )
            if compiled.exit_code != 0 or compiled.timed_out:
                return grades
            grades["compiled"] = 1.0
            passed = 0
            for index in self._test_indices:
                await grader.reset()
                await grader.write_file("solution.py", solution)
                test = self.task.answer["tests"][index]
                if self.task.answer["kind"] == "functional":
                    await grader.write_file("harness.py", _FUNCTIONAL_HARNESS)
                    command = ["python3", "-I", "-S", "harness.py", self.task.answer["fn_name"]]
                else:
                    command = ["python3", "-I", "-S", "solution.py"]
                result = await grader.exec(
                    command, timeout_s=self.config.timeout_per_test_s, stdin=test["input"]
                )
                grades["timeouts"] += float(result.timed_out)
                if result.timed_out or result.exit_code != 0 or result.truncated:
                    continue
                if self.task.answer["kind"] == "stdin":
                    matches = _stdout_lines(result.stdout) == _stdout_lines(test["output"])
                else:
                    try:
                        matches = json.loads(result.stdout) == json.loads(test["output"])
                    except (ValueError, RecursionError):
                        matches = False
                passed += int(matches)
            grades["pass_frac"] = passed / len(self._test_indices)
            grades["pass_all"] = grades["correct"] = float(passed == len(self._test_indices))
            return grades
        finally:
            await grader.close()
            self._graders.discard(grader)

    def canonical(self, submission: str | None) -> None:
        return None

    async def same_answer(self, a: str | None, b: str | None) -> bool:
        raise ConfigError("vote aggregation is undefined for code_fn")
