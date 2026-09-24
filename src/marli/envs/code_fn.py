"""Save solution source at submission so isolated grading is reproducible offline.

The submit tool snapshots solution.py; grading sees only that source and the
task's tests, never the agent workspace. Each test starts from the same saved
source in a fresh grader uid/workdir so candidate invocations cannot alter later
tests. Inputs arrive on stdin; expected outputs never enter either sandbox.

The grader provides the LiveCodeBench standard-library prelude, but numpy is
not available. Swarm peers share one sandbox and one solution.py; independent
code sandboxes for the N-independent baseline are future work.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, DecimalException
from typing import Any

from marli.envs.base import Env, Task
from marli.envs.registry import ENVS
from marli.envs.sandbox.subprocess import ResourceLimits, SubprocessSandbox
from marli.errors import ConfigError
from marli.interact.tools import Tool, ToolCtx, ToolError, ToolResult, validate_tool_spec
from marli.render.base import ToolSpec

logger = logging.getLogger(__name__)

_MAX_SOURCE_BYTES = 1 << 20
_LCB_PRELUDE = """\
from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from math import *
from builtins import *
import math, string, re, sys, random, copy, operator
import collections, itertools, functools, heapq, bisect
sys.setrecursionlimit(600000)
"""

# Execute the prelude before the source without shifting tracebacks or breaking
# a solution's __future__ imports (which must remain at the start of its module).
_STDIN_HARNESS = (
    f"exec({_LCB_PRELUDE!r}); import runpy; "
    "runpy.run_path('solution.py', init_globals=globals(), run_name='__main__')"
)

_FUNCTIONAL_HARNESS = """\
from __future__ import annotations
import contextlib
import importlib.util
import json
import os
import sys

args = [json.loads(line) for line in sys.stdin.read().splitlines()]
with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
    spec = importlib.util.spec_from_file_location("solution", "solution.py")
    module = importlib.util.module_from_spec(spec)
    exec(PRELUDE, module.__dict__)
    spec.loader.exec_module(module)
    target = (module.Solution() if hasattr(module, "Solution")
              and hasattr(module.Solution, sys.argv[1]) else module)
    result = getattr(target, sys.argv[1])(*args)
print(json.dumps(result, ensure_ascii=False))
""".replace("PRELUDE", repr(_LCB_PRELUDE))

_COMPILE = (
    "import sys, json; print(json.dumps(list(sys.version_info[:3])), flush=True); "
    "from pathlib import Path; compile(Path('solution.py').read_text(), 'solution.py', 'exec')"
)


@dataclass(frozen=True)
class CodeFnEnvConfig:
    """Eval preserves pass_frac by default; training should set stop_on_first_failure.

    float_tol opts into an absolute numeric tolerance for training only. Skipped
    tests (early stop or max_grade_s, including sandbox startup) count as failed.
    Cleanup may finish after the wall budget to preserve sandbox isolation.
    """

    timeout_per_test_s: float = 6.0
    max_tests: int | None = None
    float_tol: float | None = None
    stop_on_first_failure: bool = False
    max_grade_s: float = 300.0
    bash_timeout_s: float = 30.0
    instruction: str = (
        "The shared workspace contains problem.md and solution.py. "
        "Write your Python solution in solution.py: read stdin and "
        "print stdout for stdin tasks, or define the named function (or Solution method) "
        "for functional tasks. The grader provides standard-library imports; "
        "numpy is not available. "
        "When done, call submit() to save solution.py as your final submission. "
        "Only that saved source is graded; an optional answer note is ignored."
    )
    memory_mb: int = 2048

    def __post_init__(self) -> None:
        for name in ("timeout_per_test_s", "bash_timeout_s", "max_grade_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_tests is not None and (type(self.max_tests) is not int or self.max_tests <= 0):
            raise ValueError("max_tests must be a positive integer or None")
        if type(self.memory_mb) is not int or self.memory_mb <= 0:
            raise ValueError("memory_mb must be a positive integer")
        if type(self.stop_on_first_failure) is not bool:
            raise ValueError("stop_on_first_failure must be a boolean")
        if self.float_tol is not None and (
            type(self.float_tol) not in (int, float)
            or not math.isfinite(self.float_tol)
            or self.float_tol < 0
        ):
            raise ValueError("float_tol must be nonnegative and finite or None")


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


class _SubmitTool:
    shared = True
    control = True
    spec = ToolSpec(
        "submit",
        "Save solution.py as your final solution source. An optional answer note is ignored.",
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        validate_tool_spec(self.spec)

    async def __call__(self, ctx: ToolCtx, *, answer: str = "") -> ToolResult:
        if ctx.sandbox is None:
            raise ToolError("submit requires an active episode sandbox")
        try:
            source = await ctx.sandbox.read_file("solution.py")
        except (OSError, ValueError, UnicodeError) as exc:
            raise ToolError("submit could not read solution.py") from exc
        if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
            raise ToolError("solution.py exceeds the 1 MiB submission limit; shorten it and retry")
        return ToolResult("submitted solution.py", control={"submit": source})


def _stdout_lines(text: str) -> list[str]:
    return [line.strip() for line in text.strip().split("\n")]


def _stdio_matches(actual: str, expected: str, float_tol: float | None) -> bool:
    """LCB grade_stdio: exact lines, then Decimal tokens for differing lines."""
    actual_lines, expected_lines = _stdout_lines(actual), _stdout_lines(expected)
    if len(actual_lines) != len(expected_lines):
        return False
    tolerance = Decimal(str(float_tol)) if float_tol is not None else None
    for actual_line, expected_line in zip(actual_lines, expected_lines, strict=True):
        if actual_line == expected_line:
            continue
        try:
            actual_tokens = [Decimal(token) for token in actual_line.split()]
            expected_tokens = [Decimal(token) for token in expected_line.split()]
            if len(actual_tokens) != len(expected_tokens):
                return False
            for a, b in zip(actual_tokens, expected_tokens, strict=True):
                if a == b:
                    continue
                if tolerance is None or not (a.is_finite() and b.is_finite()):
                    return False
                if abs(a - b) > tolerance:
                    return False
        except DecimalException:
            return False
    return True


@ENVS.register("code_fn")
class CodeFnEnv(Env):
    name = "code_fn"
    supports_vote = False

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
        self._grade_cache: dict[str, dict[str, float]] = {}
        self._grade_lock = asyncio.Lock()
        self._grader_output_bytes = max(
            1 << 20,
            2 * max(len(test["output"].encode()) for test in bundle["tests"]) + (64 << 10),
        )
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
        examples = (
            "\n\nUse bash to test with the public examples in examples/NN.in and examples/NN.out."
            if self.task.answer["public"]
            else ""
        )
        return f"{self.task.prompt}\n\n{interface}{examples}\n\n{self.config.instruction}"

    def tools(self, role: str) -> list[Tool]:
        return [_BashTool(self.config.bash_timeout_s), _SubmitTool()]

    async def grade(self, submission: str | None) -> dict[str, float]:
        """Memoize saved source within this env; never inspect the agent workspace.

        n_tests includes skipped tests, which count as failed. Always stop after
        a timeout; grade_truncated marks an early stop with unrun tests or a wall
        budget expiry. Python version components describe the grader interpreter
        (zero when it did not start). Grading needs no setup.
        """
        if submission is None:
            return await self._grade_source(None)
        digest = hashlib.sha256(submission.encode()).hexdigest()
        async with self._grade_lock:
            if digest not in self._grade_cache:
                self._grade_cache[digest] = await self._grade_source(submission)
            return self._grade_cache[digest].copy()

    async def _grade_source(self, submission: str | None) -> dict[str, float]:
        grades = {
            "pass_all": 0.0,
            "pass_frac": 0.0,
            "compiled": 0.0,
            "n_tests": float(len(self._test_indices)),
            "timeouts": 0.0,
            "correct": 0.0,
            "grade_truncated": 0.0,
            "python_major": 0.0,
            "python_minor": 0.0,
            "python_micro": 0.0,
        }
        if submission is None:
            return grades
        grader = self._new_sandbox()
        grader.max_output_bytes = self._grader_output_bytes
        self._graders.add(grader)
        passed = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.max_grade_s
        try:
            async with asyncio.timeout_at(deadline):
                await grader.start()
                await grader.write_file("solution.py", submission)
                # Discard fd 2 before Python starts; only stdout can hit its cap.
                python = [
                    "bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    'exec "$@" 2>/dev/null',
                    "grader",
                    "python3",
                    "-I",
                ]
                compiled = await grader.exec(
                    [*python, "-c", _COMPILE],
                    timeout_s=self.config.timeout_per_test_s,
                )
                if compiled.stdout.strip():
                    for name, version in zip(
                        ("python_major", "python_minor", "python_micro"),
                        json.loads(compiled.stdout),
                        strict=True,
                    ):
                        grades[name] = float(version)
                if compiled.timed_out:
                    grades["timeouts"] = grades["grade_truncated"] = 1.0
                    return grades
                if compiled.exit_code != 0:
                    return grades
                grades["compiled"] = 1.0
                for position, index in enumerate(self._test_indices):
                    await grader.reset()
                    await grader.write_file("solution.py", submission)
                    test = self.task.answer["tests"][index]
                    if self.task.answer["kind"] == "functional":
                        await grader.write_file("harness.py", _FUNCTIONAL_HARNESS)
                        command = [*python, "harness.py", self.task.answer["fn_name"]]
                    else:
                        command = [*python, "-c", _STDIN_HARNESS]
                    result = await grader.exec(
                        command, timeout_s=self.config.timeout_per_test_s, stdin=test["input"]
                    )
                    grades["timeouts"] += float(result.timed_out)
                    matches = False
                    if not (result.timed_out or result.exit_code != 0 or result.truncated):
                        if self.task.answer["kind"] == "stdin":
                            matches = _stdio_matches(
                                result.stdout, test["output"], self.config.float_tol
                            )
                        else:
                            try:
                                matches = json.loads(result.stdout) == json.loads(test["output"])
                            except (ValueError, RecursionError):
                                matches = False
                    # Comparisons are synchronous, so check the deadline before
                    # awarding credit even when the last test needs no more awaits.
                    if loop.time() >= deadline:
                        grades["grade_truncated"] = 1.0
                        break
                    passed += int(matches)
                    if result.timed_out or (not matches and self.config.stop_on_first_failure):
                        grades["grade_truncated"] = float(position + 1 < len(self._test_indices))
                        break
        except TimeoutError:
            grades["grade_truncated"] = 1.0
        finally:
            await grader.close()
            self._graders.discard(grader)
        grades["pass_frac"] = passed / len(self._test_indices)
        grades["pass_all"] = grades["correct"] = float(passed == len(self._test_indices))
        return grades

    def canonical(self, submission: str | None) -> None:
        return None

    async def same_answer(self, a: str | None, b: str | None) -> bool:
        raise ConfigError("vote aggregation is undefined for code_fn")
