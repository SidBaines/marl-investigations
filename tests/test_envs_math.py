"""Math environments use the shared verifier and an episode-local Python tool."""

from __future__ import annotations

import json
import re
import sys
import types
from typing import Any

import pytest
from test_envs_sandbox import sandbox_host, sandbox_python  # noqa: F401

from marli.envs import math as math_env
from marli.envs.base import ENVS, Task
from marli.envs.math import MathEnv, MathEnvConfig, shared_verifier
from marli.envs.registry import make_env
from marli.envs.sandbox.base import ExecResult
from marli.interact.tools import ToolCtx, run_tool


class FakeVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str, str]] = []

    async def verify(self, pred: str | None, gold: str, answer_format: str) -> bool:
        self.calls.append((pred, gold, answer_format))
        return pred is not None and pred.strip() == gold.strip()


def fake_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return matches[-1] if matches else None


def fake_normalize(text: str) -> str:
    return "".join(text.split()).lower()


@pytest.fixture(autouse=True)
def fake_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("marli.tasks.verifiers")
    module.MathVerifier = FakeVerifier
    module.extract_boxed = fake_boxed
    module.normalize_answer = fake_normalize
    monkeypatch.setitem(sys.modules, "marli.tasks.verifiers", module)
    monkeypatch.setattr(math_env, "_VERIFIER", None)


@pytest.fixture
def task() -> Task:
    return Task("toy", "What is 6 times 7?", 42, {"answer_format": "integer"})


def context(sandbox: Any = None) -> ToolCtx:
    return ToolCtx("peer0", "peer", None, 0, None, sandbox, None, None, None)


def test_task_message_is_role_independent(task: Task) -> None:
    env = MathEnv({}, task)
    expected = f"{task.prompt}\n\n{MathEnvConfig().instruction}"
    assert env.task_message("peer") == expected
    assert env.task_message("worker") == expected
    assert (
        MathEnv({"instruction": "Answer now."}, task).task_message("peer").endswith("Answer now.")
    )


def test_registry_and_config(task: Task) -> None:
    assert ENVS.get("math") is MathEnv
    assert isinstance(make_env("math", {}, task), MathEnv)
    with pytest.raises(ValueError, match="unknown"):
        make_env("math", {"pythno_tool": True}, task)
    with pytest.raises(ValueError, match="python_timeout_s"):
        MathEnvConfig(python_timeout_s=0)


def test_tools(task: Task) -> None:
    assert MathEnv({}, task).tools("peer") == []
    (tool,) = MathEnv({"python_tool": True}, task).tools("worker")
    assert tool.spec.name == "python"
    assert tool.shared and not tool.control
    assert tool.spec.parameters["properties"] == {"code": {"type": "string"}}
    assert tool.spec.parameters["required"] == ["code"]


@pytest.mark.parametrize(
    ("submission", "pred", "correct", "answered"),
    [
        ("42", "42", 1.0, 1.0),
        (r"Working; \boxed{42}", "42", 1.0, 1.0),
        ("41", "41", 0.0, 1.0),
        (None, None, 0.0, 0.0),
        ("", "", 0.0, 1.0),
    ],
)
async def test_grade(
    task: Task, submission: str | None, pred: str | None, correct: float, answered: float
) -> None:
    env = MathEnv({}, task)
    assert await env.grade(submission) == {"correct": correct, "answered": answered}
    assert shared_verifier().calls == [(pred, "42", "integer")]


async def test_shared_verifier_and_format_override(task: Task) -> None:
    first = MathEnv({}, task)
    second = MathEnv({"answer_format": "latex"}, task)
    assert math_env._VERIFIER is None
    await first.grade("42")
    verifier = shared_verifier()
    await second.grade("42")
    assert verifier is shared_verifier()
    assert verifier.calls == [("42", "42", "integer"), ("42", "42", "latex")]


@pytest.mark.parametrize(
    ("submission", "canonical"),
    [
        (None, None),
        (" +0042 ", "42"),
        ("-0", "0"),
        (r"\boxed{0042}", "42"),
        (" X + Y ", "x+y"),
    ],
)
def test_canonical(task: Task, submission: str | None, canonical: str | None) -> None:
    assert MathEnv({}, task).canonical(submission) == canonical


async def test_same_answer_uses_pairwise_verifier(task: Task) -> None:
    env = MathEnv({}, task)
    assert await env.same_answer("42", " 42 ")
    assert not await env.same_answer("42", "41")
    assert not await env.same_answer(None, "42")
    assert not await env.same_answer("42", None)
    assert not await env.same_answer(None, None)
    assert shared_verifier().calls == [("42", " 42 ", "integer"), ("42", "41", "integer")]


async def test_disabled_setup_never_creates_sandbox(
    task: Task, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_sandbox() -> None:
        pytest.fail("disabled Python tool must not create a sandbox")

    monkeypatch.setattr(math_env, "SubprocessSandbox", unexpected_sandbox)
    env = MathEnv({}, task)
    await env.setup()
    await env.teardown()
    assert env.sandbox is None


async def test_python_arguments_timeout_and_output(task: Task) -> None:
    class RecordingSandbox:
        async def exec(self, cmd: list[str], *, timeout_s: float) -> ExecResult:
            assert cmd == ["python3", "-c", "pass"]
            assert timeout_s == 1.25
            return ExecResult(None, "x" * 100_000, "err", True, 1.25)

    (tool,) = MathEnv({"python_tool": True, "python_timeout_s": 1.25}, task).tools("peer")
    result = await run_tool(tool, context(RecordingSandbox()), {"code": "pass"})
    assert json.loads(result.content) == {
        "exit_code": None,
        "stdout": "x" * 100_000,
        "stderr": "err",
        "timed_out": True,
        "duration_s": 1.25,
    }
    assert (await run_tool(tool, context(), {"code": "pass"})).error
    assert (await run_tool(tool, context(), {"code": 4})).error


async def test_python_real_sandbox_lifecycle(task: Task, sandbox_host: Any) -> None:  # noqa: F811
    env = MathEnv({"python_tool": True}, task)
    assert env.sandbox is None
    await env.setup()
    sandbox = env.sandbox
    assert sandbox is not None
    workdir = sandbox.workdir
    try:
        await env.setup()
        assert env.sandbox is sandbox
        (tool,) = env.tools("peer")
        result = await run_tool(
            tool,
            context(sandbox),
            {
                "code": "from pathlib import Path; Path('answer').write_text('42'); print(6 * 7)",
            },
        )
        content = json.loads(result.content)
        assert content["exit_code"] == 0, content["stderr"]
        assert content["stdout"] == "42\n"
        assert await sandbox.read_file("answer") == "42"
        result = await run_tool(
            tool,
            context(sandbox),
            {
                "code": "from pathlib import Path; print(Path('answer').read_text())",
            },
        )
        assert json.loads(result.content)["stdout"] == "42\n"
    finally:
        await env.teardown()
    await env.teardown()
    assert env.sandbox is None
    assert not workdir.exists()
