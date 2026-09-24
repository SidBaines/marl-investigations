"""Exercise grading through the real sandbox and the normal token/tool runtime."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest
from _marli_code_fixtures import SUM_SOLUTION, code_task
from test_envs_sandbox import sandbox_host  # noqa: F401

from marli.envs.code_fn import CodeFnEnv, CodeFnEnvConfig
from marli.envs.registry import make_env
from marli.envs.sandbox.base import ExecResult
from marli.envs.sandbox.subprocess import SubprocessSandbox
from marli.errors import ConfigError
from marli.interact.limits import Limits
from marli.interact.protocols.single import SingleConfig, SingleProtocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.tools import ToolCtx, run_tool
from marli.policy.scripted import ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import FakeRenderer


def context(sandbox: Any = None) -> ToolCtx:
    return ToolCtx("worker0", "worker", None, 0, None, sandbox, None, None, None)


def test_registry_config_and_votes() -> None:
    env = make_env("code_fn", {}, code_task())
    assert isinstance(env, CodeFnEnv)
    assert env.config == CodeFnEnvConfig()
    assert env.canonical("done") is None
    assert env.task_message("solver") == env.task_message("worker")
    assert "solution.py" in env.task_message("worker")
    assert "4 5" not in env.task_message("worker")
    assert "Functional task: define add" in CodeFnEnv({}, code_task(functional=True)).task_message(
        "solver"
    )
    assert (
        CodeFnEnv({"instruction": "Finish."}, code_task())
        .task_message("solver")
        .endswith("Finish.")
    )
    with pytest.raises(ValueError, match="unknown"):
        make_env("code_fn", {"typo": 3}, code_task())


@pytest.mark.parametrize("a,b", [(None, None), ("same", "same"), ("a", "b")])
async def test_vote_equivalence_errors(a: str | None, b: str | None) -> None:
    with pytest.raises(ConfigError, match="vote"):
        await CodeFnEnv({}, code_task()).same_answer(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_per_test_s": 0},
        {"timeout_per_test_s": float("nan")},
        {"timeout_per_test_s": "6"},
        {"bash_timeout_s": -1},
        {"bash_timeout_s": float("inf")},
        {"max_tests": 0},
        {"max_tests": -1},
        {"max_tests": True},
        {"max_tests": 1.5},
        {"memory_mb": 0},
        {"memory_mb": True},
        {"memory_mb": 2.5},
    ],
)
def test_invalid_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CodeFnEnvConfig(**kwargs)


@pytest.mark.parametrize(
    "answer",
    [
        None,
        "5",
        {"kind": "unknown"},
        {"kind": "stdin", "tests": [], "public": []},
        {"kind": "functional", "tests": [{"input": "1", "output": "1"}], "public": []},
    ],
)
def test_invalid_bundle_fails_before_setup(answer: Any) -> None:
    with pytest.raises(ConfigError):
        CodeFnEnv({}, replace(code_task(), answer=answer))


def test_malformed_functional_reference_rejected_before_setup() -> None:
    task = code_task(functional=True)
    task.answer["tests"][0]["output"] = "not JSON"
    with pytest.raises(ConfigError, match="JSON"):
        CodeFnEnv({}, task)


def test_subsampling_is_deterministic_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="marli.envs.code_fn"):
        env = CodeFnEnv({"max_tests": 2}, code_task())
    assert env._test_indices == CodeFnEnv({"max_tests": 2}, code_task())._test_indices
    assert len(set(env._test_indices)) == 2
    assert "subsampled 2/3 tests" in caplog.text
    assert CodeFnEnv({"max_tests": 9}, code_task())._test_indices == (0, 1, 2)


async def test_bash_tool_schema_timeout_and_errors() -> None:
    (tool,) = CodeFnEnv({"bash_timeout_s": 1.25}, code_task()).tools("worker")
    assert tool.spec.name == "bash" and tool.shared and not tool.control

    class RecordingSandbox:
        async def exec(self, cmd: str, *, timeout_s: float) -> ExecResult:
            assert cmd == "echo hi" and timeout_s == 1.25
            return ExecResult(0, "hi\n", "", False, 0.1)

    result = await run_tool(tool, context(RecordingSandbox()), {"command": "echo hi"})
    assert json.loads(result.content)["stdout"] == "hi\n"
    assert (await run_tool(tool, context(), {"command": "echo hi"})).error
    assert (await run_tool(tool, context(), {"command": 3})).error

    class BrokenSandbox:
        async def exec(self, cmd: str, *, timeout_s: float) -> ExecResult:
            raise OSError("unavailable")

    assert (await run_tool(tool, context(BrokenSandbox()), {"command": "true"})).error


@pytest.mark.usefixtures("sandbox_host")
async def test_setup_only_public_files_and_teardown() -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    sandbox = env.sandbox
    assert sandbox is not None
    path = sandbox.workdir
    try:
        await env.setup()
        assert env.sandbox is sandbox
        assert await sandbox.read_file("problem.md") == (
            env.task.prompt + "\n\nStarter code:\n```python\n# Start here\n\n```\n"
        )
        assert await sandbox.read_file("solution.py") == "# Start here\n"
        assert await sandbox.read_file("examples/00.in") == "2 3\n"
        assert await sandbox.read_file("examples/00.out") == "5"
        assert sorted(p.name for p in (path / "examples").iterdir()) == ["00.in", "00.out"]
        (tool,) = env.tools("worker")
        result = await run_tool(tool, context(sandbox), {"command": "pwd; cat examples/00.in"})
        assert json.loads(result.content)["stdout"] == f"{path}\n2 3\n"
    finally:
        await env.teardown()
    await env.teardown()
    assert not path.exists() and env.sandbox is None


@pytest.mark.usefixtures("sandbox_host")
async def test_scripted_policy_full_loop_writes_tests_and_submits() -> None:
    renderer = FakeRenderer()
    task = code_task()
    env = CodeFnEnv({}, task)
    script = turns_by_agent(
        renderer,
        {
            "solver0": [
                Turn(
                    tool_calls=(
                        (
                            "bash",
                            {
                                "command": f"cat > solution.py <<'PY'\n{SUM_SOLUTION}PY\n"
                                "python3 solution.py < examples/00.in"
                            },
                        ),
                    )
                ),
                Turn(tool_calls=(("submit", {"answer": "Implemented and tested."}),)),
            ]
        },
    )
    policy = ScriptedPolicy("script", renderer, script)
    spec = EpisodeSpec(
        SingleProtocol(SingleConfig(env_tools=("bash",))),
        env,
        task,
        {"solver": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        Limits(),
        config_hash="synthetic-code",
        backend="scripted",
    )
    episode, buffers = await run_episode(spec)
    assert episode.ok and not episode.errors
    assert episode.outcome.final_answer == "Implemented and tested."
    expected = {
        "pass_all": 1.0,
        "pass_frac": 1.0,
        "compiled": 1.0,
        "n_tests": 3.0,
        "timeouts": 0.0,
        "correct": 1.0,
    }
    assert episode.grades == {"solver0": expected, "_system": expected}
    assert len(episode.calls) == 2 and buffers
    assert env.sandbox is None


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "solution,fraction,compiled,timeouts",
    [
        (SUM_SOLUTION, 1.0, 1.0, 0.0),
        ("print(9)\n", 1 / 3, 1.0, 0.0),  # First test fails; later success still counts.
        ("this is invalid python!\n", 0.0, 0.0, 0.0),
        ("raise RuntimeError('no')\n", 0.0, 1.0, 0.0),
        ("while True: pass\n", 0.0, 1.0, 3.0),
        ("print('{\"pass_all\": 1}')\n", 0.0, 1.0, 0.0),
    ],
)
async def test_grading_components(
    solution: str, fraction: float, compiled: float, timeouts: float
) -> None:
    env = CodeFnEnv({"timeout_per_test_s": 0.6}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", solution)
        grade = await env.grade(SUM_SOLUTION)
        assert grade == {
            "pass_all": float(fraction == 1),
            "correct": float(fraction == 1),
            "pass_frac": fraction,
            "compiled": compiled,
            "n_tests": 3.0,
            "timeouts": timeouts,
        }
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_grade_ignores_note_and_uses_root_side_read(monkeypatch: pytest.MonkeyPatch) -> None:
    env = CodeFnEnv({"max_tests": 1}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", SUM_SOLUTION)
        real_read = env.sandbox.read_file
        reads = []

        async def read(rel: str) -> str:
            reads.append(rel)
            return await real_read(rel)

        monkeypatch.setattr(env.sandbox, "read_file", read)
        for note in (None, "", "wrong answer", "pass_all=0"):
            grade = await env.grade(note)
            assert grade["pass_all"] == 1 and grade["n_tests"] == 1
        assert reads == ["solution.py"] * 4
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "solution",
    [
        "def add(a, b):\n    return a + b\n",
        "class Solution:\n    def add(self, a, b):\n        print('debug')\n        return a + b\n",
    ],
)
async def test_functional_and_leetcode_method(solution: str) -> None:
    env = CodeFnEnv({}, code_task(functional=True))
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", solution)
        assert (await env.grade("done"))["pass_all"] == 1
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "output,passed",
    [
        ("  1\t  2 \r\n 3   \n\n", 1),
        ("1 2\n3", 1),
        ("1 2\n\n 3", 1),
        ("1 2 3\n", 0),
        ("1 2\n4", 0),
    ],
)
async def test_stdout_whitespace_normalization(output: str, passed: int) -> None:
    task = code_task()
    task.answer["tests"] = [{"input": "", "output": "1 2\n3\n"}]
    env = CodeFnEnv({}, task)
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", f"print({output!r}, end='')")
        assert (await env.grade("done"))["pass_all"] == passed
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_fresh_grader_uid_and_no_hidden_tests_or_host_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    assert env.sandbox is not None
    agent = env.sandbox
    records = []
    real_start = SubprocessSandbox.start

    async def record_start(sandbox: SubprocessSandbox) -> None:
        await real_start(sandbox)
        records.append((sandbox.uid, sandbox.workdir))
        assert sandbox.uid != agent.uid and sandbox.workdir != agent.workdir

    monkeypatch.setattr(SubprocessSandbox, "start", record_start)
    solution = f"""
import os
for path in ('hidden_tests.json', 'tests.json', 'examples/00.out',
             '/etc/rp_environment', '/etc/shadow', {str(agent.workdir / "private")!r}):
    try:
        fd = os.open(path, os.O_RDONLY)
    except (PermissionError, FileNotFoundError):
        pass
    else:
        os.close(fd)
        raise AssertionError('unexpected read access')
{SUM_SOLUTION}
"""
    try:
        await agent.write_file("private", "workspace state must not enter the grader")
        await agent.write_file("solution.py", solution)
        assert (await env.grade("done"))["pass_all"] == 1
        assert len(records) == 1 and not records[0][1].exists()
        assert await agent.read_file("private") == "workspace state must not enter the grader"
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_reset_prevents_cross_test_tampering() -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file(
            "solution.py",
            (
                "from pathlib import Path\n"
                "assert not Path('previous-test').exists()\n"
                "Path('previous-test').write_text('state')\n"
                "Path('solution.py').write_text('raise RuntimeError()')\n" + SUM_SOLUTION
            ),
        )
        assert (await env.grade("done"))["pass_all"] == 1
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_missing_and_symlink_solution_fail_closed() -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.exec("rm solution.py", timeout_s=3)
        assert (await env.grade("done"))["compiled"] == 0
        await env.sandbox.exec("ln -s /etc/rp_environment solution.py", timeout_s=3)
        assert (await env.grade("done"))["compiled"] == 0
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_cancel_grade_closes_grader() -> None:
    env = CodeFnEnv({"timeout_per_test_s": 10}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", "while True: pass")
        running = asyncio.create_task(env.grade("done"))
        for _ in range(200):
            if env._graders and all(hasattr(g, "workdir") for g in env._graders):
                break
            await asyncio.sleep(0.01)
        paths = [g.workdir for g in env._graders]
        assert paths
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not env._graders and all(not path.exists() for path in paths)
    finally:
        await env.teardown()


def test_memory_limit_applies_to_both_rlimit_and_rss() -> None:
    sandbox = CodeFnEnv({"memory_mb": 128}, code_task())._new_sandbox()
    assert sandbox.limits.address_space_bytes == sandbox.max_memory_bytes == 128 * 1024**2
