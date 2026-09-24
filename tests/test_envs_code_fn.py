"""Exercise grading through the real sandbox and the normal token/tool runtime."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _marli_code_fixtures import SUM_SOLUTION, code_task
from test_envs_sandbox import sandbox_host  # noqa: F401

from marli.envs.code_fn import CodeFnEnv, CodeFnEnvConfig, _stdio_matches
from marli.envs.registry import make_env
from marli.envs.sandbox.base import ExecResult
from marli.envs.sandbox.subprocess import SubprocessSandbox
from marli.errors import ConfigError
from marli.eval.policies import PolicySpec
from marli.eval.rollout import RolloutConfig
from marli.eval.score import ScoreConfig
from marli.eval.store import read_episodes
from marli.interact.limits import Limits
from marli.interact.protocols.single import SingleConfig, SingleProtocol
from marli.interact.protocols.swarm import SwarmConfig, SwarmProtocol
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.system import SystemIO
from marli.interact.tools import ToolCtx, run_tool
from marli.interact.types import Outcome
from marli.policy.scripted import ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import FakeRenderer
from marli.tasks.taskset import TaskSet, write_tasks
from marli.verbs import run_verb


def context(sandbox: Any = None) -> ToolCtx:
    return ToolCtx("worker0", "worker", None, 0, None, sandbox, None, None, None)


def test_registry_config_and_votes() -> None:
    env = make_env("code_fn", {}, code_task())
    assert isinstance(env, CodeFnEnv)
    assert not env.supports_vote
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
        {"max_grade_s": 0},
        {"max_grade_s": float("inf")},
        {"max_grade_s": True},
        {"stop_on_first_failure": 1},
        {"float_tol": -1},
        {"float_tol": float("nan")},
        {"float_tol": float("inf")},
        {"float_tol": True},
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
    tool = CodeFnEnv({"bash_timeout_s": 1.25}, code_task()).tools("worker")[0]
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
        tool = env.tools("worker")[0]
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
    assert episode.outcome.final_answer == SUM_SOLUTION
    expected = {
        "pass_all": 1.0,
        "pass_frac": 1.0,
        "compiled": 1.0,
        "n_tests": 3.0,
        "timeouts": 0.0,
        "correct": 1.0,
    }
    assert episode.grades["solver0"] == episode.grades["_system"]
    assert episode.grades["_system"].items() >= expected.items()
    assert episode.grades["_system"]["grade_truncated"] == 0
    assert episode.grades["_system"]["python_major"] == 3
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
        ("while True: pass\n", 0.0, 1.0, 1.0),
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
        grade = await env.grade(solution)
        assert (
            grade.items()
            >= {
                "pass_all": float(fraction == 1),
                "correct": float(fraction == 1),
                "pass_frac": fraction,
                "compiled": compiled,
                "n_tests": 3.0,
                "timeouts": timeouts,
            }.items()
        )
        assert grade["grade_truncated"] == float(timeouts > 0)
        assert grade["python_major"] == 3
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_submit_snapshots_source_and_grade_never_reads_agent_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        source = SUM_SOLUTION + "# α comment\n" * 2000
        await env.sandbox.write_file("solution.py", source)
        tool = env.tools("solver")[1]
        assert tool.spec.name == "submit" and tool.shared and tool.control
        for arguments in ({}, {"answer": "This note is not source."}):
            result = await run_tool(tool, context(env.sandbox), arguments)
            assert result.control == {"submit": source}
        await env.sandbox.write_file("solution.py", "raise RuntimeError('changed')")

        async def forbidden_read(rel: str) -> str:
            pytest.fail("grading read the agent workspace")

        monkeypatch.setattr(env.sandbox, "read_file", forbidden_read)
        assert (await env.grade(source))["pass_all"] == 1
        assert (await env.grade(None))["compiled"] == 0
    finally:
        await env.teardown()
    # Regrading works even after teardown, without ever setting up an agent sandbox.
    assert (await CodeFnEnv({}, code_task()).grade(source))["pass_all"] == 1


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "solution",
    [
        "def add(a, b):\n    return a + b\n",
        "class Solution:\n    def add(self, a, b):\n        print('debug')\n        return a + b\n",
        "class Solution: pass\ndef add(a, b):\n    return a + b\n",
    ],
)
async def test_functional_and_leetcode_method(solution: str) -> None:
    env = CodeFnEnv({}, code_task(functional=True))
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", solution)
        assert (await env.grade(solution))["pass_all"] == 1
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "output,passed",
    [
        ("  1\t  2 \r\n 3   \n\n", 1),
        ("1 2\n3", 1),
        ("1 2\n\n 3", 0),
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
        solution = f"print({output!r}, end='')"
        assert (await env.grade(solution))["pass_all"] == passed
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
        assert (await env.grade(solution))["pass_all"] == 1
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
        solution = (
            "from pathlib import Path\n"
            "assert not Path('previous-test').exists()\n"
            "Path('previous-test').write_text('state')\n"
            "Path('solution.py').write_text('raise RuntimeError()')\n" + SUM_SOLUTION
        )
        assert (await env.grade(solution))["pass_all"] == 1
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_submit_missing_and_symlink_solution_fail_closed() -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        tool = env.tools("solver")[1]
        await env.sandbox.exec("rm solution.py", timeout_s=3)
        assert (await run_tool(tool, context(env.sandbox), {})).error
        await env.sandbox.exec("ln -s /etc/rp_environment solution.py", timeout_s=3)
        assert (await run_tool(tool, context(env.sandbox), {})).error
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_cancel_grade_closes_grader() -> None:
    env = CodeFnEnv({"timeout_per_test_s": 10}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        await env.sandbox.write_file("solution.py", "while True: pass")
        running = asyncio.create_task(env.grade("while True: pass"))
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


async def test_submit_requires_sandbox_and_never_exposes_read_errors() -> None:
    tool = CodeFnEnv({}, code_task()).tools("solver")[1]
    assert (await run_tool(tool, context(), {})).error

    class UnreadableSandbox:
        async def read_file(self, rel: str) -> str:
            raise UnicodeError("private content")

    result = await run_tool(tool, context(UnreadableSandbox()), {})
    assert result.error and not result.control
    assert "private content" not in result.content


@pytest.mark.usefixtures("sandbox_host")
async def test_agent_with_solution_but_no_submit_gets_zero() -> None:
    task = replace(code_task(), meta={"starter_code": SUM_SOLUTION})
    renderer = FakeRenderer()
    policy = ScriptedPolicy(
        "script",
        renderer,
        turns_by_agent(
            renderer,
            {
                "solver0": [Turn("Finished without submitting.")],
            },
        ),
    )
    spec = EpisodeSpec(
        SingleProtocol(),
        CodeFnEnv({}, task),
        task,
        {"solver": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        Limits(on_no_tool_call="end_agent"),
    )
    episode, _ = await run_episode(spec)
    assert episode.outcome.final_answer is None
    for grade in episode.grades.values():
        assert grade["compiled"] == grade["correct"] == grade["pass_all"] == grade["pass_frac"] == 0
        assert grade["n_tests"] == 3


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "submissions",
    [
        {},
        {"a": None},
        {"a": SUM_SOLUTION},
        {
            "a": SUM_SOLUTION,
            "b": SUM_SOLUTION,
        },
    ],
)
async def test_episode_system_rejects_every_code_vote(
    submissions: dict[str, str | None],
) -> None:
    class VoteProtocol(SingleProtocol):
        async def run(self, io: SystemIO) -> Outcome:
            await io.vote(submissions)
            pytest.fail("code vote was accepted")

    renderer = FakeRenderer()
    policy = ScriptedPolicy("script", renderer, turns_by_agent(renderer, {}))
    task = code_task()
    spec = EpisodeSpec(
        VoteProtocol(),
        CodeFnEnv({}, task),
        task,
        {"solver": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        Limits(),
    )
    with pytest.raises(ConfigError, match="vote.*code_fn"):
        await run_episode(spec)
    assert not policy.calls


@pytest.mark.usefixtures("sandbox_host")
async def test_code_swarm_finalizer_can_submit() -> None:
    renderer = FakeRenderer()
    task = replace(code_task(), meta={"starter_code": SUM_SOLUTION})
    policy = ScriptedPolicy(
        "script",
        renderer,
        turns_by_agent(
            renderer,
            {
                "peer0": [Turn(tool_calls=(("submit", {}),))],
                "finalizer0": [Turn(tool_calls=(("submit", {}),))],
            },
        ),
    )
    spec = EpisodeSpec(
        SwarmProtocol(SwarmConfig(n_agents=1, aggregation="finalizer")),
        CodeFnEnv({}, task),
        task,
        {"peer": "script", "finalizer": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        Limits(),
    )
    episode, _ = await run_episode(spec)
    assert episode.ok and episode.outcome.final_answer == SUM_SOLUTION
    assert episode.outcome.aggregation == "finalizer"
    assert episode.grades["_system"]["pass_all"] == 1


def sum_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy(
        "script",
        renderer,
        turns_by_agent(
            renderer,
            {
                "solver0": [
                    Turn(
                        tool_calls=(
                            ("bash", {"command": f"cat > solution.py <<'PY'\n{SUM_SOLUTION}PY\n"}),
                            ("submit", {}),
                        )
                    )
                ],
            },
        ),
    )


@pytest.mark.usefixtures("sandbox_host")
async def test_saved_code_episode_regrades_from_source(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    write_tasks(root, [code_task()])
    taskset = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="synthetic",
        split="test",
        kind="code",
        n=1,
        answer_format="tests",
        commit_text=False,
    )
    taskset.save()
    rollout = await run_verb(
        "eval rollout",
        RolloutConfig(
            tasks=str(taskset.manifest_path),
            env="code_fn",
            protocol_config={"env_tools": ["bash"]},
            policies={
                "script": PolicySpec("scripted:test_envs_code_fn:sum_policy", renderer="fake")
            },
            seating={"solver": "script"},
        ),
        out=tmp_path / "episodes",
    )
    (episode,) = list(read_episodes(rollout.handle.root))
    assert episode.outcome.final_answer == SUM_SOLUTION
    assert episode.outcome.submissions == {"solver0": SUM_SOLUTION}
    before = rollout.handle.file("episodes").read_bytes()
    result = await run_verb(
        "eval score", ScoreConfig(str(rollout.manifest), regrade=True), out=tmp_path / "regrade"
    )
    row = json.loads(result.handle.file("rows").read_text())
    assert row["correct"] == 1 and row["own_correct"] == {"solver0": 1}
    assert rollout.handle.file("episodes").read_bytes() == before


@pytest.mark.parametrize(
    "actual,expected,matches",
    [
        ("1.0", "1", True),
        ("1.50", "1.5", True),
        ("-0", "0", True),
        ("1e3", "1000", True),
        ("0.1 0.2", "0.10 0.20", True),
        ("3 4", "3  4", True),
        ("a  b", "a b", False),
        ("1\n\n2", "1\n2", False),
        ("1\n\n2", "1.0\n\n2.0", True),
        ("Yes", "yes", False),
        ("1\r\n2\r\n", "1\n2", True),
        ("\n\n5\n\n", "5", True),
        ("5 ", "5", True),
        ("", "\n", True),
        ("1 2", "1 2 3", False),
        ("NaN", "1", False),
        ("sNaN", "nan", False),
        ("100000000000000000000000000001", "100000000000000000000000000000", False),
    ],
)
def test_lcb_decimal_and_line_semantics(actual: str, expected: str, matches: bool) -> None:
    assert _stdio_matches(actual, expected, None) == matches


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize(
    "source,expected,functional",
    [
        ("import sys; sys.stderr.write('x' * 2_000_000); print('ok')", "ok", False),
        (
            "print('\\n'.join(str(i) for i in range(200000)))",
            "\n".join(str(i) for i in range(200000)),
            False,
        ),
        ("print('ok'); exit()", "ok", False),
        ("print('ok'); quit()", "ok", False),
        (
            "assert sys.flags.isolated and __file__ == 'solution.py' "
            "and __name__ == '__main__'; print('ok')",
            "ok",
            False,
        ),
        ("from __future__ import annotations\nprint('ok')", "ok", False),
        (
            "class Solution:\n    def add(self, a: List[int], b: Optional[int]):\n"
            "        return sum(a) + b",
            "6",
            True,
        ),
        ("def add(a, b):\n    print('x' * 2_000_000)\n    return sum(a) + b", "6", True),
        ("def add(a, b):\n    return tuple(a + [b])", "[1, 2, 3]", True),
        (
            "def depth(n): return 0 if n == 0 else 1 + depth(n - 1)\nprint(depth(5000))",
            "5000",
            False,
        ),
        (
            "def depth(n): return 0 if n == 0 else 1 + depth(n - 1)\n"
            "def add(a, b): return depth(5000)",
            "5000",
            True,
        ),
        ("print(Counter([1, 1])[1] + bisect_left([0, 1], 1) + factorial(3))", "9", False),
    ],
    ids=[
        "stderr",
        "large-output",
        "exit",
        "quit",
        "isolated",
        "future-import",
        "bare-annotations",
        "functional-debug",
        "tuple-json",
        "stdin-recursion",
        "functional-recursion",
        "prelude",
    ],
)
async def test_correct_solutions_with_lcb_harness(
    source: str,
    expected: str,
    functional: bool,
) -> None:
    task = code_task(functional=functional)
    task.answer["tests"] = [{"input": "[1, 2]\n3" if functional else "", "output": expected}]
    result = await CodeFnEnv({}, task).grade(source)
    assert result["pass_all"] == 1
    assert result["python_major"] == 3
    assert result["python_minor"] >= 10
    assert result["python_micro"] >= 0


@pytest.mark.usefixtures("sandbox_host")
async def test_stdout_truncation_cannot_pass_even_if_head_tail_match() -> None:
    task = code_task()
    task.answer["tests"] = [{"input": "", "output": "a" * (1 << 20)}]
    env = CodeFnEnv({}, task)
    # The expected output determines a larger cap, but output past it still fails.
    assert env._grader_output_bytes == 2 * (1 << 20) + (64 << 10)
    assert (await env.grade("print('a' * 5_000_000)"))["pass_all"] == 0


@pytest.mark.usefixtures("sandbox_host")
async def test_float_tolerance_is_opt_in_and_only_for_stdio() -> None:
    task = code_task()
    task.answer["tests"] = [{"input": "", "output": "1.5"}]
    assert (await CodeFnEnv({}, task).grade("print(1.5001)"))["pass_all"] == 0
    assert (await CodeFnEnv({"float_tol": 0.001}, task).grade("print(1.5001)"))["pass_all"] == 1
    task = code_task(functional=True)
    task.answer["tests"] = [{"input": "1\n2", "output": "1.5"}]
    result = await CodeFnEnv({"float_tol": 0.001}, task).grade("def add(a,b): return 1.5001")
    assert result["pass_all"] == 0


@pytest.mark.usefixtures("sandbox_host")
async def test_grade_memo_counts_starts_and_returns_independent_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts = 0
    original = SubprocessSandbox.start

    async def counted(sandbox: SubprocessSandbox) -> None:
        nonlocal starts
        starts += 1
        await original(sandbox)

    monkeypatch.setattr(SubprocessSandbox, "start", counted)
    env = CodeFnEnv({}, code_task())
    first, second = await asyncio.gather(env.grade(SUM_SOLUTION), env.grade(SUM_SOLUTION))
    assert first == second and starts == 1
    first["pass_all"] = 0
    assert (await env.grade(SUM_SOLUTION))["pass_all"] == 1
    assert starts == 1
    assert (await env.grade("print(9)"))["pass_frac"] == 1 / 3
    assert starts == 2
    assert (await CodeFnEnv({}, code_task()).grade(SUM_SOLUTION))["pass_all"] == 1
    assert starts == 3


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize("early_stop,fraction", [(False, 1 / 3), (True, 0)])
async def test_stop_on_first_failure_preserves_denominator(
    early_stop: bool,
    fraction: float,
) -> None:
    env = CodeFnEnv({"stop_on_first_failure": early_stop}, code_task())
    result = await env.grade("print(9)")
    assert result["pass_frac"] == fraction and result["n_tests"] == 3
    assert result["grade_truncated"] == float(early_stop)


@pytest.mark.usefixtures("sandbox_host")
async def test_timeout_stops_before_later_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    original = SubprocessSandbox.exec
    inputs: list[str] = []

    async def record(sandbox: SubprocessSandbox, cmd: Any, **kwargs: Any) -> ExecResult:
        if kwargs.get("stdin") is not None:
            inputs.append(kwargs["stdin"])
        return await original(sandbox, cmd, **kwargs)

    monkeypatch.setattr(SubprocessSandbox, "exec", record)
    result = await CodeFnEnv({"timeout_per_test_s": 0.5}, code_task()).grade("while True: pass")
    assert len(inputs) == 1
    assert result["timeouts"] == 1 and result["grade_truncated"] == 1
    assert result["pass_frac"] == 0 and result["n_tests"] == 3


@pytest.mark.usefixtures("sandbox_host")
async def test_grade_wall_budget_preserves_completed_tests_and_cleans_up() -> None:
    env = CodeFnEnv({"max_grade_s": 2, "timeout_per_test_s": 10}, code_task())
    source = "import time\na,b = map(int, input().split())\nif a == 4: time.sleep(10)\nprint(a+b)"
    started = time.monotonic()
    result = await env.grade(source)
    assert time.monotonic() - started < 4
    assert result["grade_truncated"] == 1 and result["n_tests"] == 3
    assert result["pass_frac"] == 1 / 3 and result["compiled"] == 1
    assert not env._graders


@pytest.mark.usefixtures("sandbox_host")
async def test_grade_wall_budget_includes_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    original = SubprocessSandbox.start
    paths: list[Path] = []

    async def slow_start(sandbox: SubprocessSandbox) -> None:
        await original(sandbox)
        paths.append(sandbox.workdir)
        await asyncio.sleep(10)

    monkeypatch.setattr(SubprocessSandbox, "start", slow_start)
    env = CodeFnEnv({"max_grade_s": 0.5}, code_task())
    result = await env.grade(SUM_SOLUTION)
    assert result["grade_truncated"] == 1 and result["compiled"] == 0
    assert paths and all(not path.exists() for path in paths)
    assert not env._graders


@pytest.mark.usefixtures("sandbox_host")
async def test_submit_size_limit_counts_utf8_bytes_and_allows_retry() -> None:
    env = CodeFnEnv({}, code_task())
    await env.setup()
    try:
        assert env.sandbox is not None
        tool = env.tools("solver")[1]
        limit = 1 << 20
        await env.sandbox.write_file("solution.py", "#" * limit)
        assert (await run_tool(tool, context(env.sandbox), {})).control["submit"] == "#" * limit
        await env.sandbox.write_file("solution.py", "é" * (limit // 2 + 1))
        result = await run_tool(tool, context(env.sandbox), {})
        assert result.error and "1 MiB" in result.error and not result.control
        await env.sandbox.write_file("solution.py", SUM_SOLUTION)
        assert (await run_tool(tool, context(env.sandbox), {})).control == {"submit": SUM_SOLUTION}
    finally:
        await env.teardown()


def test_no_public_examples_are_not_advertised() -> None:
    task = code_task()
    task.answer["public"] = []
    message = CodeFnEnv({}, task).task_message("solver")
    assert "examples/" not in message and "public examples" not in message
    assert "numpy is not available" in message
    assert "examples/NN.in" in CodeFnEnv({}, code_task()).task_message("solver")


@pytest.mark.usefixtures("sandbox_host")
async def test_code_consensus_is_rejected_before_any_llm_call() -> None:
    renderer = FakeRenderer()
    task = code_task()
    policy = ScriptedPolicy("script", renderer, turns_by_agent(renderer, {}))
    spec = EpisodeSpec(
        SwarmProtocol(SwarmConfig(n_agents=2, aggregation="finalizer", stop_on_consensus=2)),
        CodeFnEnv({}, task),
        task,
        {"peer": "script", "finalizer": "script"},
        {"script": policy},
        {"script": FakeRenderer},
        Limits(),
    )
    with pytest.raises(ConfigError, match="stop_on_consensus.*code_fn"):
        await run_episode(spec)
    assert not policy.calls


@pytest.mark.usefixtures("sandbox_host")
async def test_large_functional_unicode_output_fits_expected_byte_cap() -> None:
    task = code_task(functional=True)
    task.answer["tests"] = [
        {"input": "1\n2", "output": json.dumps("é" * 600_000, ensure_ascii=False)}
    ]
    result = await CodeFnEnv({}, task).grade("def add(a, b): return 'é' * 600_000")
    assert result["pass_all"] == 1


@pytest.mark.usefixtures("sandbox_host")
@pytest.mark.parametrize("functional", [False, True])
async def test_prelude_preserves_builtin_modular_pow(functional: bool) -> None:
    task = code_task(functional=functional)
    task.answer["tests"] = [{"input": "2\n10" if functional else "", "output": "2"}]
    source = "def add(a, b): return pow(a, b, 7)" if functional else "print(pow(2, 10, 7))"
    assert (await CodeFnEnv({}, task).grade(source))["pass_all"] == 1


def test_training_tolerance_rejects_decimal_arithmetic_overflow() -> None:
    assert not _stdio_matches("1e999999999", "0", 0.01)
