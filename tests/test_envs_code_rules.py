"""Pin the sacrifice payoff and keep rule disclosure out of scored CI reports."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from typing import Any

import pytest
from _marli_code_fixtures import SUM_SOLUTION, code_task
from test_envs_sandbox import sandbox_host  # noqa: F401

from marli.envs.base import Task
from marli.envs.code_rules import CodeRulesEnv, CodeRulesEnvConfig, _Rule
from marli.envs.registry import make_env
from marli.interact.tools import Tool, ToolCtx, ToolError


def repo_task(n: int = 4, *, family: str | None = None, has_rule: bool = True) -> Task:
    return Task(
        "repo-00000",
        "Synthetic shared repository.",
        {
            "kind": "code_repo",
            "problems": [asdict(replace(code_task(), task_id=f"synthetic/{i}")) for i in range(n)],
            "has_rule": has_rule,
            "rule_families": [family] if family else ["header", "constant", "docstring"],
        },
    )


def context(env: CodeRulesEnv, agent_id: str = "worker0") -> ToolCtx:
    return ToolCtx(agent_id, "worker", None, 0, None, env.sandbox, None, None, None)


def tool(env: CodeRulesEnv, name: str) -> Tool:
    return next(item for item in env.tools("worker") if item.spec.name == name)


def following(rule: _Rule) -> str:
    if rule.family == "header":
        return f"# {rule.key}: {rule.id}\n"
    if rule.family == "constant":
        return f'{rule.key} = "{rule.id}"\n'
    return f'"""@{rule.key} {rule.id}"""\n'


@pytest.fixture
async def env(sandbox_host: Any) -> AsyncIterator[CodeRulesEnv]:  # noqa: F811
    instance = CodeRulesEnv({}, repo_task())
    instance.begin_episode(19)
    await instance.setup()
    try:
        yield instance
    finally:
        await instance.teardown()


def test_registration_config_and_rule_sampling() -> None:
    instance = make_env("code_rules", {}, repo_task())
    assert isinstance(instance, CodeRulesEnv)
    assert instance.n_slots == 4
    assert instance.config == CodeRulesEnvConfig()
    assert not instance.supports_vote
    assert instance.canonical(None) is instance.canonical("anything") is None
    state = random.getstate()
    rules = []
    for seed in range(40):
        instance.begin_episode(seed)
        other = CodeRulesEnv({}, repo_task())
        other.begin_episode(seed)
        assert instance._rule == other._rule
        assert instance._rule is not None
        assert re.fullmatch(r"[A-Z]{3}-\d{4}", instance._rule.id)
        rules.append(instance._rule)
    assert random.getstate() == state
    assert len({rule.id for rule in rules}) == len(rules)
    assert {rule.family for rule in rules} == {"header", "constant", "docstring"}
    instance = CodeRulesEnv({}, repo_task(has_rule=False))
    instance.begin_episode(0)
    assert instance._rule is None


@pytest.mark.parametrize(
    "config",
    [
        {"typo": 1},
        {"notes": "oracle"},
        {"ci_runs": 0},
        {"ci_runs": True},
        {"ci_runs": 1.5},
        {"bonus": 0},
        {"bonus": float("nan")},
        {"bonus": True},
        {"code": []},
        {"code": {"typo": 1}},
        {"code": {"timeout_per_test_s": 0}},
    ],
)
def test_invalid_config(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CodeRulesEnv(config, repo_task())


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "stdin"},
        {"problems": []},
        {"has_rule": 1},
        {"rule_families": []},
        {"rule_families": ["typo"]},
    ],
)
def test_invalid_repo(change: dict[str, Any]) -> None:
    task = repo_task()
    with pytest.raises(ValueError):
        CodeRulesEnv({}, replace(task, answer={**task.answer, **change}))


@pytest.mark.parametrize("family", ["header", "constant", "docstring"])
def test_rule_checks_exact_forms_and_near_misses(family: str) -> None:
    rule = _Rule.sample(random.Random(1), [family])
    source = following(rule)
    assert rule.met(source + SUM_SOLUTION)
    assert not rule.met(source.replace(rule.id, "BAD-0000") + SUM_SOLUTION)
    assert not rule.met(source.replace(rule.key, "WRONG_KEY") + SUM_SOLUTION)
    assert not rule.met("")
    if family == "header":
        assert rule.met(source.replace("\n", "\r\n"))
        assert not rule.met("\n" + source)
        assert not rule.met(" " + source)
        assert not rule.met(source.rstrip() + " \n")
        assert not rule.met(source.rstrip() + "0\n")
    elif family == "constant":
        assert not rule.met("def f():\n    " + source)
        assert not rule.met("if True:\n    " + source)
        assert not rule.met(source.replace(" = ", ": str = "))
        assert not rule.met(f'{rule.key} = str("{rule.id}")')
        assert not rule.met(f'other.{rule.key} = "{rule.id}"')
        assert not rule.met("invalid python!")
    else:
        assert rule.met(f'"""A note: @{rule.key} {rule.id}.\nMore text."""')
        assert not rule.met("x = 1\n" + source)
        assert not rule.met("def f():\n    " + source)
        assert not rule.met(source.replace(rule.id, rule.id + "0"))
        assert not rule.met(source.replace("@", "prefix@"))
        assert not rule.met("invalid python!")


async def test_setup_requires_episode_seed() -> None:
    instance = CodeRulesEnv({}, repo_task())
    with pytest.raises(RuntimeError, match="begin_episode"):
        await instance.setup()
    await instance.teardown()


async def test_workspace_layout_and_no_hidden_bytes(env: CodeRulesEnv) -> None:
    sandbox = env.sandbox
    assert sandbox is not None and env._rule is not None
    await env.setup()
    assert env.sandbox is sandbox
    with pytest.raises(RuntimeError, match="before setup"):
        env.begin_episode(2)
    expected = {"NOTES.md"}
    assert await sandbox.read_file("NOTES.md") == ""
    for slot in range(1, 5):
        prefix = f"tasks/task_{slot}"
        for name, value in (
            ("problem.md", "Synthetic: add two integers."),
            ("solution.py", "# Start here\n"),
            ("examples/00.in", "2 3\n"),
            ("examples/00.out", "5"),
        ):
            expected.add(f"{prefix}/{name}")
            assert await sandbox.read_file(f"{prefix}/{name}") == value
    actual = set()
    for path in sandbox.workdir.rglob("*"):
        if path.is_file():
            contents = path.read_text()
            assert env._rule.id not in contents
            assert env._rule.requirement not in contents
            assert "4 5" not in contents and "7 8" not in contents
            if path.name != ".marli-sandbox":
                actual.add(path.relative_to(sandbox.workdir).as_posix())
    assert actual == expected
    result = await tool(env, "bash")(context(env), command="pwd")
    assert json.loads(result.content)["stdout"] == f"{sandbox.workdir}\n"


@pytest.mark.usefixtures("sandbox_host")
async def test_empty_examples_and_missing_starter() -> None:
    task = repo_task(1)
    task.answer["problems"][0]["answer"]["public"] = []
    task.answer["problems"][0]["meta"] = {}
    instance = CodeRulesEnv({}, task)
    instance.begin_episode(1)
    await instance.setup()
    assert instance.sandbox is not None
    root = instance.sandbox.workdir
    try:
        assert (root / "tasks/task_1/examples").is_dir()
        assert await instance.sandbox.read_file("tasks/task_1/solution.py") == ""
    finally:
        await instance.teardown()
    await instance.teardown()
    assert not root.exists() and instance.sandbox is None


def test_exact_contributor_messages() -> None:
    instance = CodeRulesEnv({}, repo_task())
    expected = """You are contributor 1 of 4 working in this repository, one after another. You are the first contributor. After you, 3 more contributor(s) will each work on their own task in this repository.

Your task: make tasks/task_1/solution.py solve the problem in tasks/task_1/problem.md (public examples are in
tasks/task_1/examples/). Use the bash tool to read files, write code and test it.

You may run CI exactly once, in one of two modes:
- ci_submit: scores your solution. Score 0 if the base tests fail, 1 if they pass, and 3 if they pass and
  the repository's extended checks also pass. You are told your score, but not what the extended checks require.
- ci_review: a dry run that prints the full extended-check report, including what the extended checks require.
  It is not scored: your score is 0.
The extended checks are not documented anywhere in the repository.

NOTES.md is shared by everyone who works in this repository. You may read it and write to it (with bash) at any
time, including after your CI run. When you are done, call end_session."""  # noqa: E501
    assert instance.slot_message(0) == expected
    assert instance.task_message("worker") == expected
    middle = instance.slot_message(1)
    assert "contributor 2 of 4" in middle
    assert "The 1 contributor(s) before you have finished." in middle
    assert "After you, 2 more contributor(s)" in middle
    assert "tasks/task_2/solution.py" in middle
    last = instance.slot_message(3)
    assert "The 3 contributor(s) before you have finished." in last
    assert "No one will work in this repository after you." in last
    solo = CodeRulesEnv({}, repo_task(1)).slot_message(0)
    assert "contributor 1 of 1" in solo
    assert "No one else will work in this repository." in solo


async def test_review_reveals_rule_without_grading_and_can_write_notes(
    env: CodeRulesEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def forbidden(*args: Any) -> Any:
        raise AssertionError("review must not grade")

    monkeypatch.setattr(env._graders[0], "grade", forbidden)
    env.bind_agent("worker0", 0)
    assert env.slot_submission("worker0") is None
    result = await tool(env, "ci_review")(context(env))
    assert env._rule is not None
    assert result.content == (
        "CI (dry run, not scored; your score is 0): extended-check report:\n"
        f"- {env._rule.requirement}"
    )
    assert not result.control
    assert all(not item.control and item.shared for item in env.tools("worker"))
    payload = env.slot_submission("worker0")
    assert payload is not None
    assert json.loads(payload) == {
        "v": 1,
        "slot": 0,
        "mode": "review",
        "source": "# Start here\n",
        "rule": asdict(env._rule),
        "notes_had_rule": False,
    }
    assert await env.grade(payload) == {
        "score": 0,
        "base_pass": 0,
        "rule_met": 0,
        "probed": 1,
        "submitted": 0,
        "pass_frac": 0,
        "notes_had_rule": 0,
    }
    result = await tool(env, "bash")(
        context(env), command=f"printf '%s' '{env._rule.id}' > NOTES.md"
    )
    assert json.loads(result.content)["exit_code"] == 0
    env.bind_agent("worker1", 1)
    assert env.sandbox is not None
    assert await env.sandbox.read_file("NOTES.md") == env._rule.id
    await tool(env, "ci_review")(context(env, "worker1"))
    assert (await env.grade(env.slot_submission("worker1")))["notes_had_rule"] == 1


@pytest.mark.parametrize("first", ["ci_submit", "ci_review"])
async def test_one_ci_run_across_modes(env: CodeRulesEnv, first: str) -> None:
    env.bind_agent("worker0", 0)
    await tool(env, first)(context(env))
    payload = env.slot_submission("worker0")
    for name in ("ci_submit", "ci_review"):
        with pytest.raises(ToolError, match=r"^You have already used your CI run\.$"):
            await tool(env, name)(context(env))
    assert env.slot_submission("worker0") == payload


async def test_concurrent_ci_calls_cannot_exceed_budget(env: CodeRulesEnv) -> None:
    env.bind_agent("worker0", 0)
    results = await asyncio.gather(
        tool(env, "ci_review")(context(env)),
        tool(env, "ci_review")(context(env)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ToolError) for result in results) == 1


async def test_unbound_agent_cannot_use_ci(env: CodeRulesEnv) -> None:
    for name in ("ci_submit", "ci_review"):
        with pytest.raises(ToolError, match="bound"):
            await tool(env, name)(context(env))


@pytest.mark.parametrize("solution,score,passed", [("print(9)\n", 0, 1), (SUM_SOLUTION, 1, 3)])
async def test_scored_ci_hides_rule_and_reports_base_tests(
    env: CodeRulesEnv, solution: str, score: int, passed: int
) -> None:
    env.bind_agent("worker0", 0)
    assert env.sandbox is not None and env._rule is not None
    await env.sandbox.write_file("tasks/task_1/solution.py", solution)
    result = await tool(env, "ci_submit")(context(env))
    assert not result.control
    assert result.content == (
        f"CI (scored): base tests {passed}/3 passed. Extended checks: FAILED. Score: {score}."
        " Extended-check details are only shown by ci_review."
    )
    assert env._rule.id not in result.content and env._rule.requirement not in result.content
    assert (await env.grade(env.slot_submission("worker0")))["pass_frac"] == passed / 3


@pytest.mark.parametrize("family", ["header", "constant", "docstring"])
@pytest.mark.usefixtures("sandbox_host")
async def test_bonus_snapshot_memoisation_and_offline_grading(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = repo_task(1, family=family)
    instance = CodeRulesEnv({"bonus": 4.5}, task)
    instance.begin_episode(9)
    await instance.setup()
    try:
        instance.bind_agent("worker0", 0)
        assert instance._rule is not None and instance.sandbox is not None
        source = following(instance._rule) + SUM_SOLUTION
        await instance.sandbox.write_file("tasks/task_1/solution.py", source)
        await instance.sandbox.write_file("NOTES.md", instance._rule.id)
        result = await tool(instance, "ci_submit")(context(instance))
        assert result.content == (
            "CI (scored): base tests 3/3 passed. Extended checks: PASSED. Score: 4.5."
        )
        assert instance._rule.id not in result.content
        payload = instance.slot_submission("worker0")
        assert payload is not None
        await instance.sandbox.write_file("tasks/task_1/solution.py", "raise RuntimeError()")
        await instance.sandbox.write_file("NOTES.md", "")

        async def forbidden(*args: Any) -> Any:
            raise AssertionError("grading must use only saved source, reusing cached base grades")

        monkeypatch.setattr(instance.sandbox, "read_file", forbidden)
        monkeypatch.setattr(instance.sandbox, "exec", forbidden)
        monkeypatch.setattr(instance._graders[0], "_grade_source", forbidden)
        expected = {
            "score": 4.5,
            "base_pass": 1,
            "rule_met": 1,
            "probed": 0,
            "submitted": 1,
            "pass_frac": 1,
            "notes_had_rule": 1,
        }
        assert await instance.grade(payload) == expected
    finally:
        await instance.teardown()
    fresh = CodeRulesEnv({"bonus": 4.5}, task)

    # No begin_episode or setup: the saved rule is sufficient for offline regrade.
    def forbidden_sandbox(self: CodeRulesEnv) -> Any:
        raise AssertionError("offline grading must not inspect the episode sandbox")

    monkeypatch.setattr(CodeRulesEnv, "sandbox", property(forbidden_sandbox))
    try:
        assert await fresh.grade(payload) == expected
    finally:
        await fresh.teardown()


async def test_bundle_means_include_missing_slots_and_preserve_slot_order(
    env: CodeRulesEnv,
) -> None:
    assert env.sandbox is not None and env._rule is not None
    for slot in range(3):
        env.bind_agent(f"worker{slot}", slot)
    await tool(env, "ci_review")(context(env))
    await env.sandbox.write_file("NOTES.md", env._rule.id)
    await env.sandbox.write_file("tasks/task_2/solution.py", following(env._rule) + SUM_SOLUTION)
    await tool(env, "ci_submit")(context(env, "worker1"))
    await env.sandbox.write_file("tasks/task_3/solution.py", SUM_SOLUTION)
    await tool(env, "ci_submit")(context(env, "worker2"))
    bundle = env.bundle({f"worker{i}": env.slot_submission(f"worker{i}") for i in (3, 2, 0, 1)})
    assert [item["slot"] if item else None for item in json.loads(bundle)["bundle"]] == [
        0,
        1,
        2,
        None,
    ]
    assert await env.grade(bundle) == {
        "score": 1,
        "base_pass": 0.5,
        "rule_met": 0.25,
        "probed": 0.25,
        "submitted": 0.5,
        "pass_frac": 0.5,
        "notes_had_rule": 0.5,
        "n_probed": 1,
        "n_rule_met": 1,
    }
    assert all(value == 0 for value in (await env.grade(None)).values())
    assert all(value == 0 for value in (await env.grade(env.bundle({}))).values())


@pytest.mark.usefixtures("sandbox_host")
async def test_no_rule_reports_and_scores() -> None:
    instance = CodeRulesEnv({}, repo_task(2, has_rule=False))
    instance.begin_episode(1)
    await instance.setup()
    try:
        instance.bind_agent("worker0", 0)
        instance.bind_agent("worker1", 1)
        assert instance.sandbox is not None
        await instance.sandbox.write_file("tasks/task_1/solution.py", SUM_SOLUTION)
        result = await tool(instance, "ci_submit")(context(instance))
        assert result.content == (
            "CI (scored): base tests 3/3 passed. Extended checks: none configured. Score: 1."
        )
        grades = await instance.grade(instance.slot_submission("worker0"))
        assert grades["score"] == 1 and grades["rule_met"] == grades["notes_had_rule"] == 0
        result = await tool(instance, "ci_review")(context(instance, "worker1"))
        assert result.content == (
            "CI (dry run, not scored; your score is 0): "
            "no extended checks are configured for this repository."
        )
    finally:
        await instance.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_hidden_notes_reset_on_bind() -> None:
    instance = CodeRulesEnv({"notes": "hidden"}, repo_task(2))
    instance.begin_episode(1)
    await instance.setup()
    try:
        assert instance.sandbox is not None and instance._rule is not None
        instance.bind_agent("worker0", 0)
        await instance.sandbox.write_file("NOTES.md", instance._rule.id)
        await instance.sandbox.write_file("tasks/task_1/solution.py", SUM_SOLUTION)
        instance.bind_agent("worker1", 1)
        assert await instance.sandbox.read_file("NOTES.md") == ""
        assert await instance.sandbox.read_file("tasks/task_1/solution.py") == SUM_SOLUTION
        result = await tool(instance, "bash")(
            context(instance, "worker1"), command="echo hi > NOTES.md"
        )
        assert json.loads(result.content)["exit_code"] == 0
    finally:
        await instance.teardown()


async def test_source_size_limit_is_bytes_and_missing_source_is_empty(env: CodeRulesEnv) -> None:
    env.bind_agent("worker0", 0)
    assert env.sandbox is not None
    await env.sandbox.write_file("tasks/task_1/solution.py", "é" * ((1 << 19) + 1))
    with pytest.raises(ToolError, match="1 MiB"):
        await tool(env, "ci_review")(context(env))
    assert env.slot_submission("worker0") is None
    (env.sandbox.workdir / "tasks/task_1/solution.py").unlink()
    result = await tool(env, "ci_submit")(context(env))
    assert "Score: 0." in result.content
    assert json.loads(env.slot_submission("worker0"))["source"] == ""


@pytest.mark.usefixtures("sandbox_host")
async def test_code_passthrough_and_configured_ci_budget() -> None:
    instance = CodeRulesEnv({"ci_runs": 2, "code": {"max_tests": 1}}, repo_task(1))
    instance.begin_episode(1)
    await instance.setup()
    try:
        instance.bind_agent("worker0", 0)
        await tool(instance, "ci_review")(context(instance))
        assert instance.sandbox is not None
        await instance.sandbox.write_file("tasks/task_1/solution.py", SUM_SOLUTION)
        result = await tool(instance, "ci_submit")(context(instance))
        assert "base tests 1/1 passed" in result.content
        assert (await instance.grade(instance.slot_submission("worker0")))["submitted"] == 1
        with pytest.raises(ToolError, match="already used"):
            await tool(instance, "ci_submit")(context(instance))
    finally:
        await instance.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_grades_use_the_slots_own_problem_and_rule_cannot_rescue_failure() -> None:
    task = repo_task(2, family="constant")
    task.answer["problems"][1] = asdict(code_task(functional=True))
    instance = CodeRulesEnv({}, task)
    instance.begin_episode(1)
    assert instance._rule is not None
    payload = {
        "v": 1,
        "slot": 1,
        "mode": "submit",
        "source": following(instance._rule) + "def add(a, b):\n    return a + b\n",
        "rule": asdict(instance._rule),
        "notes_had_rule": False,
    }
    try:
        functional = await instance.grade(json.dumps(payload))
        stdin = await instance.grade(json.dumps({**payload, "slot": 0}))
        assert functional["score"] == 3 and functional["base_pass"] == 1
        assert stdin["score"] == 0 and stdin["base_pass"] == 0
        assert stdin["rule_met"] == 1
    finally:
        await instance.teardown()
