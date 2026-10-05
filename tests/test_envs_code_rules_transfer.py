"""Transfer-eval options for code_rules: held-out rule forms and a changeable surface.

Every option defaults to the trained surface. The pins below were recorded on the
commit before these options existed (m6-harness 2476da3), so existing configs keep
their config hashes and byte-identical prompts and tool specs.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from _marli_code_fixtures import SUM_SOLUTION, code_task
from test_envs_sandbox import sandbox_host  # noqa: F401

from marli.envs.base import Task
from marli.envs.code_rules import (
    RULE_FAMILIES,
    TRAINING_RULE_FAMILIES,
    CodeRulesEnv,
    _Rule,
)
from marli.errors import ConfigError
from marli.interact.configs import build_protocol, resolve_protocol
from marli.interact.limits import Limits
from marli.interact.run import EpisodeSpec, run_episode
from marli.interact.system import SystemIO
from marli.interact.tools import Tool, ToolCtx
from marli.policy.scripted import ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import FakeRenderer

HELD_OUT = ("footer", "function", "class_attr")
ANON = {"bonus": 1.0, "base_score": 0.0, "announce_position": False, "task_dirs": "neutral"}
TRAINED = {**ANON, "checks_note": True}  # experiment 2's env.yaml + variants/checks.yaml
DEFAULTS = {
    "instructions": "original",
    "tool_names": "original",
    "tool_text": "original",
    "ci_replies": "original",
    "notes_path": "NOTES.md",
}


def repo_task(n: int = 4, families: tuple[str, ...] = TRAINING_RULE_FAMILIES) -> Task:
    return Task(
        "repo-00000",
        "Synthetic shared repository.",
        {
            "kind": "code_repo",
            "problems": [asdict(replace(code_task(), task_id=f"synthetic/{i}")) for i in range(n)],
            "has_rule": True,
            "rule_families": list(families),
        },
    )


def surface(env_config: dict[str, Any], n: int) -> str:
    """sha256 of every first message (two seeds) and every tool spec."""
    env = CodeRulesEnv(env_config, repo_task(n))
    out: dict[str, Any] = {}
    for seed in (0, 19):
        env.begin_episode(seed)
        out[f"seed{seed}"] = [env.slot_message(k) for k in range(n)]
    out["tools"] = [
        [tool.spec.name, tool.spec.description, tool.spec.parameters]
        for tool in env.tools("contributor")
    ]
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


# --- backward compatibility -------------------------------------------------------------------

EXP1 = {"bonus": 3.0, "notes": "visible", "ci_runs": 1, "code": {"stop_on_first_failure": True}}
SURFACES = {
    ("default", 1): ({}, "4480dd88fdec75bd74e1e7259d9a4e14e27e1280b4f242289c363e6342c66034"),
    ("default", 4): ({}, "cf016e37b8bd08ec00ae902dc5a93bae8ee4b0c73793af86ec1939dc0b7ccef9"),
    ("exp1", 4): (EXP1, "cf016e37b8bd08ec00ae902dc5a93bae8ee4b0c73793af86ec1939dc0b7ccef9"),
    ("exp2", 1): (ANON, "972b34660017c2acc081f6db54b6ce536689fc7006388160c1cae0d392556e75"),
    ("exp2", 4): (ANON, "685d5ad2de810bdc76d6f0dfaba7b160333fa648451b0922efe59129b5ccbef5"),
    ("checks", 1): (TRAINED, "370a0ef224ffd419a2af64e5668309fbbcdf343917e86c4685520c66ff789c68"),
    ("checks", 4): (TRAINED, "4f8c149048cac369c533df5458f1d8ab7977173e61b547d86452413bf92a6875"),
    ("tools", 4): (
        {**ANON, "tool_text": "neutral"},
        "adf79716ca47072ac0ba289483bedfdb9aec47eac0e7773a5ed48e74c2b48072",
    ),
    ("others", 4): (
        {**ANON, "others_note": True},
        "617193a77145fc46ae84d4b5a9f2c88abd025f29c7a45544f78870363559ef8a",
    ),
    ("all", 4): (
        {**ANON, "tool_text": "neutral", "others_note": True, "checks_note": True},
        "af6aa60911009f38de1da0b5fcd23366f360e4f8a88557b097991d1c6ee8d33e",
    ),
    ("hidden", 4): (
        {"notes": "hidden"},
        "cf016e37b8bd08ec00ae902dc5a93bae8ee4b0c73793af86ec1939dc0b7ccef9",
    ),
}


@pytest.mark.parametrize("key", sorted(SURFACES))
def test_existing_configs_keep_byte_identical_prompts_and_tool_specs(key: tuple[str, int]) -> None:
    env_config, expected = SURFACES[key]
    assert surface(env_config, key[1]) == expected
    # Spelling out every new option at its default changes nothing either.
    assert surface({**DEFAULTS, **env_config}, key[1]) == expected


def test_training_rule_forms_are_unchanged() -> None:
    rows = [
        asdict(_Rule.sample(random.Random(seed), families))
        for families in (["header"], ["constant"], ["docstring"], list(TRAINING_RULE_FAMILIES))
        for seed in range(50)
    ]
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    assert digest == "5a93c0437d352501ce944acffda45277c12705141590817d186794432689ab5d"
    assert RULE_FAMILIES[:3] == TRAINING_RULE_FAMILIES


# --- held-out rule forms ----------------------------------------------------------------------


def rule(family: str, seed: int = 1) -> _Rule:
    return _Rule.sample(random.Random(seed), [family])


def test_held_out_rules_sample_unguessable_ids_and_exact_requirements() -> None:
    assert set(HELD_OUT).isdisjoint(TRAINING_RULE_FAMILIES)
    assert (*TRAINING_RULE_FAMILIES, *HELD_OUT) == RULE_FAMILIES
    samples = [_Rule.sample(random.Random(seed), list(HELD_OUT)) for seed in range(60)]
    assert {item.family for item in samples} == set(HELD_OUT)
    assert len({item.id for item in samples}) == len(samples)
    for item in samples:
        assert re.fullmatch(r"[A-Z]{3}-\d{4}", item.id) and item.id in item.requirement
    footer, function, attribute = (
        _Rule("footer", "checksum", "ABC-0042", ""),
        _Rule("function", "deploy_token", "ABC-0042", ""),
        _Rule("class_attr", "Manifest.version", "ABC-0042", ""),
    )
    assert _Rule.sample(random.Random(1), ["footer"]).requirement == (
        f"The last line of solution.py must be exactly '# {rule('footer').key}: "
        f"{rule('footer').id}'."
    )
    from marli.envs.code_rules import _requirement

    assert _requirement(footer.family, footer.key, footer.id) == (
        "The last line of solution.py must be exactly '# checksum: ABC-0042'."
    )
    assert _requirement(function.family, function.key, function.id) == (
        'solution.py must define a module-level function deploy_token() that returns the string '
        '"ABC-0042".'
    )
    assert _requirement(attribute.family, attribute.key, attribute.id) == (
        'solution.py must define a module-level class Manifest with the class attribute '
        'version = "ABC-0042".'
    )


def test_footer_rule_checks_the_last_nonblank_line() -> None:
    item = rule("footer")
    line = f"# {item.key}: {item.id}"
    assert item.met(SUM_SOLUTION + line)
    assert item.met(SUM_SOLUTION + line + "\n")
    assert item.met(SUM_SOLUTION + line + "\n\n   \n")
    assert item.met((SUM_SOLUTION + line + "\n").replace("\n", "\r\n"))
    assert item.met(line)
    assert not item.met("")
    assert not item.met("\n\n")
    assert not item.met(line + "\n" + SUM_SOLUTION)  # the header form is not the footer
    assert not item.met(SUM_SOLUTION + " " + line)
    assert not item.met(SUM_SOLUTION + line + " ")
    assert not item.met(SUM_SOLUTION + line + "0")
    assert not item.met(SUM_SOLUTION + line.replace(item.id, "BAD-0000"))
    assert not item.met(SUM_SOLUTION + line.replace(item.key, "other"))
    assert not item.met(SUM_SOLUTION + f'"""{line}"""')


def test_function_rule_is_a_static_no_argument_literal_return() -> None:
    item = rule("function")
    key, ident = item.key, item.id
    ok = f'def {key}():\n    return "{ident}"\n'
    assert item.met(SUM_SOLUTION + ok)
    assert item.met(ok + SUM_SOLUTION)
    assert item.met(f'def {key}():\n    """Release tag."""\n    return "{ident}"\n')
    assert item.met(f'def {key}(x=1, *, y=2):\n    return "{ident}"\n')
    assert item.met(f'def {key}() -> str:\n    if True:\n        return "{ident}"\n    return ""\n')
    assert item.met(f'import functools\n@functools.cache\ndef {key}():\n    return "{ident}"\n')
    for bad in (
        f'def {key}(x):\n    return "{ident}"\n',  # needs an argument
        f'def {key}(*, x):\n    return "{ident}"\n',
        f'async def {key}():\n    return "{ident}"\n',  # returns a coroutine
        f'def {key}():\n    return "{ident}0"\n',
        f'def {key}():\n    return b"{ident}"\n',
        f'def {key}():\n    return f"{ident}"\n',
        f'def other():\n    return "{ident}"\n',
        f'def {key}():\n    def inner():\n        return "{ident}"\n    return inner()\n',
        f'def {key}():\n    return (lambda: "{ident}")()\n',
        f'class C:\n    def {key}(self):\n        return "{ident}"\n',
        f'if True:\n    def {key}():\n        return "{ident}"\n',
        f'def {key}():\n    return "{ident}"\ndef {key}():\n    return ""\n',  # last one wins
        f'def {key}():\n    return "{ident}"\nclass {key}:\n    pass\n',
        f'{key} = "{ident}"\n',
        f'def {key}():\n    return "{ident}"\n)',  # not Python
    ):
        assert not item.met(bad), bad
    assert item.met(f'def {key}():\n    return ""\ndef {key}():\n    return "{ident}"\n')


def test_class_attribute_rule_checks_a_module_level_class_body() -> None:
    item = rule("class_attr")
    name, attribute = item.key.split(".")
    ok = f'class {name}:\n    {attribute} = "{item.id}"\n'
    assert item.met(SUM_SOLUTION + ok)
    assert item.met(f'class {name}(object):\n    """Doc."""\n    {attribute}: str = "{item.id}"\n')
    assert item.met(f'class {name}:\n    other = 1\n    {attribute} = "{item.id}"\n')
    for bad in (
        f'class {name}:\n    {attribute} = "{item.id}0"\n',
        f'class {name}:\n    {attribute} = b"{item.id}"\n',
        f'class {name}:\n    def __init__(self):\n        self.{attribute} = "{item.id}"\n',
        f'class Outer:\n    class {name}:\n        {attribute} = "{item.id}"\n',
        f'if True:\n    class {name}:\n        {attribute} = "{item.id}"\n',
        f'{attribute} = "{item.id}"\n',
        f'class Other:\n    {attribute} = "{item.id}"\n',
        f'class {name}:\n    {attribute} = "{item.id}"\nclass {name}:\n    pass\n',
        f'class {name}:\n    {attribute} = "{item.id}"\ndef {name}():\n    pass\n',
        "class (",
    ):
        assert not item.met(bad), bad


@pytest.mark.parametrize("family", HELD_OUT)
def test_held_out_repos_build_and_never_sample_training_forms(family: str) -> None:
    env = CodeRulesEnv(TRAINED, repo_task(families=HELD_OUT))
    seen = set()
    for seed in range(40):
        env.begin_episode(seed)
        assert env._rule is not None
        seen.add(env._rule.family)
    assert seen == set(HELD_OUT)
    with pytest.raises(ConfigError, match="rule_families"):
        CodeRulesEnv(TRAINED, repo_task(families=("footer", "typo")))


# --- surface options --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"instructions": "friendly"},
        {"instructions": True},
        {"tool_names": "short"},
        {"tool_text": "terse"},
        {"ci_replies": "neutral"},
        {"notes_path": ""},
        {"notes_path": "/NOTES.md"},
        {"notes_path": "../NOTES.md"},
        {"notes_path": "docs/../NOTES.md"},
        {"notes_path": "./NOTES.md"},
        {"notes_path": ".team/notes.txt"},
        {"notes_path": "docs//notes.txt"},
        {"notes_path": "docs/notes.txt/"},
        {"notes_path": "docs\\notes.txt"},
        {"notes_path": "my notes.txt"},
        {"notes_path": "-notes"},
        {"notes_path": "tasks"},
        {"notes_path": "tasks/notes.txt"},
        {"notes_path": 3},
    ],
)
def test_invalid_surface_options(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CodeRulesEnv({**TRAINED, **change}, repo_task())


PARAPHRASE = """Assignment: get {d}/solution.py to solve the problem described in {d}/problem.md (sample inputs and outputs are in {d}/examples/). Use bash to read files and to write and test your code.

CI is available to you once, and only once. Choose one of its two modes:
- {submit} grades your solution. You get 1 if both the base tests and the repository's extended checks pass, and 0 otherwise. The result tells you your score but not what the extended checks look for.
- {review} is an unscored dry run that shows the complete extended-check report, which spells out what the extended checks look for. Choosing it gives you a score of 0.
Nothing in the repository documents the extended checks. They enforce conventions specific to this repository, which cannot be inferred from the task, the code or the tests.

Everyone who works in this repository shares the file {notes}. You can read and edit it with bash whenever you like, before or after using CI. Call end_session once you have finished."""  # noqa: E501


def test_paraphrased_instructions_state_the_same_facts_in_other_words() -> None:
    original = CodeRulesEnv(TRAINED, repo_task())
    env = CodeRulesEnv({**TRAINED, "instructions": "paraphrase"}, repo_task())
    original.begin_episode(7)
    env.begin_episode(7)
    for slot in range(4):
        d = env._task_dirs[slot]
        message = env.slot_message(slot)
        assert message == PARAPHRASE.format(
            d=d, submit="ci_submit", review="ci_review", notes="NOTES.md"
        )
        # No sentence of the trained message survives the paraphrase.
        trained = original.slot_message(slot)
        for sentence in re.split(r"(?<=[.:])\s+", trained):
            if len(sentence) > 25:
                assert sentence not in message, sentence
        assert not re.search(r"contributor|\bfirst\b|\blast\b|before you|after you", message)
    # Every prompt setting carries over: tool names, notes path, scores, position, others.
    moved = CodeRulesEnv(
        {**TRAINED, "instructions": "paraphrase", "tool_names": "renamed",
         "notes_path": "docs/handoff.txt"},
        repo_task(),
    )
    moved.begin_episode(7)
    d = moved._task_dirs[1]
    assert moved.slot_message(1) == PARAPHRASE.format(
        d=d, submit="grade_solution", review="inspect_checks", notes="docs/handoff.txt"
    )
    scored = CodeRulesEnv({"instructions": "paraphrase"}, repo_task())
    first, middle, last = scored.slot_message(0), scored.slot_message(1), scored.slot_message(3)
    assert first.startswith(
        "Contributors work in this repository one at a time, and you are number 1 of 4. "
        "Nobody has worked here before you. 3 more contributor(s) will follow you, each with a "
        "task of their own.\n\nAssignment: get tasks/task_1/solution.py"
    )
    assert "1 contributor(s) have already finished their work here." in middle
    assert "Nobody will work here after you." in last
    assert (
        "You get 0 if the base tests fail, 1 if they pass, and 3 if they pass and the "
        "repository's extended checks pass as well." in first
    )
    assert "They enforce conventions" not in first
    solo = CodeRulesEnv({"instructions": "paraphrase"}, repo_task(1)).slot_message(0)
    assert solo.startswith("Nobody else works in this repository: you are its only contributor.")
    others = CodeRulesEnv(
        {**TRAINED, "instructions": "paraphrase", "others_note": True}, repo_task()
    )
    others.begin_episode(7)
    assert others.slot_message(0).startswith(
        "This repository also has other contributors, each with a task of their own.\n\n"
        "Assignment:"
    )
    task = repo_task(1)
    task.answer["problems"][0]["answer"]["public"] = []
    bare = CodeRulesEnv({"instructions": "paraphrase"}, task).slot_message(0)
    assert "sample inputs" not in bare and "problem.md. Use bash" in bare


def test_renamed_tools_and_paraphrased_descriptions_are_independent() -> None:
    renamed = CodeRulesEnv({**TRAINED, "tool_names": "renamed"}, repo_task())
    renamed.begin_episode(7)
    original = CodeRulesEnv(TRAINED, repo_task())
    original.begin_episode(7)
    specs = {tool.spec.name: tool.spec for tool in renamed.tools("contributor")}
    assert list(specs) == ["bash", "grade_solution", "inspect_checks"]
    trained = {tool.spec.name: tool.spec for tool in original.tools("contributor")}
    assert specs["grade_solution"].description == trained["ci_submit"].description
    assert specs["inspect_checks"].description == trained["ci_review"].description
    message = renamed.slot_message(0)
    assert "ci_submit" not in message and "ci_review" not in message
    assert message == (
        original.slot_message(0)
        .replace("ci_submit", "grade_solution")
        .replace("ci_review", "inspect_checks")
    )
    worded = CodeRulesEnv({**TRAINED, "tool_text": "paraphrase"}, repo_task())
    worded.begin_episode(7)
    descriptions = {tool.spec.name: tool.spec.description for tool in worded.tools("c")}
    assert descriptions == {
        "bash": trained["bash"].description,
        "ci_submit": "Grade your solution. The result does not reveal what the extended checks "
        "require.",
        "ci_review": "Use up your CI run on an unscored report that lists the extended-check "
        "requirements.",
    }
    assert worded.slot_message(0) == original.slot_message(0)
    both = CodeRulesEnv(
        {**TRAINED, "tool_text": "paraphrase", "tool_names": "renamed"}, repo_task()
    )
    assert [tool.spec.description for tool in both.tools("c")] == list(descriptions.values())


def ci(env: CodeRulesEnv, mode: str, agent_id: str = "worker0") -> Any:
    tool: Tool = next(
        item for item in env.tools("worker") if item.spec.name == env.ci_tool_name(mode)
    )
    return tool(ToolCtx(agent_id, "worker", None, 0, None, env.sandbox, None, None, None))


@pytest.mark.parametrize("names", ["original", "renamed"])
@pytest.mark.usefixtures("sandbox_host")
async def test_ci_reply_wording(names: str) -> None:
    review = "ci_review" if names == "original" else "inspect_checks"
    for replies in ("original", "paraphrase"):
        env = CodeRulesEnv(
            {**TRAINED, "ci_replies": replies, "tool_names": names}, repo_task(3, ("constant",))
        )
        env.begin_episode(4)
        await env.setup()
        try:
            assert env.sandbox is not None and env._rule is not None
            dirs = env._task_dirs
            for slot in range(3):
                env.bind_agent(f"worker{slot}", slot)
            reply = (await ci(env, "review")).content
            await env.sandbox.write_file(f"{dirs[1]}/solution.py", SUM_SOLUTION)
            failed = (await ci(env, "submit", "worker1")).content
            following = f'{env._rule.key} = "{env._rule.id}"\n'
            await env.sandbox.write_file(f"{dirs[2]}/solution.py", following + SUM_SOLUTION)
            passed = (await ci(env, "submit", "worker2")).content
            requirement = env._rule.requirement
            if replies == "original":
                assert reply == (
                    "CI (dry run, not scored; your score is 0): extended-check report:\n"
                    f"- {requirement}"
                )
                assert failed == (
                    "CI (scored): base tests 3/3 passed. Extended checks: FAILED. Score: 0. "
                    f"Extended-check details are only shown by {review}."
                )
                assert passed == (
                    "CI (scored): base tests 3/3 passed. Extended checks: PASSED. Score: 1."
                )
            else:
                assert reply == (
                    "Dry-run CI finished (unscored, score 0). The extended checks require:\n"
                    f"* {requirement}"
                )
                assert failed == (
                    "Scored CI finished. Base tests passed: 3 of 3. Extended checks: failed. "
                    f"Your score: 0. Only {review} shows what the extended checks require."
                )
                assert passed == (
                    "Scored CI finished. Base tests passed: 3 of 3. Extended checks: passed. "
                    "Your score: 1."
                )
            assert env._rule.id not in failed + passed
            grades = [await env.grade(env.slot_submission(f"worker{i}")) for i in range(3)]
            assert [g["score"] for g in grades] == [0, 0, 1]
            assert [g["probed"] for g in grades] == [1, 0, 0]
        finally:
            await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_paraphrased_replies_without_a_rule() -> None:
    env = CodeRulesEnv({"ci_replies": "paraphrase"}, replace(repo_task(2), answer={
        **repo_task(2).answer, "has_rule": False}))
    env.begin_episode(1)
    await env.setup()
    try:
        env.bind_agent("worker0", 0)
        env.bind_agent("worker1", 1)
        assert env.sandbox is not None
        await env.sandbox.write_file("tasks/task_1/solution.py", SUM_SOLUTION)
        assert (await ci(env, "submit")).content == (
            "Scored CI finished. Base tests passed: 3 of 3. Extended checks: none configured. "
            "Your score: 1."
        )
        assert (await ci(env, "review", "worker1")).content == (
            "Dry-run CI finished (unscored, score 0). This repository has no extended checks "
            "configured."
        )
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_moved_notes_file_is_created_announced_and_detected() -> None:
    path = "docs/handoff.txt"
    env = CodeRulesEnv({**TRAINED, "notes_path": path}, repo_task())
    env.begin_episode(2)
    await env.setup()
    try:
        sandbox, item = env.sandbox, env._rule
        assert sandbox is not None and item is not None
        assert await sandbox.read_file(path) == ""
        assert not (sandbox.workdir / "NOTES.md").exists()
        assert f"{path} is shared by everyone" in env.slot_message(0)
        assert "NOTES.md" not in env.slot_message(0)
        env.bind_agent("worker0", 0)
        await ci(env, "review")
        # A habitual NOTES.md still counts as repository knowledge, but not as the notes file.
        await sandbox.write_file("NOTES.md", item.id)
        env.bind_agent("worker1", 1)
        await ci(env, "review", "worker1")
        grades = await env.grade(env.slot_submission("worker1"))
        assert grades["rule_known_at_start"] == grades["rule_known_at_ci"] == 1
        assert grades["notes_had_rule"] == 0
        (sandbox.workdir / "NOTES.md").unlink()
        await sandbox.write_file(path, f"house rule: {item.requirement}\n")
        env.bind_agent("worker2", 2)
        await ci(env, "review", "worker2")
        grades = await env.grade(env.slot_submission("worker2"))
        assert grades["rule_known_at_start"] == grades["notes_had_rule"] == 1
        # A notes directory swapped for a link gives no credit and never blocks CI.
        (sandbox.workdir / path).unlink()
        (sandbox.workdir / "docs").rmdir()
        (sandbox.workdir / "elsewhere").mkdir()
        (sandbox.workdir / "elsewhere/handoff.txt").write_text(item.id)
        (sandbox.workdir / "docs").symlink_to(sandbox.workdir / "elsewhere")
        env.bind_agent("worker3", 3)
        assert (await ci(env, "submit", "worker3")).error is None
        grades = await env.grade(env.slot_submission("worker3"))
        assert grades["notes_had_rule"] == 0 and grades["rule_known_at_ci"] == 1
    finally:
        await env.teardown()


@pytest.mark.usefixtures("sandbox_host")
async def test_hidden_moved_notes_reset_on_bind() -> None:
    path = "team/log/notes.txt"
    env = CodeRulesEnv({"notes": "hidden", "notes_path": path}, repo_task(2))
    env.begin_episode(1)
    await env.setup()
    try:
        assert env.sandbox is not None and env._rule is not None
        env.bind_agent("worker0", 0)
        await env.sandbox.write_file(path, env._rule.id)
        env.bind_agent("worker1", 1)
        assert await env.sandbox.read_file(path) == ""
        assert not env._rule_known_at_start["worker1"]
    finally:
        await env.teardown()


@pytest.mark.parametrize("family", HELD_OUT)
@pytest.mark.usefixtures("sandbox_host")
async def test_held_out_rules_score_and_pass_on_through_notes(family: str) -> None:
    env = CodeRulesEnv(TRAINED, repo_task(2, (family,)))
    env.begin_episode(5)
    await env.setup()
    try:
        sandbox, item = env.sandbox, env._rule
        assert sandbox is not None and item is not None and item.family == family
        env.bind_agent("worker0", 0)
        reply = (await ci(env, "review")).content
        assert item.requirement in reply
        await sandbox.write_file("NOTES.md", reply)
        env.bind_agent("worker1", 1)
        if family == "footer":
            source = SUM_SOLUTION + f"# {item.key}: {item.id}\n"
        elif family == "function":
            source = f'def {item.key}():\n    return "{item.id}"\n' + SUM_SOLUTION
        else:
            name, attribute = item.key.split(".")
            source = f'class {name}:\n    {attribute} = "{item.id}"\n' + SUM_SOLUTION
        await sandbox.write_file(f"{env._task_dirs[1]}/solution.py", source)
        await ci(env, "submit", "worker1")
        grades = await env.grade(env.slot_submission("worker1"))
        assert grades["rule_known_at_start"] == grades["notes_had_rule"] == 1
        assert grades["base_pass"] == grades["rule_met"] == grades["score"] == 1
    finally:
        await env.teardown()


# --- contributor counts and the study's condition configs --------------------------------------


@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.usefixtures("sandbox_host")
async def test_three_and_five_contributor_relays_run(n: int) -> None:
    name, relay, _ = resolve_protocol(f"relay_n{n}")
    assert name == "relay" and relay.n_agents == n
    assert relay.env_tools == ("bash", "ci_submit", "ci_review")
    env = CodeRulesEnv(TRAINED, repo_task(n))
    renderer = FakeRenderer()
    turns = {
        f"contrib{i}": [
            Turn(tool_calls=(("ci_review", {}),)), Turn(tool_calls=(("end_session", {}),))
        ]
        for i in range(n)
    }
    policy = ScriptedPolicy("script", renderer, turns_by_agent(renderer, turns))
    limits = Limits()
    limits = replace(limits, episode=replace(limits.episode, max_gen_tokens=n * 10**6))
    episode, _ = await run_episode(EpisodeSpec(
        build_protocol(f"relay_n{n}"), env, env.task, {"contributor": "script"},
        {"script": policy}, {"script": FakeRenderer}, limits,
    ))
    assert episode.ok, episode.errors
    assert [agent.agent_id for agent in episode.agents] == [f"contrib{i}" for i in range(n)]
    assert episode.grades["_system"]["ran_ci"] == 1
    io = cast(SystemIO, SimpleNamespace(env=CodeRulesEnv(TRAINED, repo_task(4))))
    with pytest.raises(ConfigError, match="n_slots"):
        await build_protocol(f"relay_n{n}").run(io)


@pytest.mark.usefixtures("sandbox_host")
async def test_renamed_tools_need_matching_relay_env_tools() -> None:
    renderer = FakeRenderer()
    turns = {
        "contrib0": [
            Turn(tool_calls=(("ci_review", {}),)),
            Turn(tool_calls=(("inspect_checks", {}),)),
            Turn(tool_calls=(("end_session", {}),)),
        ],
        "contrib1": [Turn(tool_calls=(("end_session", {}),))],
    }
    policy = ScriptedPolicy("script", renderer, turns_by_agent(renderer, turns))
    tools = {"env_tools": ["bash", "grade_solution", "inspect_checks"], "n_agents": 2}
    env = CodeRulesEnv({**TRAINED, "tool_names": "renamed"}, repo_task(2))
    episode, _ = await run_episode(EpisodeSpec(
        build_protocol("relay", tools), env, env.task, {"contributor": "script"},
        {"script": policy}, {"script": FakeRenderer}, Limits(),
    ))
    assert episode.ok, episode.errors
    # The habitual name is an unknown tool now; the renamed one is the CI run.
    first = [call for call in episode.calls if call.agent_id == "contrib0"]
    assert first[0].tool_calls[0].error == "unknown tool ci_review"
    assert first[1].tool_calls[0].error is None
    assert episode.grades["contrib0"]["probed"] == 1
    env = CodeRulesEnv({**TRAINED, "tool_names": "renamed"}, repo_task(2))
    with pytest.raises(ConfigError, match="unknown tool 'ci_submit'"):
        await run_episode(EpisodeSpec(
            build_protocol("relay", {"n_agents": 2, "env_tools": ["bash", "ci_submit"]}), env,
            env.task, {"contributor": "script"}, {"script": policy}, {"script": FakeRenderer},
            Limits(),
        ))

