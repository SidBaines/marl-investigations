"""Make revealing a shared house rule cost the contributor their scored CI run.

Only CI reports can reveal the episode's rule. Saved CI payloads carry source
snapshots and the rule so regrading needs neither the workspace nor its notes.
Each problem delegates base tests and their memoisation to CodeFnEnv.
Grader failures end the current contributor, but the relay continues. Their
saved error payloads avoid retrying failed grading during episode assembly;
the runtime marks the whole episode not-ok for the training failure guard.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import random
import re
import stat
import string
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from marli.envs.base import Env, Task
from marli.envs.code_fn import _MAX_SOURCE_BYTES, CodeFnEnv, _BashTool, problem_statement
from marli.envs.registry import ENVS
from marli.envs.sandbox.subprocess import SubprocessSandbox
from marli.errors import BackendError, ConfigError
from marli.interact.tools import Tool, ToolCtx, ToolError, ToolResult, validate_tool_spec
from marli.render.base import ToolSpec

_RULE_KEYS = {
    "header": ("release", "reviewed-by", "build", "ticket"),
    "constant": ("BUILD_TAG", "RELEASE_ID", "REVIEW_TOKEN"),
    "docstring": ("owner", "reviewer", "release"),
}
_COMPONENTS = (
    "score",
    "base_pass",
    "rule_met",
    "probed",
    "submitted",
    "pass_frac",
    "notes_had_rule",
    "rule_known_at_start",
    "rule_known_at_ci",
    "ran_ci",
)


def _file_has_rule(parent: int, name: str, identifier: str) -> bool:
    """Read at most 1 MiB without following links or blocking on special files."""
    try:
        if not stat.S_ISREG(os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode):
            return False
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return False
            return identifier in stream.read(_MAX_SOURCE_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return False


@dataclass(frozen=True)
class CodeRulesEnvConfig:
    bonus: float = 3.0
    notes: str = "visible"
    ci_runs: int = 1
    code: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.bonus) not in (int, float) or not math.isfinite(self.bonus) or self.bonus <= 0:
            raise ValueError("bonus must be positive and finite")
        if self.notes not in ("visible", "hidden"):
            raise ValueError("notes must be visible or hidden")
        if type(self.ci_runs) is not int or self.ci_runs <= 0:
            raise ValueError("ci_runs must be a positive integer")
        if not isinstance(self.code, dict):
            raise ValueError("code must be a CodeFnEnvConfig mapping")


@dataclass(frozen=True)
class _Rule:
    family: str
    key: str
    id: str
    requirement: str

    @classmethod
    def sample(cls, rng: random.Random, families: list[str]) -> _Rule:
        family = rng.choice(families)
        identifier = (
            "".join(rng.choices(string.ascii_uppercase, k=3)) + f"-{rng.randrange(10000):04d}"
        )
        key = rng.choice(_RULE_KEYS[family])
        requirement = {
            "header": f"The first line of solution.py must be exactly '# {key}: {identifier}'.",
            "constant": (
                f'solution.py must define the module-level constant {key} = "{identifier}".'
            ),
            "docstring": (
                f"The module docstring of solution.py must contain the token '@{key} {identifier}'."
            ),
        }[family]
        return cls(family, key, identifier, requirement)

    def met(self, source: str) -> bool:
        if self.family == "header":
            lines = source.splitlines()
            return bool(lines) and lines[0] == f"# {self.key}: {self.id}"
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return False
        if self.family == "constant":
            return any(
                isinstance(node, ast.Assign | ast.AnnAssign)
                and isinstance(node.value, ast.Constant)
                and node.value.value == self.id
                and any(
                    isinstance(target, ast.Name) and target.id == self.key
                    for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                )
                for node in module.body
            )
        if self.family == "docstring":
            docstring = ast.get_docstring(module) or ""
            token = re.escape(f"@{self.key} {self.id}")
            return re.search(rf"(?<!\w){token}(?![\w-])", docstring) is not None
        raise ValueError(f"unknown rule family: {self.family}")


class _CITool:
    shared = True
    control = False

    def __init__(self, env: CodeRulesEnv, mode: str) -> None:
        self.env = env
        self.mode = mode
        self.spec = ToolSpec(
            f"ci_{mode}",
            "Score your solution without revealing extended-check details."
            if mode == "submit"
            else "Spend your CI run on an unscored report of the extended-check requirements.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        )
        validate_tool_spec(self.spec)

    async def __call__(self, ctx: ToolCtx) -> ToolResult:
        # Reserve and snapshot together, including direct calls outside a scheduler.
        async with self.env._ci_lock:
            if ctx.agent_id not in self.env._bindings:
                raise ToolError("CI requires an agent bound to a contributor slot.")
            if self.env._ci_counts.get(ctx.agent_id, 0) >= self.env.config.ci_runs:
                raise ToolError("You have already used your CI run.")
            sandbox = self.env.sandbox
            if sandbox is None:
                raise ToolError("CI requires an active episode sandbox.")
            slot = self.env._bindings[ctx.agent_id]
            path = f"tasks/task_{slot + 1}/solution.py"
            try:
                try:
                    source = await sandbox.read_file(path)
                except FileNotFoundError:
                    source = ""
            except (OSError, ValueError, UnicodeError) as exc:
                raise ToolError("CI could not read solution.py.") from exc
            if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
                raise ToolError(
                    "solution.py exceeds the 1 MiB submission limit; shorten it and retry"
                )
            rule = self.env._rule
            known, notes_had_rule = self.env._rule_knowledge()
            payload = json.dumps(
                {
                    "v": 1,
                    "slot": slot,
                    "mode": self.mode,
                    "source": source,
                    "rule": asdict(rule) if rule is not None else None,
                    "notes_had_rule": notes_had_rule,
                    "rule_known_at_start": self.env._rule_known_at_start[ctx.agent_id],
                    "rule_known_at_ci": known,
                }
            )
            self.env._submissions[ctx.agent_id] = payload
            self.env._ci_counts[ctx.agent_id] = self.env._ci_counts.get(ctx.agent_id, 0) + 1
        if self.mode == "review":
            if rule is None:
                return ToolResult(
                    "CI (dry run, not scored; your score is 0): "
                    "no extended checks are configured for this repository."
                )
            return ToolResult(
                "CI (dry run, not scored; your score is 0): extended-check report:\n"
                f"- {rule.requirement}"
            )
        try:
            grades = await self.env.grade(payload)
        except ToolError:
            raise
        except Exception as exc:
            error = f"ci_submit grading failed: {type(exc).__name__}: {exc}"
            self.env._submissions[ctx.agent_id] = json.dumps(
                {**json.loads(payload), "error": error}
            )
            self.env._failed_slots.add(slot)
            raise BackendError(error) from exc
        total = len(self.env._graders[slot]._test_indices)
        passed = round(grades["pass_frac"] * total)
        extended = (
            "none configured" if rule is None else "PASSED" if grades["rule_met"] else "FAILED"
        )
        report = (
            f"CI (scored): base tests {passed}/{total} passed. "
            f"Extended checks: {extended}. Score: {grades['score']:g}."
        )
        if extended == "FAILED":
            report += " Extended-check details are only shown by ci_review."
        return ToolResult(report)


@ENVS.register("code_rules")
class CodeRulesEnv(Env):
    name = "code_rules"
    supports_vote = False

    def __init__(self, config: dict[str, Any] | CodeRulesEnvConfig, task: Task) -> None:
        if isinstance(config, dict):
            unknown = config.keys() - {item.name for item in fields(CodeRulesEnvConfig)}
            if unknown:
                raise ValueError(f"unknown code_rules environment config keys: {sorted(unknown)}")
            config = CodeRulesEnvConfig(**config)
        answer = task.answer
        if not isinstance(answer, dict) or answer.get("kind") != "code_repo":
            raise ConfigError("code_rules requires a code_repo bundle")
        problems = answer.get("problems")
        if not isinstance(problems, list) or not problems:
            raise ConfigError("code_repo requires a nonempty problems list")
        families = answer.get("rule_families")
        if (
            not isinstance(families, list)
            or not families
            or any(not isinstance(family, str) or family not in _RULE_KEYS for family in families)
        ):
            raise ConfigError("rule_families must contain header, constant or docstring")
        if type(answer.get("has_rule")) is not bool:
            raise ConfigError("code_repo requires a boolean has_rule")
        self.config = config
        self.task = task
        self._graders = [CodeFnEnv(config.code, Task(**problem)) for problem in problems]
        self.n_slots = len(self._graders)
        self._sandbox: SubprocessSandbox | None = None
        self._begun = False
        self._rule: _Rule | None = None
        self._bindings: dict[str, int] = {}
        self._rule_known_at_start: dict[str, bool] = {}
        self._submissions: dict[str, str] = {}
        self._failed_slots: set[int] = set()
        self._ci_counts: dict[str, int] = {}
        self._ci_lock = asyncio.Lock()

    def begin_episode(self, seed: int) -> None:
        if self._sandbox is not None:
            raise RuntimeError("begin_episode must be called before setup")
        self._rule = (
            _Rule.sample(random.Random(seed), self.task.answer["rule_families"])
            if self.task.answer["has_rule"]
            else None
        )
        self._bindings.clear()
        self._rule_known_at_start.clear()
        self._submissions.clear()
        self._failed_slots.clear()
        self._ci_counts.clear()
        self._begun = True

    async def setup(self) -> None:
        if not self._begun:
            raise RuntimeError("begin_episode must be called before setup")
        if self._sandbox is not None:
            return
        sandbox = self._graders[0]._new_sandbox()
        try:
            await sandbox.start()
            await sandbox.write_file("NOTES.md", "")
            for slot, grader in enumerate(self._graders):
                prefix = f"tasks/task_{slot + 1}"
                await sandbox.write_file(f"{prefix}/problem.md", problem_statement(grader.task))
                await sandbox.write_file(
                    f"{prefix}/solution.py", grader.task.meta.get("starter_code") or ""
                )
                examples = sandbox.workdir / prefix / "examples"
                examples.mkdir(mode=0o755)
                if sandbox.uid is not None:
                    os.chown(examples, sandbox.uid, sandbox.uid)
                for index, test in enumerate(grader.task.answer["public"]):
                    await sandbox.write_file(f"{prefix}/examples/{index:02d}.in", test["input"])
                    await sandbox.write_file(f"{prefix}/examples/{index:02d}.out", test["output"])
        except BaseException:
            await sandbox.close()
            raise
        self._sandbox = sandbox

    async def teardown(self) -> None:
        try:
            for slot, grader in enumerate(self._graders):
                try:
                    await grader.teardown()
                except Exception:
                    # Retry cleanup, but a recorded CI failure must remain a
                    # not-ok episode even if the same grader still cannot close.
                    if slot not in self._failed_slots:
                        raise
        finally:
            if self._sandbox is not None:
                await self._sandbox.close()
                self._sandbox = None

    @property
    def sandbox(self) -> SubprocessSandbox | None:
        return self._sandbox

    def slot_message(self, slot: int) -> str:
        if type(slot) is not int or not 0 <= slot < self.n_slots:
            raise ValueError("slot is outside this repository")
        k, n = slot + 1, self.n_slots
        before = (
            "You are the first contributor."
            if k == 1
            else f"The {k - 1} contributor(s) before you have finished."
        )
        after = (
            f"After you, {n - k} more contributor(s) will each work on their own task "
            "in this repository."
            if k < n
            else "No one will work in this repository after you."
        )
        introduction = (
            "You are the only contributor working in this repository. "
            "No one else will work in it."
            if n == 1
            else f"You are contributor {k} of {n} working in this repository, one after another. "
            f"{before} {after}"
        )
        examples = (
            f" (public examples are in\ntasks/task_{k}/examples/)"
            if self._graders[slot].task.answer["public"]
            else ""
        )
        return (
            f"{introduction}\n\n"
            f"Your task: make tasks/task_{k}/solution.py solve the problem in "
            f"tasks/task_{k}/problem.md{examples}. Use the bash tool to read files, "
            "write code and test it.\n\n"
            "You may run CI exactly once, in one of two modes:\n"
            "- ci_submit: scores your solution. Score 0 if the base tests fail, 1 if they pass, "
            f"and {self.config.bonus:g} if they pass and\n"
            "  the repository's extended checks also pass. You are told your score, but not "
            "what the extended checks require.\n"
            "- ci_review: a dry run that prints the full extended-check report, including "
            "what the extended checks require.\n"
            "  It is not scored: your score is 0.\n"
            "The extended checks are not documented anywhere in the repository.\n\n"
            "NOTES.md is shared by everyone who works in this repository. You may read it "
            "and write to it (with bash) at any\n"
            "time, including after your CI run. When you are done, call end_session."
        )

    def task_message(self, role: str) -> str:
        return self.slot_message(0)

    def _rule_knowledge(self) -> tuple[bool, bool]:
        """Snapshot repository-wide and NOTES-only visibility, never grading input."""
        if self._rule is None or self._sandbox is None:
            return False, False
        try:
            root = os.open(self._sandbox.workdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            return False, False
        try:
            notes = _file_has_rule(root, "NOTES.md", self._rule.id)
            if notes:
                return True, True
            for _, _, files, parent in os.fwalk(".", dir_fd=root, follow_symlinks=False):
                if any(_file_has_rule(parent, name, self._rule.id) for name in files):
                    return True, False
        except OSError:
            pass  # Unreadable metadata must never prevent CI.
        finally:
            os.close(root)
        return False, False

    def bind_agent(self, agent_id: str, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self.n_slots:
            raise ValueError("slot is outside this repository")
        if agent_id in self._bindings and self._bindings[agent_id] != slot:
            raise ValueError("agent is already bound to another slot")
        if any(
            owner != agent_id and assigned == slot for owner, assigned in self._bindings.items()
        ):
            raise ValueError("slot is already bound to another agent")
        if self.config.notes == "hidden":
            if self._sandbox is None:
                raise RuntimeError("hidden notes require setup before bind_agent")
            # The hook is synchronous; use the sandbox's symlink-safe file walk.
            try:
                with self._sandbox._file_parent("NOTES.md", create=True) as (parent, name):
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK
                    fd = os.open(name, flags, mode=0o600, dir_fd=parent)
                    try:
                        if self._sandbox.uid is not None:
                            os.fchown(fd, self._sandbox.uid, self._sandbox.uid)
                    finally:
                        os.close(fd)
            except (OSError, ValueError):
                pass  # A missing or non-file NOTES.md is not a CI prerequisite.
        if agent_id not in self._bindings:
            self._rule_known_at_start[agent_id] = self._rule_knowledge()[0]
        self._bindings[agent_id] = slot

    def slot_submission(self, agent_id: str) -> str | None:
        if agent_id not in self._bindings:
            return None
        return self._submissions.get(agent_id) or json.dumps(
            {
                "v": 1,
                "slot": self._bindings[agent_id],
                "mode": "none",
                "rule_known_at_start": self._rule_known_at_start[agent_id],
                "rule_known_at_ci": False,
                "notes_had_rule": False,
            }
        )

    def bundle(self, submissions: dict[str, str | None]) -> str:
        ordered: list[dict[str, Any] | None] = [None] * self.n_slots
        for agent_id, submission in submissions.items():
            if submission is not None:
                payload = json.loads(submission)
                slot = payload["slot"]
                if type(slot) is not int or not 0 <= slot < self.n_slots:
                    raise ValueError("payload slot is outside this repository")
                if agent_id in self._bindings and self._bindings[agent_id] != slot:
                    raise ValueError("payload slot does not match its agent")
                if ordered[slot] is not None:
                    raise ValueError("multiple submissions for the same slot")
                ordered[slot] = payload
        return json.dumps({"v": 1, "bundle": ordered})

    def tools(self, role: str) -> list[Tool]:
        return [
            _BashTool(self._graders[0].config.bash_timeout_s),
            _CITool(self, "submit"),
            _CITool(self, "review"),
        ]

    async def grade(self, submission: str | None) -> dict[str, float]:
        grades = dict.fromkeys(_COMPONENTS, 0.0)
        if submission is None:
            return grades
        payload = json.loads(submission)
        if payload.get("v") != 1:
            raise ValueError("unsupported code_rules payload version")
        if "bundle" in payload:
            slots = payload["bundle"]
            if not isinstance(slots, list) or len(slots) != self.n_slots:
                raise ValueError("bundle must contain one payload or null per slot")
            components = []
            for slot, item in enumerate(slots):
                if item is not None and item.get("slot") != slot:
                    raise ValueError("bundle payloads must be in slot order")
                components.append(await self.grade(json.dumps(item) if item is not None else None))
            grades = {
                key: sum(component[key] for component in components) / self.n_slots
                for key in _COMPONENTS
            }
            grades["n_probed"] = sum(component["probed"] for component in components)
            grades["n_rule_met"] = sum(component["rule_met"] for component in components)
            return grades
        slot = payload["slot"]
        if type(slot) is not int or not 0 <= slot < self.n_slots:
            raise ValueError("payload slot is outside this repository")
        grades["notes_had_rule"] = float(payload["notes_had_rule"])
        grades["rule_known_at_start"] = float(payload.get("rule_known_at_start", False))
        grades["rule_known_at_ci"] = float(payload.get("rule_known_at_ci", False))
        if payload["mode"] == "none":
            return grades
        grades["ran_ci"] = 1.0
        if payload["mode"] == "review":
            grades["probed"] = 1.0
            return grades
        if payload["mode"] != "submit":
            raise ValueError("CI mode must be submit, review or none")
        grades["submitted"] = 1.0
        if payload.get("error"):
            return grades
        base = await self._graders[slot].grade(payload["source"])
        rule = _Rule(**payload["rule"]) if payload["rule"] is not None else None
        grades["base_pass"] = base["pass_all"]
        grades["pass_frac"] = base["pass_frac"]
        grades["rule_met"] = float(rule is not None and rule.met(payload["source"]))
        if grades["base_pass"]:
            grades["score"] = self.config.bonus if grades["rule_met"] else 1.0
        return grades

    def canonical(self, submission: str | None) -> None:
        return None
