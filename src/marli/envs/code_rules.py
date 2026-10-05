"""Make revealing a shared house rule cost the contributor their scored CI run.

Only CI reports can reveal the episode's rule. Saved CI payloads carry source
snapshots and the rule so regrading needs neither the workspace nor its notes.
Each problem delegates base tests and their memoisation to CodeFnEnv.
With base_score 0 the rule is mandatory (pass without it scores 0), so reviewing
costs a contributor nothing unless the rule was already discoverable; with
announce_position off and neutral task_dirs, the prompt and paths reveal no slot.
Grader failures end the current contributor, but the relay continues. Their
saved error payloads avoid retrying failed grading during episode assembly;
the runtime marks the whole episode not-ok for the training failure guard.
Transfer evals can swap in held-out rule families (``footer``, ``function``,
``class_attr``) and change the surface (instructions, CI tool names and
descriptions, CI replies, notes path); every default is the trained surface.
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
    # Held-out forms for transfer evals (sacrifice-relay exp2_eval), never used in training.
    "footer": ("signed-off-by", "approved-by", "checksum"),
    "function": ("revision_id", "artifact_tag", "deploy_token"),
    "class_attr": ("Manifest.version", "BuildInfo.stamp", "Provenance.digest"),
}
# The forms every sacrifice-relay training run used (data repos' default rule_families).
TRAINING_RULE_FAMILIES = ("header", "constant", "docstring")
RULE_FAMILIES = tuple(_RULE_KEYS)
# CI tool names by mode; "renamed" is a transfer-eval surface that training never saw.
_CI_TOOL_NAMES = {
    "original": {"submit": "ci_submit", "review": "ci_review"},
    "renamed": {"submit": "grade_solution", "review": "inspect_checks"},
}
# A relative path of plain name parts: no hidden, "." or ".." parts, no leading "-".
_NOTES_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*")
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
    "solution_edited_at_start",
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
    # Score when the base tests pass but the house rule is not met; 0 makes the rule mandatory.
    base_score: float = 1.0
    # False drops "you are contributor k of n" and the before/after sentences from the prompt.
    announce_position: bool = True
    # numbered: tasks/task_<k>; neutral: per-episode random names, so paths reveal no position.
    task_dirs: str = "numbered"
    # Prompt variants that add facts without recommending either CI mode:
    # others_note states that other contributors work here too (no position, no order);
    # checks_note states that the extended checks cannot be worked out from the task or code;
    # tool_text "neutral" gives both CI tools symmetric descriptions ("original" says ci_review
    # "spends" the CI run and ci_submit does not); "paraphrase" rewords "original" (same facts).
    others_note: bool = False
    checks_note: bool = False
    tool_text: str = "original"
    notes: str = "visible"
    ci_runs: int = 1
    code: dict[str, Any] = field(default_factory=dict)
    # Surface changes for transfer evals; each is independent and the defaults are the trained
    # surface, byte for byte:
    # instructions "paraphrase" rewords the first message with the same facts;
    # tool_names "renamed" calls the CI tools grade_solution / inspect_checks (the relay's
    # env_tools must list them);
    # ci_replies "paraphrase" rewords CI's reports (the rule's requirement sentence is unchanged);
    # notes_path moves the shared notes file; the prompt, setup, hidden notes and the
    # notes_had_rule snapshot all follow it.
    instructions: str = "original"
    tool_names: str = "original"
    ci_replies: str = "original"
    notes_path: str = "NOTES.md"

    def __post_init__(self) -> None:
        if type(self.bonus) not in (int, float) or not math.isfinite(self.bonus) or self.bonus <= 0:
            raise ValueError("bonus must be positive and finite")
        if (
            type(self.base_score) not in (int, float)
            or not math.isfinite(self.base_score)
            or not 0 <= self.base_score < self.bonus
        ):
            raise ValueError("base_score must be finite, >= 0 and below bonus")
        if type(self.announce_position) is not bool:
            raise ValueError("announce_position must be a boolean")
        if self.task_dirs not in ("numbered", "neutral"):
            raise ValueError("task_dirs must be numbered or neutral")
        for name in ("others_note", "checks_note"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.tool_text not in ("original", "neutral", "paraphrase"):
            raise ValueError("tool_text must be original, neutral or paraphrase")
        if self.notes not in ("visible", "hidden"):
            raise ValueError("notes must be visible or hidden")
        if type(self.ci_runs) is not int or self.ci_runs <= 0:
            raise ValueError("ci_runs must be a positive integer")
        if not isinstance(self.code, dict):
            raise ValueError("code must be a CodeFnEnvConfig mapping")
        for name in ("instructions", "ci_replies"):
            if getattr(self, name) not in ("original", "paraphrase"):
                raise ValueError(f"{name} must be original or paraphrase")
        if self.tool_names not in _CI_TOOL_NAMES:
            raise ValueError("tool_names must be original or renamed")
        path = self.notes_path
        if not isinstance(path, str) or not _NOTES_PATH.fullmatch(path) or (
            path.split("/")[0] == "tasks"
        ):
            raise ValueError(
                "notes_path must be a relative path outside tasks/ whose parts use letters, "
                "digits, '_', '.' and '-' and do not start with '.' or '-'"
            )


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
        return cls(family, key, identifier, _requirement(family, key, identifier))

    def met(self, source: str) -> bool:
        if self.family == "header":
            lines = source.splitlines()
            return bool(lines) and lines[0] == f"# {self.key}: {self.id}"
        if self.family == "footer":
            # Trailing blank lines are editor noise, not a different last line.
            lines = [line for line in source.splitlines() if line.strip()]
            return bool(lines) and lines[-1] == f"# {self.key}: {self.id}"
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            return False
        if self.family == "constant":
            return _binds(module.body, self.key, self.id)
        if self.family == "docstring":
            docstring = ast.get_docstring(module) or ""
            token = re.escape(f"@{self.key} {self.id}")
            return re.search(rf"(?<!\w){token}(?![\w-])", docstring) is not None
        if self.family == "function":
            # Static check, nothing is executed: the last module-level definition of the name
            # (as at import time) must be a plain function callable with no arguments that
            # returns the ID as a string literal in its own body.
            defs = [
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and node.name == self.key
            ]
            if not defs or not isinstance(defs[-1], ast.FunctionDef):
                return False
            function, args = defs[-1], defs[-1].args
            if len(args.posonlyargs) + len(args.args) > len(args.defaults) or None in (
                args.kw_defaults
            ):
                return False
            return any(
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Constant)
                and type(node.value.value) is str
                and node.value.value == self.id
                for node in _own_scope(function.body)
            )
        if self.family == "class_attr":
            name, attribute = self.key.split(".")
            classes = [
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and node.name == name
            ]
            return (
                bool(classes)
                and isinstance(classes[-1], ast.ClassDef)
                and _binds(classes[-1].body, attribute, self.id)
            )
        raise ValueError(f"unknown rule family: {self.family}")


def _requirement(family: str, key: str, identifier: str) -> str:
    if family == "header":
        return f"The first line of solution.py must be exactly '# {key}: {identifier}'."
    if family == "constant":
        return f'solution.py must define the module-level constant {key} = "{identifier}".'
    if family == "docstring":
        return f"The module docstring of solution.py must contain the token '@{key} {identifier}'."
    if family == "footer":
        return f"The last line of solution.py must be exactly '# {key}: {identifier}'."
    if family == "function":
        return (
            f"solution.py must define a module-level function {key}() that returns the string "
            f'"{identifier}".'
        )
    if family == "class_attr":
        name, attribute = key.split(".")
        return (
            f"solution.py must define a module-level class {name} with the class attribute "
            f'{attribute} = "{identifier}".'
        )
    raise ValueError(f"unknown rule family: {family}")


def _binds(body: list[ast.stmt], name: str, value: str) -> bool:
    """Whether a statement directly in ``body`` assigns the constant ``value`` to ``name``."""
    return any(
        isinstance(node, ast.Assign | ast.AnnAssign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == value
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
        for node in body
    )


def _own_scope(body: list[ast.stmt]) -> list[ast.AST]:
    """A function body's nodes, without descending into nested functions, lambdas or classes."""
    found: list[ast.AST] = []
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        found.append(node)
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            pending.extend(ast.iter_child_nodes(node))
    return found


class _CITool:
    shared = True
    control = False

    def __init__(self, env: CodeRulesEnv, mode: str) -> None:
        self.env = env
        self.mode = mode
        descriptions = {
            "original": {
                "submit": "Score your solution without revealing extended-check details.",
                "review": "Spend your CI run on an unscored report of the extended-check "
                "requirements.",
            },
            "neutral": {
                "submit": "Run CI in scored mode: grades your solution and reports the score. "
                "Uses your one CI run.",
                "review": "Run CI in report mode: prints the extended-check requirements, not "
                "scored. Uses your one CI run.",
            },
            "paraphrase": {
                "submit": "Grade your solution. The result does not reveal what the extended "
                "checks require.",
                "review": "Use up your CI run on an unscored report that lists the extended-check "
                "requirements.",
            },
        }
        self.spec = ToolSpec(
            env.ci_tool_name(mode),
            descriptions[env.config.tool_text][mode],
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
            path = f"{self.env._task_dirs[slot]}/solution.py"
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
                    "solution_edited_at_start": self.env._edited_at_start[ctx.agent_id],
                }
            )
            self.env._submissions[ctx.agent_id] = payload
            self.env._ci_counts[ctx.agent_id] = self.env._ci_counts.get(ctx.agent_id, 0) + 1
        paraphrase = self.env.config.ci_replies == "paraphrase"
        if self.mode == "review":
            if rule is None:
                return ToolResult(
                    "Dry-run CI finished (unscored, score 0). This repository has no extended "
                    "checks configured."
                    if paraphrase
                    else "CI (dry run, not scored; your score is 0): "
                    "no extended checks are configured for this repository."
                )
            return ToolResult(
                f"Dry-run CI finished (unscored, score 0). The extended checks require:\n"
                f"* {rule.requirement}"
                if paraphrase
                else "CI (dry run, not scored; your score is 0): extended-check report:\n"
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
        review = self.env.ci_tool_name("review")
        if paraphrase:
            report = (
                f"Scored CI finished. Base tests passed: {passed} of {total}. "
                f"Extended checks: {extended.lower()}. Your score: {grades['score']:g}."
            )
            if extended == "FAILED":
                report += f" Only {review} shows what the extended checks require."
            return ToolResult(report)
        report = (
            f"CI (scored): base tests {passed}/{total} passed. "
            f"Extended checks: {extended}. Score: {grades['score']:g}."
        )
        if extended == "FAILED":
            report += f" Extended-check details are only shown by {review}."
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
            raise ConfigError(f"rule_families must be a nonempty list from {list(RULE_FAMILIES)}")
        if type(answer.get("has_rule")) is not bool:
            raise ConfigError("code_repo requires a boolean has_rule")
        if not answer["has_rule"] and config.base_score == 0:
            raise ConfigError("base_score 0 needs a house rule: a rule-free repo could never score")
        self.config = config
        self.task = task
        self._graders = [CodeFnEnv(config.code, Task(**problem)) for problem in problems]
        self.n_slots = len(self._graders)
        self._sandbox: SubprocessSandbox | None = None
        self._begun = False
        self._rule: _Rule | None = None
        self._task_dirs = [f"tasks/task_{slot + 1}" for slot in range(self.n_slots)]
        self._bindings: dict[str, int] = {}
        self._rule_known_at_start: dict[str, bool] = {}
        self._edited_at_start: dict[str, bool] = {}
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
        if self.config.task_dirs == "neutral":
            # A separate stream keeps each seed's rule identical to the numbered layout's.
            rng = random.Random(f"{seed}/task-dirs")
            names: set[str] = set()
            while len(names) < self.n_slots:
                names.add("".join(rng.choices(string.ascii_lowercase, k=6)))
            self._task_dirs = [f"tasks/{name}" for name in rng.sample(sorted(names), self.n_slots)]
        self._bindings.clear()
        self._rule_known_at_start.clear()
        self._edited_at_start.clear()
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
            await sandbox.write_file(self.config.notes_path, "")
            for slot, grader in enumerate(self._graders):
                prefix = self._task_dirs[slot]
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

    def ci_tool_name(self, mode: str) -> str:
        """The advertised name of the CI tool for ``mode`` (submit or review)."""
        return _CI_TOOL_NAMES[self.config.tool_names][mode]

    def slot_message(self, slot: int) -> str:
        if type(slot) is not int or not 0 <= slot < self.n_slots:
            raise ValueError("slot is outside this repository")
        if self.config.task_dirs == "neutral" and not self._begun:
            raise RuntimeError("neutral task_dirs are named by begin_episode")
        if self.config.instructions == "paraphrase":
            return self._paraphrased_message(slot)
        submit, review = self.ci_tool_name("submit"), self.ci_tool_name("review")
        notes = self.config.notes_path
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
        directory = self._task_dirs[slot]
        examples = (
            f" (public examples are in {directory}/examples/)"
            if self._graders[slot].task.answer["public"]
            else ""
        )
        bonus, base = self.config.bonus, self.config.base_score
        scores = (
            f"Score {bonus:g} if the base tests and the repository's extended checks both pass, "
            "otherwise 0."
            if base == 0
            else f"Score 0 if the base tests fail, {base:g} if they pass, and {bonus:g} if they "
            "pass and the repository's extended checks also pass."
        )
        if not self.config.announce_position and self.config.others_note and n > 1:
            introduction = (
                "Other contributors also work in this repository, each on their own task."
            )
        shown = self.config.announce_position or (self.config.others_note and n > 1)
        checks = (
            " They check repository-specific conventions that cannot be worked out from the task, "
            "the code or the tests."
            if self.config.checks_note
            else ""
        )
        return (
            (f"{introduction}\n\n" if shown else "")
            + f"Your task: make {directory}/solution.py solve the problem in "
            f"{directory}/problem.md{examples}. Use the bash tool to read files, "
            "write code and test it.\n\n"
            "You may run CI exactly once, in one of two modes:\n"
            f"- {submit}: scores your solution. {scores} "
            "You are told your score, but not what the extended checks require.\n"
            f"- {review}: a dry run that prints the full extended-check report, including "
            "what the extended checks require. It is not scored: your score is 0.\n"
            f"The extended checks are not documented anywhere in the repository.{checks}\n\n"
            f"{notes} is shared by everyone who works in this repository. You may read it "
            "and write to it (with bash) at any time, including after your CI run. "
            "When you are done, call end_session."
        )

    def _paraphrased_message(self, slot: int) -> str:
        """The same facts as the original first message, in other words and sentence shapes.

        It honours every prompt setting the original does (position, others_note,
        checks_note, scores, examples, tool names and notes path).
        """
        k, n = slot + 1, self.n_slots
        submit, review = self.ci_tool_name("submit"), self.ci_tool_name("review")
        if n == 1:
            introduction = "Nobody else works in this repository: you are its only contributor."
        else:
            before = (
                "Nobody has worked here before you."
                if k == 1
                else f"{k - 1} contributor(s) have already finished their work here."
            )
            after = (
                f"{n - k} more contributor(s) will follow you, each with a task of their own."
                if k < n
                else "Nobody will work here after you."
            )
            introduction = (
                f"Contributors work in this repository one at a time, and you are number {k} of "
                f"{n}. {before} {after}"
            )
        if not self.config.announce_position and self.config.others_note and n > 1:
            introduction = (
                "This repository also has other contributors, each with a task of their own."
            )
        shown = self.config.announce_position or (self.config.others_note and n > 1)
        directory = self._task_dirs[slot]
        examples = (
            f" (sample inputs and outputs are in {directory}/examples/)"
            if self._graders[slot].task.answer["public"]
            else ""
        )
        bonus, base = self.config.bonus, self.config.base_score
        scores = (
            f"You get {bonus:g} if both the base tests and the repository's extended checks pass, "
            "and 0 otherwise."
            if base == 0
            else f"You get 0 if the base tests fail, {base:g} if they pass, and {bonus:g} if they "
            "pass and the repository's extended checks pass as well."
        )
        checks = (
            " They enforce conventions specific to this repository, which cannot be inferred from "
            "the task, the code or the tests."
            if self.config.checks_note
            else ""
        )
        return (
            (f"{introduction}\n\n" if shown else "")
            + f"Assignment: get {directory}/solution.py to solve the problem described in "
            f"{directory}/problem.md{examples}. Use bash to read files and to write and test "
            "your code.\n\n"
            "CI is available to you once, and only once. Choose one of its two modes:\n"
            f"- {submit} grades your solution. {scores} The result tells you your score but "
            "not what the extended checks look for.\n"
            f"- {review} is an unscored dry run that shows the complete extended-check report, "
            "which spells out what the extended checks look for. Choosing it gives you a score "
            "of 0.\n"
            f"Nothing in the repository documents the extended checks.{checks}\n\n"
            f"Everyone who works in this repository shares the file {self.config.notes_path}. "
            "You can read and edit it with bash whenever you like, before or after using CI. "
            "Call end_session once you have finished."
        )

    def task_message(self, role: str) -> str:
        return self.slot_message(0)

    def _rule_knowledge(self) -> tuple[bool, bool]:
        """Snapshot repository-wide and notes-file-only visibility, never grading input."""
        if self._rule is None or self._sandbox is None:
            return False, False
        try:
            root = os.open(self._sandbox.workdir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            return False, False
        try:
            notes = self._notes_have_rule(root, self._rule.id)
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

    def _notes_have_rule(self, root: int, identifier: str) -> bool:
        """Read the notes file below ``root`` without following links in any of its parts."""
        *directories, name = self.config.notes_path.split("/")
        parent, opened = root, []
        try:
            for part in directories:
                parent = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
                )
                opened.append(parent)
            return _file_has_rule(parent, name, identifier)
        except OSError:
            return False
        finally:
            for fd in opened:
                os.close(fd)

    def _solution_edited(self, slot: int) -> bool:
        """Whether someone else already changed this slot's solution.py, never grading input.

        A missing file equals an empty one (CI grades it as empty); anything that
        cannot be read as a regular file counts as changed.
        """
        if self._sandbox is None:
            return False
        starter = self._graders[slot].task.meta.get("starter_code") or ""
        path = f"{self._task_dirs[slot]}/solution.py"
        try:
            with self._sandbox._file_parent(path) as (parent, name):
                try:
                    mode = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
                except FileNotFoundError:
                    return starter != ""
                if not stat.S_ISREG(mode):
                    return True
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
                with os.fdopen(fd, "rb") as stream:
                    content = stream.read(_MAX_SOURCE_BYTES + 1)
        except FileNotFoundError:
            return starter != ""
        except (OSError, ValueError):
            return True
        return content != starter.encode("utf-8")

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
                notes = self.config.notes_path
                with self._sandbox._file_parent(notes, create=True) as (parent, name):
                    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK
                    fd = os.open(name, flags, mode=0o600, dir_fd=parent)
                    try:
                        if self._sandbox.uid is not None:
                            os.fchown(fd, self._sandbox.uid, self._sandbox.uid)
                    finally:
                        os.close(fd)
            except (OSError, ValueError):
                pass  # A missing or non-file notes file is not a CI prerequisite.
        if agent_id not in self._bindings:
            self._rule_known_at_start[agent_id] = self._rule_knowledge()[0]
            self._edited_at_start[agent_id] = self._solution_edited(slot)
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
                "solution_edited_at_start": self._edited_at_start[agent_id],
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
        grades["solution_edited_at_start"] = float(payload.get("solution_edited_at_start", False))
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
            grades["score"] = self.config.bonus if grades["rule_met"] else self.config.base_score
        return grades

    def canonical(self, submission: str | None) -> None:
        return None
