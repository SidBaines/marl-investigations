"""Small local environments keep runtime integration tests independent of backends."""

from __future__ import annotations

from marli.envs.base import Env, Task
from marli.interact.tools import Tool


class ArithEnv(Env):
    name = "arith"

    def __init__(self, a: int = 2, b: int = 3) -> None:
        self.task = Task("arithmetic", f"What is {a}+{b}?", answer=a + b)
        self.setup_count = 0
        self.teardown_count = 0
        self.graded: list[str | None] = []

    async def setup(self) -> None:
        self.setup_count += 1

    async def teardown(self) -> None:
        self.teardown_count += 1

    def task_message(self, role: str) -> str:
        return self.task.prompt

    def tools(self, role: str) -> list[Tool]:
        return []

    async def grade(self, submission: str | None) -> dict[str, float]:
        self.graded.append(submission)
        return {"correct": float(self.canonical(submission) == str(self.task.answer))}

    def canonical(self, submission: str | None) -> str | None:
        if submission is None:
            return None
        text = submission.strip()
        try:
            return str(int(text))
        except ValueError:
            return text


class ScratchEnv(ArithEnv):
    name = "scratch"

    async def grade(self, submission: str | None) -> dict[str, float]:
        self.graded.append(submission)
        return {}
