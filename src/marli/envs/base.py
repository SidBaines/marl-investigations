"""Environment contract: tasks, env tools, sandboxes, grading.

An ``Env`` is created per episode from an env config and a ``Task``. It
contributes the task prompt, environment tools (e.g. ``python`` for math,
``bash`` for coding) and grading. It owns the episode's sandbox (if any),
which persists across a multi-session agent's sessions and is shared by
workers (``worker_sandbox: shared`` in v1).

Grading sees **only** the submission and the sandbox — never scratchpads,
notes or transcripts — so agents cannot leak answers into what is graded.
Hidden tests are restored (coding) before grading.

``grade`` returns raw components (e.g. ``{"correct": 1.0, "format": 1.0}`` for
math, ``{"pass_all": 0.0, "pass_frac": 0.8}`` for code). Rewards are computed
from these by the training loop (train/credit.py) — never stored at rollout.
``canonical`` returns the verifier-equivalence key used for votes (math: a
normalized math-verify form; MCQ: the letter; code: None — votes are not
defined for code).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str  # problem statement shown to agents
    # reference (math answer / hidden-test bundle ref); never shown to agents
    answer: Any = None
    meta: dict[str, Any] = field(default_factory=dict)  # source, difficulty, split, ...


class Env(ABC):
    name: str
    supports_vote: bool = True

    @abstractmethod
    async def setup(self) -> None: ...  # create the sandbox if the env needs one

    @abstractmethod
    async def teardown(self) -> None: ...  # always called (finally)

    @abstractmethod
    def task_message(self, role: str) -> str: ...  # the problem as presented to a given role

    @abstractmethod
    def tools(self, role: str) -> list[Any]:
        """Trusted environment tools override built-ins with the same advertised name.

        The runtime preserves built-in control semantics. Overrides must return
        the corresponding control effect, e.g. ``control={"submit": source}``.
        """
        ...

    @abstractmethod
    async def grade(self, submission: str | None) -> dict[str, float]: ...

    @abstractmethod
    def canonical(self, submission: str | None) -> str | None: ...

    async def same_answer(self, a: str | None, b: str | None) -> bool:
        """Verifier equivalence of two submissions (used by vote aggregation).

        Default: equal non-None canonical keys. Math envs override this with the
        math verifier (pairwise ``verify``), which is more robust than canonical
        strings for LaTeX answers."""
        ka, kb = self.canonical(a), self.canonical(b)
        return ka is not None and ka == kb

    @property
    def sandbox(self) -> Any | None:  # envs.sandbox.base.Sandbox or None
        return None

    def begin_episode(self, seed: int) -> None:
        """Called by the runtime with the episode's seed before ``setup()``.

        Envs that sample hidden per-episode state (e.g. code_rules' house rule)
        derive it from this seed so repeated tasks never repeat hidden state.
        Default: no-op.
        """
        return None

    # --- Optional per-agent slots -------------------------------------------------
    # A *slotted* env gives each agent its own sub-task (e.g. one problem per
    # contributor in a shared repo; interact/protocols/relay.py). Envs with
    # ``n_slots == 0`` present the same task to every agent and are graded from
    # the built-in ``submit``. A slotted env's graded submissions come from the
    # env itself (``slot_submission``), because its terminal actions (e.g. a CI
    # run) need not end the agent's turn.
    n_slots: int = 0

    def slot_message(self, slot: int) -> str:
        """The task as presented to the agent that owns ``slot`` (0-based)."""
        raise NotImplementedError(f"{type(self).__name__} is not a slotted env")

    def bind_agent(self, agent_id: str, slot: int) -> None:
        """Called by the protocol before starting the agent that owns ``slot``."""
        raise NotImplementedError(f"{type(self).__name__} is not a slotted env")

    def slot_submission(self, agent_id: str) -> str | None:
        """That agent's graded submission (e.g. its CI payload), or None."""
        raise NotImplementedError(f"{type(self).__name__} is not a slotted env")

    def bundle(self, submissions: dict[str, str | None]) -> str | None:
        """The system submission graded as ``_system`` (e.g. all agents' payloads,
        graded as the team mean). Keys are agent ids in slot order."""
        raise NotImplementedError(f"{type(self).__name__} is not a slotted env")


from marli.envs.registry import ENVS as ENVS  # noqa: E402
