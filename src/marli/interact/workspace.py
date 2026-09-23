"""The shared workspace of one episode: scratchpads, notes, artifacts.

Each agent owns a **scratchpad** (append or overwrite) and optionally
**notes** (a capped file that persists across a multi-session agent's
sessions). Every write is versioned per ``(writer, key)`` and logged
(``Episode.workspace_log``). Visibility is governed by per-role
``Permissions`` set by the protocol (e.g. swarm peers read all scratchpads;
a coordinator reads its workers'; workers read only their own).

Commit semantics come from the scheduler: under lockstep, writes are *staged*
and committed at tick end in seat order (``commit_staged``); under async they
commit immediately. Readers always see committed state plus their own staged
writes. ``WorkspaceView`` is the immutable snapshot a ``Ticket`` carries.

How others' state reaches an agent (``delivery``, configured per protocol):

- ``notify`` (default): each turn the harness pushes a compact index of
  writes the reader hasn't seen — ``(writer, key, version, n_chars,
  first line)`` — and the agent pulls contents with ``read_scratchpad``.
- ``push``: each turn the harness pushes the new content itself
  (``view``: ``latest`` = newest version per writer, ``full`` = every unseen
  version, capped by ``push_max_chars``).
- ``pull``: nothing is pushed; agents use ``list_scratchpads`` /
  ``read_scratchpad`` themselves.

``last_seen`` bookkeeping is per reader, so pushes are deltas and the agent's
append-only token buffer never has history rewritten. Every delivery (push,
notify entry, pull) becomes a ``WorkspaceRead`` on the call whose prompt
contains it — the raw material for cross-read metrics.

Graders never receive the workspace (``env.grade(submission, sandbox)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from marli.interact.types import WorkspaceRead, Write  # noqa: F401  (referenced in the contract)


@dataclass(frozen=True)
class Permissions:
    read_others: bool = True  # may read other agents' scratchpads
    readable_roles: tuple[str, ...] = ()  # if non-empty, restrict reads to these writer roles
    write_scratchpad: bool = True
    notes: bool = False  # has a private notes file (multi-session carry)


@dataclass(frozen=True)
class DeliverySpec:
    mode: str = "notify"  # notify | push | pull
    view: str = "latest"  # latest | full   (push payload shape)
    push_max_chars: int = 6000
    index_first_line_chars: int = 120


@dataclass(frozen=True)
class IndexEntry:
    writer: str
    key: str
    version: int
    n_chars: int
    first_line: str


@dataclass(frozen=True)
class WorkspaceView:
    """Immutable committed state (+ the reader's own staged writes) for one turn."""

    versions: dict[tuple[str, str], int]  # (writer, key) -> latest committed version
    # (writer, key, version) -> content (committed only)
    contents: dict[tuple[str, str, int], str]
    # key -> content staged by the reader this tick
    own_staged: dict[str, str] = field(default_factory=dict)


class Workspace:
    """Methods (M1-6):

    __init__(*, roles: dict[agent_id, role], permissions: dict[role, Permissions], delivery:
    DeliverySpec,
             staged: bool)                              # staged=True under lockstep
    add_agent(agent_id, role) -> None                   # dynamic workers
    write(writer, key, content, *, mode: "append"|"overwrite", tick, seq) -> Write
    commit_staged(seat_order: list[agent_id]) -> list[Write]   # lockstep tick end; no-op if not
    staged
    view(reader) -> WorkspaceView
    read(reader, writer, key="scratchpad", version=None) -> tuple[str, int]  # permission-checked;
    raises ToolError
    list_index(reader) -> list[IndexEntry]              # everything readable
    pending_delivery(reader) -> tuple[str | None, list[WorkspaceRead]]
        # the notify/push message text for this reader's next turn (None if nothing new) and the
        # WorkspaceRead records it implies; advances last_seen. mode=pull -> (None, [])
    log() -> tuple[Write, ...]
    """
