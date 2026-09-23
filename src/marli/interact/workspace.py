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
  The first notification includes only the latest version per writer.
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

import re
from dataclasses import dataclass, field, replace

from marli.errors import ConfigError
from marli.interact.tools import ToolError
from marli.interact.types import ReadVia, WorkspaceRead, Write


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

    def __post_init__(self) -> None:
        if self.mode not in {"notify", "push", "pull"}:
            raise ConfigError(f"unknown delivery mode: {self.mode!r}")
        if self.view not in {"latest", "full"}:
            raise ConfigError(f"unknown delivery view: {self.view!r}")
        if self.push_max_chars <= 64:
            raise ConfigError("push_max_chars must be greater than 64")
        if self.index_first_line_chars < 0:
            raise ConfigError("index_first_line_chars must be non-negative")


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
    """Versioned state with private notes and scheduler-controlled visibility."""

    def __init__(
        self,
        *,
        roles: dict[str, str],
        permissions: dict[str, Permissions],
        delivery: DeliverySpec,
        staged: bool,
        notes_cap_chars: int = 4000,
    ) -> None:
        """Create an episode workspace; copy role maps so permissions stay stable."""
        if notes_cap_chars < 0:
            raise ConfigError("notes_cap_chars must be non-negative")
        self.delivery = delivery
        self.staged = staged
        self.notes_cap_chars = notes_cap_chars
        self._roles: dict[str, str] = {}
        self._permissions = dict(permissions)
        self._history: dict[tuple[str, str], list[Write]] = {}
        self._staged: dict[str, list[Write]] = {}
        self._log: list[Write] = []
        self._last_seen: dict[str, dict[tuple[str, str], int]] = {}
        for agent_id, role in roles.items():
            self.add_agent(agent_id, role)

    def add_agent(self, agent_id: str, role: str) -> None:
        """Register a dynamic worker with an existing role's permissions."""
        if agent_id in self._roles:
            raise ConfigError(f"agent {agent_id!r} is already registered")
        if role not in self._permissions:
            raise ConfigError(f"no workspace permissions for role {role!r}")
        self._roles[agent_id] = role

    def write(
        self,
        writer: str,
        key: str,
        content: str,
        *,
        mode: str,
        tick: int | None,
        seq: int,
    ) -> Write:
        """Append or overwrite, returning the full resulting content in a Write.

        Staged records carry the prospective per-key version for tool feedback;
        only ``commit_staged`` assigns committed versions and extends the log.
        Appends include earlier staged writes by the same writer.
        """
        permissions = self._agent_permissions(writer)
        self._validate_key(key)
        if key == "scratchpad" and not permissions.write_scratchpad:
            raise ToolError(f"agent {writer!r} may not write scratchpad")
        if key == "notes" and not permissions.notes:
            raise ToolError(f"agent {writer!r} may not write notes")
        if mode not in {"append", "overwrite"}:
            raise ToolError(f"unknown write mode {mode!r}; expected append or overwrite")
        history = self._history.get((writer, key), [])
        pending = [write for write in self._staged.get(writer, []) if write.key == key]
        previous = pending[-1] if pending else history[-1] if history else None
        if mode == "append" and previous is not None:
            content = previous.content + "\n" + content
        if key == "notes" and len(content) > self.notes_cap_chars:
            raise ToolError(f"notes exceed the cap of {self.notes_cap_chars} chars")
        write = Write(writer, key, len(history) + len(pending) + 1, content, tick, seq)
        if self.staged:
            self._staged.setdefault(writer, []).append(write)
            return write
        return self._commit(write)

    def commit_staged(self, seat_order: list[str]) -> list[Write]:
        """Commit listed agents in seat order, retaining each agent's write order.

        Unlisted agents' writes remain staged. Immediate workspaces return an
        empty list. Versions are assigned from committed history here.
        """
        for agent_id in seat_order:
            self._agent_permissions(agent_id)
        committed = []
        for agent_id in seat_order:
            for write in self._staged.pop(agent_id, []):
                committed.append(self._commit(write))
        return committed

    def has_staged(self) -> bool:
        """Whether any writes still await a scheduler commit or episode-end flush."""
        return bool(self._staged)

    def flush_staged(self, seat_order: list[str]) -> list[Write]:
        """Commit all pending writes, including agents omitted from a partial order.

        Listed agents commit first in seat order; remaining agents follow in
        agent-id order. Each agent's writes retain their original order.
        """
        remaining = sorted(self._staged.keys() - set(seat_order))
        return self.commit_staged([*seat_order, *remaining])

    def view(self, reader: str) -> WorkspaceView:
        """Snapshot readable committed history and the reader's own staged content.

        The dictionaries are fresh copies: changing a view cannot change the
        workspace or another ticket's snapshot.
        """
        self._agent_permissions(reader)
        versions = {}
        contents = {}
        for (writer, key), history in self._history.items():
            if self._can_read(reader, writer, key):
                versions[writer, key] = history[-1].version
                for write in history:
                    contents[writer, key, write.version] = write.content
        return WorkspaceView(
            versions,
            contents,
            {write.key: write.content for write in self._staged.get(reader, [])},
        )

    def read(
        self, reader: str, writer: str, key: str = "scratchpad", version: int | None = None
    ) -> tuple[str, int]:
        """Read committed state or the writer's own latest staged version.

        Zero or never written is empty. An unknown harness reader raises
        ValueError; an unknown target writer, forbidden reads and nonexistent
        versions raise ToolError. Own pending content is returned for the latest
        read or its exact pending version.
        Reading does not acknowledge pending notify/push deliveries.
        """
        self._agent_permissions(reader)
        if writer not in self._roles:
            raise ToolError(f"unknown agent {writer!r}")
        self._validate_key(key)
        if not self._can_read(reader, writer, key):
            if key == "notes":
                raise ToolError(f"notes are private to agent {writer!r}")
            raise ToolError(f"agent {reader!r} may not read {writer!r}'s {key}")
        if version is not None and (type(version) is not int or version < 0):
            raise ToolError("version must be a non-negative integer")
        if reader == writer:
            pending = next(
                (write for write in reversed(self._staged.get(reader, [])) if write.key == key),
                None,
            )
            if pending is not None and (version is None or version == pending.version):
                return pending.content, pending.version
        history = self._history.get((writer, key), [])
        if version == 0 or not history:
            return "", 0
        if version is None:
            write = history[-1]
        elif version > len(history):
            raise ToolError(f"no version {version} of {writer!r}'s {key}")
        else:
            write = history[version - 1]
        return write.content, write.version

    def list_index(self, reader: str) -> list[IndexEntry]:
        """List every readable scratchpad in agent registration order (empty is v0)."""
        self._agent_permissions(reader)
        entries = []
        for writer in self._roles:
            if self._can_read(reader, writer, "scratchpad"):
                content, version = self.read(reader, writer)
                entries.append(
                    IndexEntry(
                        writer, "scratchpad", version, len(content), self._first_line(content)
                    )
                )
        return entries

    def pending_delivery(self, reader: str) -> tuple[str | None, list[WorkspaceRead]]:
        """Deliver unseen readable peer scratchpads and acknowledge the versions.

        The first notify collapses history to the latest version per writer;
        later notifies list every unseen version. Push selects all versions or
        the latest per writer, in commit order. Oversized blocks keep their
        headers and a marked content tail; whole blocks are then dropped
        oldest-first to fit the total cap. Omitted versions are acknowledged
        too; read records describe only blocks whose headers are delivered.
        """
        self._agent_permissions(reader)
        if self.delivery.mode == "pull":
            return None, []
        first_delivery = reader not in self._last_seen
        seen = self._last_seen.get(reader, {})
        writes = [
            write
            for write in self._log
            if write.writer != reader
            and write.key == "scratchpad"
            and self._can_read(reader, write.writer, write.key)
            and write.version > seen.get((write.writer, write.key), 0)
        ]
        if not writes:
            return None, []
        self._last_seen[reader] = seen
        for write in writes:
            seen[write.writer, write.key] = write.version
        if (self.delivery.mode == "notify" and first_delivery) or (
            self.delivery.mode == "push" and self.delivery.view == "latest"
        ):
            writes = [write for write in writes if write.version == seen[write.writer, write.key]]
        if self.delivery.mode == "notify":
            text = "\n".join(
                f"[workspace] {write.writer} wrote {write.key} v{write.version} "
                f"({len(write.content)} chars): {self._first_line(write.content)}"
                for write in writes
            )
            return text, [
                WorkspaceRead(write.writer, write.key, write.version, ReadVia.NOTIFY)
                for write in writes
            ]
        blocks = []
        marker = "…[truncated]"
        for write in writes:
            header = f"[workspace] {write.writer} {write.key} v{write.version}:\n"
            content = write.content
            if len(header) + len(content) > self.delivery.push_max_chars:
                keep = max(0, self.delivery.push_max_chars - len(header) - len(marker))
                content = marker + (content[-keep:] if keep else "")
            blocks.append(header + content)
        total = sum(map(len, blocks)) + 2 * (len(blocks) - 1)
        start = 0
        while start < len(blocks) and total > self.delivery.push_max_chars:
            total -= len(blocks[start]) + (2 if start + 1 < len(blocks) else 0)
            start += 1
        reads = [
            WorkspaceRead(write.writer, write.key, write.version, ReadVia.PUSH)
            for write in writes[start:]
        ]
        return "\n\n".join(blocks[start:]) or None, reads

    def log(self) -> tuple[Write, ...]:
        """Return committed writes in commit order, excluding staged records."""
        return tuple(self._log)

    def _commit(self, write: Write) -> Write:
        history = self._history.setdefault((write.writer, write.key), [])
        write = replace(write, version=len(history) + 1)
        history.append(write)
        self._log.append(write)
        return write

    def _agent_permissions(self, agent_id: str) -> Permissions:
        if agent_id not in self._roles:
            raise ValueError(f"unknown agent {agent_id!r}")
        return self._permissions[self._roles[agent_id]]

    def _can_read(self, reader: str, writer: str, key: str) -> bool:
        if reader == writer:
            return True
        if key == "notes":
            return False
        permissions = self._permissions[self._roles[reader]]
        return permissions.read_others and (
            not permissions.readable_roles or self._roles[writer] in permissions.readable_roles
        )

    @staticmethod
    def _validate_key(key: str) -> None:
        if not re.fullmatch(r"[a-z0-9_./-]+", key):
            raise ToolError(f"invalid workspace key {key!r}; expected [a-z0-9_./-]+")

    def _first_line(self, content: str) -> str:
        lines = content.splitlines()
        return lines[0][: self.delivery.index_first_line_chars] if lines else ""
