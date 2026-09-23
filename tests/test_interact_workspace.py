"""Workspace history and deliveries must preserve visibility and deterministic order."""

from __future__ import annotations

from typing import Any

import pytest

from marli.errors import ConfigError
from marli.interact.tools import ToolError
from marli.interact.types import ReadVia, WorkspaceRead, Write
from marli.interact.workspace import DeliverySpec, IndexEntry, Permissions, Workspace


@pytest.fixture
def workspace() -> Workspace:
    return Workspace(
        roles={"alice": "peer", "bob": "peer", "worker": "worker"},
        permissions={"peer": Permissions(notes=True), "worker": Permissions()},
        delivery=DeliverySpec(),
        staged=False,
    )


def test_versions_append_overwrite_and_committed_log(workspace: Workspace) -> None:
    assert workspace.notes_cap_chars == 4000
    assert workspace.read("alice", "bob") == ("", 0)
    assert workspace.read("alice", "bob", version=12) == ("", 0)
    first = workspace.write("bob", "scratchpad", "one", mode="append", tick=None, seq=1)
    second = workspace.write("bob", "scratchpad", "two", mode="append", tick=None, seq=2)
    third = workspace.write("bob", "scratchpad", "new", mode="overwrite", tick=None, seq=3)
    assert first == Write("bob", "scratchpad", 1, "one", None, 1)
    assert second == Write("bob", "scratchpad", 2, "one\ntwo", None, 2)
    assert third == Write("bob", "scratchpad", 3, "new", None, 3)
    assert workspace.read("alice", "bob", version=1) == ("one", 1)
    assert workspace.read("alice", "bob", version=2) == ("one\ntwo", 2)
    assert workspace.read("alice", "bob") == ("new", 3)
    assert workspace.read("alice", "bob", version=0) == ("", 0)
    assert workspace.log() == (first, second, third)
    assert workspace.commit_staged(["bob", "alice"]) == []
    assert workspace.view("bob").own_staged == {}


def test_versions_are_per_writer_and_key(workspace: Workspace) -> None:
    for writer, key in [
        ("alice", "scratchpad"),
        ("alice", "notes"),
        ("alice", "artifacts/result_1-2.txt"),
        ("bob", "scratchpad"),
    ]:
        result = workspace.write(writer, key, "", mode="append", tick=0, seq=0)
        assert result.version == 1
        appended = workspace.write(writer, key, "next", mode="append", tick=0, seq=1)
        assert appended.version == 2
        assert appended.content == "\nnext"


def test_staged_visibility_and_seat_order(workspace: Workspace) -> None:
    workspace.staged = True
    before = workspace.view("alice")
    workspace.write("bob", "scratchpad", "b1", mode="append", tick=2, seq=10)
    workspace.write("alice", "scratchpad", "a1", mode="append", tick=2, seq=11)
    workspace.write("bob", "notes", "private", mode="overwrite", tick=2, seq=12)
    workspace.write("bob", "scratchpad", "b2", mode="append", tick=2, seq=13)
    workspace.write("alice", "scratchpad", "a2", mode="overwrite", tick=2, seq=14)
    assert workspace.log() == ()
    assert workspace.read("bob", "bob") == ("b1\nb2", 2)
    assert workspace.read("bob", "bob", version=2) == ("b1\nb2", 2)
    assert workspace.read("bob", "bob", "notes") == ("private", 1)
    assert workspace.read("bob", "bob", "notes", version=1) == ("private", 1)
    assert workspace.read("bob", "bob", version=0) == ("", 0)
    assert workspace.read("alice", "bob") == ("", 0)
    assert workspace.pending_delivery("alice") == (None, [])
    bob_view = workspace.view("bob")
    assert bob_view.versions == bob_view.contents == {}
    assert bob_view.own_staged == {"scratchpad": "b1\nb2", "notes": "private"}
    assert workspace.view("alice").own_staged == {"scratchpad": "a2"}
    assert before.own_staged == {}

    committed = workspace.commit_staged(["alice", "bob"])
    assert committed == [
        Write("alice", "scratchpad", 1, "a1", 2, 11),
        Write("alice", "scratchpad", 2, "a2", 2, 14),
        Write("bob", "scratchpad", 1, "b1", 2, 10),
        Write("bob", "notes", 1, "private", 2, 12),
        Write("bob", "scratchpad", 2, "b1\nb2", 2, 13),
    ]
    assert workspace.log() == tuple(committed)
    assert workspace.view("bob").own_staged == {}
    assert bob_view.own_staged == {"scratchpad": "b1\nb2", "notes": "private"}
    assert workspace.commit_staged(["alice", "bob"]) == []
    next_write = workspace.write("bob", "scratchpad", "b3", mode="append", tick=3, seq=15)
    assert next_write.version == 3
    assert next_write.content == "b1\nb2\nb3"
    assert workspace.read("bob", "bob") == ("b1\nb2\nb3", 3)
    assert workspace.read("bob", "bob", version=2) == ("b1\nb2", 2)
    assert workspace.read("alice", "bob") == ("b1\nb2", 2)
    assert workspace.commit_staged(["bob"])[0] == next_write


def test_unlisted_writes_remain_staged(workspace: Workspace) -> None:
    workspace.staged = True
    workspace.write("bob", "scratchpad", "b", mode="append", tick=0, seq=1)
    assert workspace.commit_staged(["alice"]) == []
    assert workspace.view("bob").own_staged == {"scratchpad": "b"}
    assert workspace.commit_staged(["bob"])[0].content == "b"


@pytest.mark.parametrize("seat_order", [[], ["worker"], ["worker", "bob", "alice"]])
def test_flush_commits_every_staged_write_in_deterministic_order(
    workspace: Workspace, seat_order: list[str]
) -> None:
    workspace.staged = True
    assert not workspace.has_staged()
    pending = {}
    for writer in ["worker", "bob", "alice"]:
        pending[writer] = [
            workspace.write(writer, "scratchpad", content, mode="append", tick=2, seq=seq)
            for seq, content in enumerate(["first", "second"])
        ]
    assert workspace.has_staged()
    expected_order = seat_order + sorted(pending.keys() - set(seat_order))
    expected = [write for writer in expected_order for write in pending[writer]]
    assert workspace.flush_staged(seat_order) == expected
    assert workspace.log() == tuple(expected)
    assert not workspace.has_staged()
    assert workspace.flush_staged(seat_order) == []


def test_flush_after_partial_commit_preserves_history(workspace: Workspace) -> None:
    workspace.staged = True
    alice = workspace.write("alice", "notes", "a", mode="append", tick=0, seq=1)
    bob = workspace.write("bob", "scratchpad", "b", mode="append", tick=0, seq=2)
    assert workspace.commit_staged(["alice"]) == [alice]
    assert workspace.has_staged()
    assert workspace.flush_staged(["alice"]) == [bob]
    assert workspace.log() == (alice, bob)
    assert not workspace.has_staged()


def test_immediate_workspace_has_nothing_to_flush(workspace: Workspace) -> None:
    write = workspace.write("alice", "notes", "a", mode="append", tick=None, seq=1)
    assert not workspace.has_staged()
    assert workspace.flush_staged([]) == []
    assert workspace.log() == (write,)


@pytest.mark.parametrize("method", ["commit_staged", "flush_staged"])
@pytest.mark.parametrize("staged", [False, True])
def test_unknown_seat_rejected_before_any_commit(
    workspace: Workspace, method: str, staged: bool
) -> None:
    workspace.staged = staged
    workspace.write("alice", "scratchpad", "a", mode="append", tick=0, seq=1)
    before = workspace.log()
    with pytest.raises(ValueError, match="unknown agent"):
        getattr(workspace, method)(["alice", "missing"])
    assert workspace.log() == before
    assert workspace.has_staged() is staged


@pytest.mark.parametrize(
    ("permissions", "readable"),
    [
        (Permissions(read_others=False), ["reader"]),
        (Permissions(readable_roles=("worker",)), ["reader", "worker"]),
        (Permissions(), ["reader", "peer", "worker"]),
    ],
)
def test_read_permissions_filter_reads_views_indexes_and_delivery(
    permissions: Permissions, readable: list[str]
) -> None:
    workspace = Workspace(
        roles={"reader": "reader", "peer": "peer", "worker": "worker"},
        permissions={
            "reader": permissions,
            "peer": Permissions(notes=True),
            "worker": Permissions(),
        },
        delivery=DeliverySpec(),
        staged=False,
    )
    for writer in ("reader", "peer", "worker"):
        workspace.write(writer, "scratchpad", writer, mode="overwrite", tick=0, seq=1)
    workspace.write("peer", "notes", "secret", mode="overwrite", tick=0, seq=2)
    view = workspace.view("reader")
    assert {writer for writer, _ in view.versions} == set(readable)
    assert {writer for writer, _, _ in view.contents} == set(readable)
    assert all(key == "scratchpad" for _, key in view.versions)
    assert [entry.writer for entry in workspace.list_index("reader")] == readable
    for writer in ("reader", "peer", "worker"):
        if writer in readable:
            assert workspace.read("reader", writer) == (writer, 1)
        else:
            with pytest.raises(ToolError, match="may not read"):
                workspace.read("reader", writer)
    text, reads = workspace.pending_delivery("reader")
    assert [read.writer for read in reads] == [writer for writer in readable if writer != "reader"]
    assert "secret" not in (text or "")


def test_notes_are_private_even_at_version_zero(workspace: Workspace) -> None:
    workspace.write("bob", "notes", "secret", mode="overwrite", tick=0, seq=1)
    assert workspace.read("bob", "bob", "notes") == ("secret", 1)
    assert workspace.view("bob").contents["bob", "notes", 1] == "secret"
    assert ("bob", "notes") not in workspace.view("alice").versions
    assert ("bob", "notes", 1) not in workspace.view("alice").contents
    for version in (None, 0, 1):
        with pytest.raises(ToolError, match="notes are private"):
            workspace.read("alice", "bob", "notes", version)


@pytest.mark.parametrize(
    ("key", "permissions"),
    [
        ("scratchpad", Permissions(write_scratchpad=False)),
        ("notes", Permissions(notes=False)),
    ],
)
def test_write_permissions(key: str, permissions: Permissions) -> None:
    workspace = Workspace(
        roles={"reader": "reader"},
        permissions={"reader": permissions},
        delivery=DeliverySpec(),
        staged=True,
    )
    with pytest.raises(ToolError, match=f"may not write {key}"):
        workspace.write("reader", key, "forbidden", mode="append", tick=0, seq=0)
    assert workspace.log() == ()
    assert workspace.view("reader").own_staged == {}


@pytest.mark.parametrize("key", ["", "Bad", "has space", "x:y", "x\ny", "é"])
def test_invalid_keys_are_tool_errors(workspace: Workspace, key: str) -> None:
    with pytest.raises(ToolError, match="invalid workspace key"):
        workspace.write("alice", key, "x", mode="append", tick=0, seq=0)
    with pytest.raises(ToolError, match="invalid workspace key"):
        workspace.read("alice", "bob", key)


@pytest.mark.parametrize("version", [-1, True, 1.5, 2])
def test_invalid_versions_are_tool_errors(workspace: Workspace, version: Any) -> None:
    workspace.write("bob", "scratchpad", "x", mode="append", tick=0, seq=0)
    with pytest.raises(ToolError, match="version"):
        workspace.read("alice", "bob", version=version)


@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    [
        ("read", ("missing", "alice"), {}),
        ("write", ("missing", "scratchpad", "x"), {"mode": "append", "tick": 0, "seq": 0}),
        ("view", ("missing",), {}),
        ("list_index", ("missing",), {}),
        ("pending_delivery", ("missing",), {}),
    ],
)
def test_unknown_agents(
    workspace: Workspace, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        getattr(workspace, method)(*args, **kwargs)


def test_unknown_model_read_target_is_tool_error(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="unknown agent"):
        workspace.read("alice", "missing")


def test_invalid_mode_preserves_state(workspace: Workspace) -> None:
    with pytest.raises(ToolError, match="write mode"):
        workspace.write("alice", "scratchpad", "x", mode="insert", tick=0, seq=0)
    assert workspace.log() == ()


def test_add_agent_and_snapshot_isolation(workspace: Workspace) -> None:
    workspace.add_agent("new", "worker")
    assert workspace.read("alice", "new") == ("", 0)
    workspace.write("new", "scratchpad", "new worker", mode="append", tick=1, seq=2)
    view = workspace.view("alice")
    assert view.versions == {("new", "scratchpad"): 1}
    assert view.contents == {("new", "scratchpad", 1): "new worker"}
    view.versions.clear()
    view.contents.clear()
    view.own_staged["scratchpad"] = "changed snapshot"
    assert workspace.read("alice", "new") == ("new worker", 1)
    assert workspace.view("alice").own_staged == {}
    with pytest.raises(ConfigError, match="already registered"):
        workspace.add_agent("new", "peer")
    with pytest.raises(ConfigError, match="permissions for role"):
        workspace.add_agent("other", "unknown")


def test_index_has_empty_scratchpads_and_capped_first_lines(workspace: Workspace) -> None:
    workspace.delivery = DeliverySpec(index_first_line_chars=3)
    workspace.write("bob", "scratchpad", "abcdef\nsecond", mode="overwrite", tick=0, seq=1)
    assert workspace.list_index("alice") == [
        IndexEntry("alice", "scratchpad", 0, 0, ""),
        IndexEntry("bob", "scratchpad", 1, 13, "abc"),
        IndexEntry("worker", "scratchpad", 0, 0, ""),
    ]


def test_notify_every_version_once_and_bookkeeping_is_per_reader(workspace: Workspace) -> None:
    workspace.delivery = DeliverySpec(index_first_line_chars=3)
    workspace.write("bob", "scratchpad", "initial", mode="overwrite", tick=0, seq=0)
    assert workspace.pending_delivery("alice")[1] == [
        WorkspaceRead("bob", "scratchpad", 1, ReadVia.NOTIFY)
    ]
    workspace.write("bob", "scratchpad", "abcdef\nsecond", mode="overwrite", tick=0, seq=1)
    workspace.write("bob", "scratchpad", "new", mode="overwrite", tick=0, seq=2)
    workspace.write("alice", "scratchpad", "mine", mode="overwrite", tick=0, seq=3)
    workspace.write("bob", "notes", "private", mode="overwrite", tick=0, seq=4)
    workspace.write("bob", "file.txt", "artifact", mode="overwrite", tick=0, seq=5)
    assert workspace.read("alice", "bob") == ("new", 3)
    assert workspace.pending_delivery("alice") == (
        "[workspace] bob wrote scratchpad v2 (13 chars): abc\n"
        "[workspace] bob wrote scratchpad v3 (3 chars): new",
        [
            WorkspaceRead("bob", "scratchpad", 2, ReadVia.NOTIFY),
            WorkspaceRead("bob", "scratchpad", 3, ReadVia.NOTIFY),
        ],
    )
    assert workspace.pending_delivery("alice") == (None, [])
    assert [read.writer for read in workspace.pending_delivery("worker")[1]] == [
        "bob",
        "alice",
    ]
    workspace.write("bob", "scratchpad", "", mode="overwrite", tick=1, seq=6)
    assert workspace.pending_delivery("alice") == (
        "[workspace] bob wrote scratchpad v4 (0 chars): ",
        [WorkspaceRead("bob", "scratchpad", 4, ReadVia.NOTIFY)],
    )


@pytest.mark.parametrize(
    ("mode", "view"), [("notify", "latest"), ("notify", "full"), ("push", "latest")]
)
@pytest.mark.parametrize("initial_empty_delivery", [False, True])
def test_late_joiners_first_delivery_has_only_latest_per_writer(
    workspace: Workspace, mode: str, view: str, initial_empty_delivery: bool
) -> None:
    workspace.delivery = DeliverySpec(mode=mode, view=view)
    if initial_empty_delivery:
        workspace.add_agent("late", "worker")
        assert workspace.pending_delivery("late") == (None, [])
    workspace.write("bob", "scratchpad", "old", mode="overwrite", tick=0, seq=1)
    workspace.write("alice", "scratchpad", "alice", mode="overwrite", tick=0, seq=2)
    workspace.write("bob", "scratchpad", "new", mode="overwrite", tick=1, seq=3)
    if not initial_empty_delivery:
        workspace.add_agent("late", "worker")
    text, reads = workspace.pending_delivery("late")
    assert text is not None and "old" not in text
    assert reads == [
        WorkspaceRead("alice", "scratchpad", 1, ReadVia(mode)),
        WorkspaceRead("bob", "scratchpad", 2, ReadVia(mode)),
    ]
    assert workspace.pending_delivery("late") == (None, [])
    workspace.write("bob", "scratchpad", "three", mode="overwrite", tick=2, seq=4)
    workspace.write("bob", "scratchpad", "four", mode="overwrite", tick=2, seq=5)
    expected_versions = [3, 4] if mode == "notify" else [4]
    assert [read.version for read in workspace.pending_delivery("late")[1]] == expected_versions


@pytest.mark.parametrize("view", ["latest", "full"])
def test_push_latest_vs_full_in_commit_order(workspace: Workspace, view: str) -> None:
    workspace.delivery = DeliverySpec(mode="push", view=view)
    workspace.write("bob", "scratchpad", "old", mode="overwrite", tick=0, seq=9)
    workspace.write("worker", "scratchpad", "worker", mode="overwrite", tick=0, seq=3)
    workspace.write("bob", "scratchpad", "new", mode="overwrite", tick=0, seq=4)
    workspace.write("alice", "scratchpad", "mine", mode="overwrite", tick=0, seq=5)
    workspace.write("bob", "notes", "private", mode="overwrite", tick=0, seq=6)
    workspace.write("bob", "file.txt", "artifact", mode="overwrite", tick=0, seq=7)
    expected_text = (
        "[workspace] worker scratchpad v1:\nworker\n\n[workspace] bob scratchpad v2:\nnew"
    )
    expected_reads = [
        WorkspaceRead("worker", "scratchpad", 1, ReadVia.PUSH),
        WorkspaceRead("bob", "scratchpad", 2, ReadVia.PUSH),
    ]
    if view == "full":
        expected_text = "[workspace] bob scratchpad v1:\nold\n\n" + expected_text
        expected_reads.insert(0, WorkspaceRead("bob", "scratchpad", 1, ReadVia.PUSH))
    assert workspace.pending_delivery("alice") == (expected_text, expected_reads)
    assert workspace.pending_delivery("alice") == (None, [])
    workspace.write("worker", "scratchpad", "next", mode="overwrite", tick=1, seq=8)
    assert workspace.pending_delivery("alice") == (
        "[workspace] worker scratchpad v2:\nnext",
        [WorkspaceRead("worker", "scratchpad", 2, ReadVia.PUSH)],
    )


def test_push_tail_cap_records_only_included_versions(workspace: Workspace) -> None:
    workspace.delivery = DeliverySpec(mode="push", view="full", push_max_chars=80)
    workspace.write("bob", "scratchpad", "old", mode="overwrite", tick=0, seq=1)
    content = "0123456789abcdef" * 10
    workspace.write("bob", "scratchpad", content, mode="overwrite", tick=0, seq=2)
    text, reads = workspace.pending_delivery("alice")
    header = "[workspace] bob scratchpad v2:\n"
    marker = "…[truncated]"
    assert text == header + marker + content[-(80 - len(header) - len(marker)) :]
    assert len(text) == 80
    assert reads == [WorkspaceRead("bob", "scratchpad", 2, ReadVia.PUSH)]
    assert workspace.pending_delivery("alice") == (None, [])


def test_push_drops_whole_oldest_blocks_without_truncating_headers(workspace: Workspace) -> None:
    workspace.delivery = DeliverySpec(mode="push", view="full", push_max_chars=70)
    for seq in range(1, 4):
        workspace.write("bob", "scratchpad", str(seq), mode="overwrite", tick=0, seq=seq)
    assert workspace.pending_delivery("alice") == (
        "[workspace] bob scratchpad v2:\n2\n\n[workspace] bob scratchpad v3:\n3",
        [WorkspaceRead("bob", "scratchpad", version, ReadVia.PUSH) for version in [2, 3]],
    )
    assert workspace.pending_delivery("alice") == (None, [])


@pytest.mark.parametrize("writer", ["b" * 40, "b" * 100])
def test_push_omits_block_when_header_and_truncation_marker_cannot_fit(
    workspace: Workspace, writer: str
) -> None:
    workspace.delivery = DeliverySpec(mode="push", push_max_chars=65)
    workspace.add_agent(writer, "peer")
    workspace.write(writer, "scratchpad", "x" * 100, mode="overwrite", tick=0, seq=1)
    assert workspace.pending_delivery("alice") == (None, [])
    assert workspace.pending_delivery("alice") == (None, [])


def test_pull_never_delivers(workspace: Workspace) -> None:
    workspace.delivery = DeliverySpec(mode="pull")
    workspace.write("bob", "scratchpad", "new", mode="overwrite", tick=0, seq=1)
    assert workspace.pending_delivery("alice") == (None, [])
    assert workspace.pending_delivery("alice") == (None, [])
    assert workspace.read("alice", "bob") == ("new", 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "invalid"},
        {"view": "invalid"},
        {"push_max_chars": -1},
        {"push_max_chars": 0},
        {"push_max_chars": 64},
        {"index_first_line_chars": -1},
    ],
)
def test_invalid_delivery_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        DeliverySpec(**kwargs)


def test_invalid_workspace_configuration() -> None:
    with pytest.raises(ConfigError, match="notes_cap_chars"):
        Workspace(
            roles={}, permissions={}, delivery=DeliverySpec(), staged=False, notes_cap_chars=-1
        )
    with pytest.raises(ConfigError, match="permissions for role"):
        Workspace(roles={"alice": "unknown"}, permissions={}, delivery=DeliverySpec(), staged=False)
