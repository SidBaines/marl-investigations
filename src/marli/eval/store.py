"""Retries supersede earlier attempts without losing completed episode identities.

Token rows precede their episode commit. Matching occurrences discards an
orphan token append, including one from a retry of an already recorded id.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from marli.errors import MarliError
from marli.handles import atomic_write_text
from marli.interact import records
from marli.interact.types import Episode
from marli.rundir import RunDir


def episode_id(task_id: str, config_hash: str, episode_idx: int) -> str:
    """Match run_episode's default group and repeat identity."""
    return f"{task_id}/{config_hash[:8]}/e{episode_idx}"


def read_episodes(root: Path) -> Iterator[Episode]:
    """Keep the last committed attempt for each episode, including incomplete runs."""
    latest = {
        episode.episode_id: episode for episode, _ in records.read_episodes(root, with_tokens=False)
    }
    yield from latest.values()


def compact(run: RunDir, *, record_tokens: bool) -> list[dict[str, Any]]:
    """Replace both stores atomically per file, recovering an interrupted rewrite.

    A durable journal bridges the two renames: after a crash neither sidecar
    generations nor their episode occurrences may be mixed during recovery.
    """
    journal = run.path(".episode-compaction.json")
    if journal.exists():
        snapshots = json.loads(journal.read_text(encoding="utf-8"))
        _replace_stores(run, snapshots)
    latest: dict[str, dict[str, Any]] = {}
    tokens: dict[str, dict[str, Any]] = {}
    pending: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    if record_tokens:
        for row in run.read_rows("tokens.jsonl"):
            pending[row["episode_id"]].append(row)
    for row in run.read_rows("episodes.jsonl"):
        key = row["episode_id"]
        latest[key] = row
        if record_tokens:
            if not pending[key]:
                raise MarliError(f"Missing tokens for episode {key}")
            tokens[key] = pending[key].popleft()
    snapshots = {"episodes.jsonl": list(latest.values())}
    if record_tokens:
        snapshots["tokens.jsonl"] = list(tokens.values())
    atomic_write_text(journal, json.dumps(snapshots, sort_keys=True) + "\n")
    _replace_stores(run, snapshots)
    return list(latest.values())


def _replace_stores(run: RunDir, snapshots: dict[str, list[dict[str, Any]]]) -> None:
    for filename, rows in snapshots.items():
        atomic_write_text(
            run.path(filename), "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
    run.path(".episode-compaction.json").unlink()
    directory_fd = os.open(run.out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
