"""Read-only git/host provenance adapted from scimt for embedding in run manifests.

Training requires a reproducible commit unless explicitly opted out; collecting
provenance alone also works outside a checkout or on a host without git.
"""

from __future__ import annotations

import os
import socket
import subprocess
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from marli import __version__
from marli.errors import DirtyTreeError


@dataclass(frozen=True)
class GitInfo:
    """Commit/dirty are None outside a git checkout, without commits, or without git."""

    commit: str | None
    dirty: bool | None


def git_info(repo_dir: str | Path | None = None) -> GitInfo:
    """Read the current commit and worktree status, returning unknown on failure."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_dir
        )
        status = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return GitInfo(commit=None, dirty=None)
    if commit.returncode != 0 or status.returncode != 0:
        return GitInfo(commit=None, dirty=None)
    return GitInfo(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()))


def provenance(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Collect the small provenance record stored by run directories."""
    git = git_info(repo_dir)
    return {
        "git_commit": git.commit,
        "git_dirty": git.dirty,
        "host": socket.gethostname(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "marli_version": __version__,
    }


def require_clean_tree(repo_dir: str | Path | None = None) -> GitInfo:
    """Refuse training without clean git provenance unless explicitly allowed."""
    git = git_info(repo_dir)
    if git.commit is None or git.dirty:
        reason = "git commit is unavailable" if git.commit is None else "git tree is dirty"
        if os.environ.get("MARLI_ALLOW_DIRTY") != "1":
            raise DirtyTreeError(
                f"{reason}: a checkpoint you can't map to a commit is a result "
                "you can't reproduce. "
                "Commit your changes or set MARLI_ALLOW_DIRTY=1 to opt out."
            )
        warnings.warn(f"{reason}: proceeding because MARLI_ALLOW_DIRTY=1", stacklevel=2)
    return git
