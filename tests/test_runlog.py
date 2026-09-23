"""Provenance uses local git only and never changes machine-wide git settings."""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from marli import __version__
from marli.errors import DirtyTreeError
from marli.runlog import GitInfo, git_info, provenance, require_clean_tree


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("MARLI_ALLOW_DIRTY", raising=False)
    git(tmp_path, "init")
    (tmp_path / "tracked.txt").write_text("initial\n")
    git(tmp_path, "add", "tracked.txt")
    git(
        tmp_path,
        "-c",
        "user.name=Marli Test",
        "-c",
        "user.email=marli-test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Initial test commit",
    )
    return tmp_path


def test_git_info_clean_and_frozen(repo: Path) -> None:
    info = git_info(repo)
    assert info == GitInfo(commit=git(repo, "rev-parse", "HEAD"), dirty=False)
    assert require_clean_tree(str(repo)) == info
    with pytest.raises(FrozenInstanceError):
        info.dirty = True


@pytest.mark.parametrize("filename", ["tracked.txt", "untracked.txt"])
def test_git_info_dirty_and_guard(repo: Path, filename: str) -> None:
    (repo / filename).write_text("changed\n")
    assert git_info(repo) == GitInfo(commit=git(repo, "rev-parse", "HEAD"), dirty=True)
    with pytest.raises(DirtyTreeError, match="dirty.*can't reproduce.*MARLI_ALLOW_DIRTY=1"):
        require_clean_tree(repo)


def test_dirty_guard_explicit_opt_out(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (repo / "untracked.txt").write_text("new\n")
    monkeypatch.setenv("MARLI_ALLOW_DIRTY", "true")
    with pytest.raises(DirtyTreeError):
        require_clean_tree(repo)
    monkeypatch.setenv("MARLI_ALLOW_DIRTY", "1")
    with pytest.warns(UserWarning, match="git tree is dirty.*MARLI_ALLOW_DIRTY=1"):
        info = require_clean_tree(repo)
    assert info == git_info(repo)
    assert info.dirty is True


def test_clean_tree_opt_out_does_not_warn(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARLI_ALLOW_DIRTY", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert require_clean_tree(repo) == git_info(repo)


def test_default_repo_is_current_directory(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    assert git_info() == git_info(repo)
    assert require_clean_tree() == git_info(repo)


def test_non_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # tmp_path may sit inside a git checkout (e.g. --basetemp under the repo);
    # stop discovery at its parent.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.delenv("MARLI_ALLOW_DIRTY", raising=False)
    assert git_info(tmp_path) == GitInfo(commit=None, dirty=None)
    with pytest.raises(DirtyTreeError, match="commit is unavailable.*can't reproduce"):
        require_clean_tree(tmp_path)
    monkeypatch.setenv("MARLI_ALLOW_DIRTY", "1")
    with pytest.warns(UserWarning, match="git commit is unavailable.*MARLI_ALLOW_DIRTY=1"):
        assert require_clean_tree(tmp_path) == GitInfo(commit=None, dirty=None)


def test_git_info_disables_optional_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    def fake_git(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout="abc123\n" if "rev-parse" in argv else "", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_git)
    assert git_info() == GitInfo(commit="abc123", dirty=False)
    assert commands == [
        ["git", "rev-parse", "HEAD"],
        ["git", "--no-optional-locks", "status", "--porcelain"],
    ]


def test_git_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git is missing")

    monkeypatch.setattr(subprocess, "run", missing_git)
    assert git_info() == GitInfo(commit=None, dirty=None)


def test_missing_directory(tmp_path: Path) -> None:
    assert git_info(tmp_path / "absent") == GitInfo(commit=None, dirty=None)


def test_invalid_directory() -> None:
    assert git_info("invalid\0directory") == GitInfo(commit=None, dirty=None)


@pytest.mark.parametrize("failed_command", ["rev-parse", "status"])
def test_failed_git_command_refuses_training(
    monkeypatch: pytest.MonkeyPatch, failed_command: str
) -> None:
    def failed_git(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failed_command in argv:
            return subprocess.CompletedProcess(argv, 128, stdout="", stderr="git failed")
        return subprocess.CompletedProcess(
            argv, 0, stdout="abc123\n" if "rev-parse" in argv else "", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", failed_git)
    monkeypatch.delenv("MARLI_ALLOW_DIRTY", raising=False)
    assert git_info() == GitInfo(commit=None, dirty=None)
    with pytest.raises(DirtyTreeError, match="commit is unavailable"):
        require_clean_tree()


def test_provenance_keys_and_pod_id(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    before = datetime.now(UTC).replace(microsecond=0)
    record = provenance(repo)
    after = datetime.now(UTC).replace(microsecond=0)
    assert set(record) == {
        "git_commit",
        "git_dirty",
        "host",
        "started_at",
        "pod_id",
        "marli_version",
    }
    assert record["git_commit"] == git(repo, "rev-parse", "HEAD")
    assert record["git_dirty"] is False
    assert isinstance(record["host"], str) and record["host"]
    started_at = datetime.fromisoformat(record["started_at"])
    assert started_at.tzinfo == UTC
    assert started_at.microsecond == 0
    assert before <= started_at <= after
    assert record["pod_id"] is None
    assert record["marli_version"] == __version__
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    assert provenance(repo)["pod_id"] == "pod-123"
