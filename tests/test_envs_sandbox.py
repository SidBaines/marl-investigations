"""Exercise kernel-enforced sandbox boundaries with small, bounded programs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import venv
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from marli.envs.sandbox import landlock
from marli.envs.sandbox.subprocess import ResourceLimits, SubprocessSandbox
from marli.errors import ConfigError


@pytest.fixture(scope="session")
def sandbox_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # The shared development venv may install another worktree. An isolated
    # stdlib-only venv gives -I -m a trusted import of this checkout, offline.
    root = tmp_path_factory.mktemp("sandbox-interpreter")
    root.parent.chmod(0o755)
    root.chmod(0o755)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(root)
    packages = root / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    (packages / "marli.pth").write_text(str(Path(__file__).resolve().parents[1] / "src") + "\n")
    return root / "bin/python"


@pytest.fixture
def sandbox_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandbox_python: Path) -> Path:
    if landlock.abi_version() < 4:
        pytest.skip("real sandbox requires Linux x86_64 with Landlock ABI >= 4")
    tmp_path.chmod(0o755)
    monkeypatch.setenv("MARLI_SANDBOX_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "executable", str(sandbox_python))
    return tmp_path


@pytest.fixture
async def sandbox(sandbox_host: Path) -> AsyncIterator[SubprocessSandbox]:
    instance = SubprocessSandbox()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.close()


def require_root() -> None:
    if os.geteuid() != 0:
        pytest.skip("dedicated sandbox uid requires root")


def uid_processes(uid: int) -> list[int]:
    result = []
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = status.read_text().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if line.startswith("Uid:") and int(line.split()[1]) == uid:
                result.append(int(status.parent.name))
    return result


async def test_echo_exit_codes_stdin_and_duration(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            "import sys; print(sys.stdin.read()); print('err', file=sys.stderr); sys.exit(7)",
        ],
        timeout_s=5,
        stdin="héllo",
    )
    assert result.exit_code == 7
    assert result.stdout == "héllo\n"
    assert result.stderr == "err\n"
    assert not result.timed_out
    assert 0 < result.duration_s < 5
    result = await sandbox.exec("printf shell", timeout_s=5)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "shell"


@pytest.mark.parametrize(
    "code",
    [
        "import time; time.sleep(100)",
        "import os, time\nwhile True:\n try:\n  if os.fork() == 0: time.sleep(100)\n"
        " except OSError: time.sleep(.01)\n",
    ],
)
async def test_timeout_kills_every_uid_process(sandbox_host: Path, code: str) -> None:
    require_root()
    instance = SubprocessSandbox(limits=ResourceLimits(processes=12))
    await instance.start()
    uid = instance.uid
    assert uid is not None
    try:
        started = time.monotonic()
        result = await instance.exec(["python3", "-c", code], timeout_s=0.4)
        assert result.timed_out
        assert result.exit_code is None
        assert time.monotonic() - started < 2.4
        assert uid_processes(uid) == []
        assert (await instance.exec("true", timeout_s=3)).exit_code == 0
    finally:
        await instance.close()


@pytest.mark.parametrize("dumpable", [True, False])
async def test_detached_descendant_is_cleaned_up(
    sandbox: SubprocessSandbox, dumpable: bool
) -> None:
    require_root()
    uid = sandbox.uid
    code = (
        "import ctypes, os, time\n"
        "if os.fork() == 0:\n os.setsid()\n"
        f" ctypes.CDLL(None).prctl(4, {int(dumpable)}, 0, 0, 0)\n"
        " os.close(1)\n os.close(2)\n time.sleep(100)\n"
    )
    result = await sandbox.exec(["python3", "-c", code], timeout_s=3)
    assert result.exit_code == 0, result.stderr
    assert uid is not None and uid_processes(uid) == []


async def test_filesystem_restrictions(sandbox: SubprocessSandbox, sandbox_host: Path) -> None:
    outside = sandbox_host / "outside"
    outside.mkdir(mode=0o777)
    outside.chmod(0o777)  # DAC alone must not account for the denied write.
    forbidden = outside / "escape"
    result = await sandbox.exec(
        ["python3", "-c", f"open({str(forbidden)!r}, 'w').write('bad')"],
        timeout_s=3,
    )
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr
    assert not forbidden.exists()
    result = await sandbox.exec('echo bad > "$HOME/../escape"', timeout_s=3)
    assert result.exit_code != 0
    assert not (sandbox_host / "escape").exists()
    result = await sandbox.exec(["cat", "/etc/hostname"], timeout_s=3)
    assert result.exit_code == 0
    assert result.stdout == Path("/etc/hostname").read_text()
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            "import os, tempfile; p=tempfile.NamedTemporaryFile(); print(os.path.dirname(p.name))",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == str(sandbox.workdir / "tmp")


@pytest.mark.parametrize("operation", ["connect(('127.0.0.1', 9))", "bind(('127.0.0.1', 0))"])
async def test_landlock_denies_tcp(sandbox: SubprocessSandbox, operation: str) -> None:
    result = await sandbox.exec(
        ["python3", "-c", f"import socket; socket.socket().{operation}"],
        timeout_s=3,
    )
    assert result.exit_code != 0
    assert "PermissionError: [Errno 13]" in result.stderr


async def test_memory_blowup_is_contained(sandbox_host: Path) -> None:
    instance = SubprocessSandbox(limits=ResourceLimits(address_space_bytes=128 * 1024**2))
    await instance.start()
    try:
        result = await instance.exec(["python3", "-c", "bytearray(512 * 1024**2)"], timeout_s=3)
        assert result.exit_code != 0
        assert "MemoryError" in result.stderr
        assert not result.timed_out
        assert (await instance.exec("true", timeout_s=3)).exit_code == 0
    finally:
        await instance.close()


async def test_environment_is_scrubbed(
    sandbox: SubprocessSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-secret")
    monkeypatch.setenv("HF_TOKEN", "test-secret")
    monkeypatch.setenv("PYTHONPATH", "/bad/pythonpath")
    result = await sandbox.exec(
        ["python3", "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout) == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(sandbox.workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(sandbox.workdir / "tmp"),
    }


async def test_identity_no_new_privs_and_limits(sandbox: SubprocessSandbox) -> None:
    require_root()
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            "import ctypes, json, os, resource; "
            "print(json.dumps([os.getuid(), os.getgid(), os.getgroups(), "
            "ctypes.CDLL(None).prctl(39, 0, 0, 0, 0), "
            "[resource.getrlimit(r) for r in [resource.RLIMIT_AS, resource.RLIMIT_CPU, "
            "resource.RLIMIT_FSIZE, resource.RLIMIT_NOFILE, resource.RLIMIT_NPROC]]]))",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr
    uid, gid, groups, no_new_privs, limits = json.loads(result.stdout)
    assert uid == gid == sandbox.uid
    assert groups == []
    assert no_new_privs == 1
    assert limits == [[4 * 1024**3] * 2, [8] * 2, [256 * 1024**2] * 2, [256] * 2, [256] * 2]


async def test_read_write_path_confinement(sandbox: SubprocessSandbox, sandbox_host: Path) -> None:
    await sandbox.write_file("dir/example.txt", "hello ☃")
    assert await sandbox.read_file("dir/example.txt") == "hello ☃"
    result = await sandbox.exec("cat dir/example.txt", timeout_s=3)
    assert result.stdout == "hello ☃"
    for path in ("/etc/hostname", "../escape", "dir/../../escape"):
        with pytest.raises(ValueError):
            await sandbox.read_file(path)
        with pytest.raises(ValueError):
            await sandbox.write_file(path, "bad")
    target = sandbox_host / "untouched"
    target.write_text("safe")
    (sandbox.workdir / "link").symlink_to(target)
    (sandbox.workdir / "linked-dir").symlink_to(sandbox_host, target_is_directory=True)
    for path in ("link", "linked-dir/untouched"):
        with pytest.raises(ValueError, match="symlink"):
            await sandbox.read_file(path)
        with pytest.raises(ValueError, match="symlink"):
            await sandbox.write_file(path, "bad")
    assert target.read_text() == "safe"


async def test_reset_close_and_uid_reuse(sandbox_host: Path) -> None:
    require_root()
    first = SubprocessSandbox()
    await first.start()
    await first.start()
    uid, workdir = first.uid, first.workdir
    await first.write_file("nested/file", "old")
    await first.reset()
    await first.reset()
    assert list(workdir.iterdir()) == [workdir / "tmp"]
    await first.close()
    await first.close()
    assert not workdir.exists()
    second = SubprocessSandbox()
    await second.start()
    try:
        assert second.uid == uid
    finally:
        await second.close()
    unused = SubprocessSandbox()
    await unused.close()
    await unused.close()


async def test_close_interrupts_active_exec(sandbox: SubprocessSandbox) -> None:
    require_root()
    uid = sandbox.uid
    running = asyncio.create_task(sandbox.exec("sleep 100", timeout_s=100))
    for _ in range(100):
        if uid is not None and uid_processes(uid):
            break
        await asyncio.sleep(0.01)
    await asyncio.wait_for(sandbox.close(), 3)
    result = await asyncio.wait_for(running, 3)
    assert result.exit_code != 0
    assert not sandbox.workdir.exists()
    assert uid is not None and uid_processes(uid) == []


async def test_cancel_exec_cleans_up(sandbox: SubprocessSandbox) -> None:
    require_root()
    uid = sandbox.uid
    running = asyncio.create_task(sandbox.exec("sleep 100", timeout_s=100))
    await asyncio.sleep(0.1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert uid is not None and uid_processes(uid) == []


async def test_cancel_during_launch_cleans_up(
    sandbox: SubprocessSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_root()
    uid = sandbox.uid
    created = asyncio.Event()
    create_process = asyncio.create_subprocess_exec

    async def delayed_launch(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await create_process(*args, **kwargs)
        created.set()
        await asyncio.sleep(0.1)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_launch)
    running = asyncio.create_task(sandbox.exec("sleep 100", timeout_s=100))
    await asyncio.wait_for(created.wait(), 3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, 3)
    assert uid is not None and uid_processes(uid) == []


async def test_launcher_cannot_be_shadowed(sandbox: SubprocessSandbox) -> None:
    await sandbox.write_file("sitecustomize.py", "raise RuntimeError('unsafe startup')")
    await sandbox.write_file("marli/__init__.py", "raise RuntimeError('unsafe import')")
    result = await sandbox.exec(["/bin/echo", "safe startup"], timeout_s=3)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "safe startup\n"


@pytest.mark.parametrize("abi", [0, 3])
async def test_missing_landlock_fails_before_creating_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abi: int
) -> None:
    monkeypatch.setattr(landlock, "abi_version", lambda: abi)
    monkeypatch.setenv("MARLI_SANDBOX_ROOT", str(tmp_path))
    instance = SubprocessSandbox()
    with pytest.raises(ConfigError, match="Landlock ABI >= 4"):
        await instance.start()
    assert list(tmp_path.iterdir()) == []
    await instance.close()


async def test_degraded_mode_warns(sandbox_host: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(landlock, "abi_version", lambda: 0)
    instance = SubprocessSandbox(require_landlock=False)
    with pytest.warns(RuntimeWarning, match="degraded mode"):
        await instance.start()
    await instance.close()


async def test_uid_pool_exhaustion_and_recovery(sandbox_host: Path) -> None:
    require_root()
    first = SubprocessSandbox(uid_range=(61999, 61999))
    second = SubprocessSandbox(uid_range=(61999, 61999))
    await first.start()
    try:
        with pytest.raises(ConfigError, match="uid pool exhausted"):
            await second.start()
    finally:
        await first.close()
    await second.start()
    await second.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_invalid_timeout(sandbox: SubprocessSandbox, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        await sandbox.exec("true", timeout_s=timeout)
