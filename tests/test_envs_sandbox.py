"""Exercise kernel-enforced sandbox boundaries with small, bounded programs."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from marli.envs.sandbox import landlock
from marli.envs.sandbox import subprocess as sandbox_module
from marli.envs.sandbox.subprocess import ResourceLimits, SubprocessSandbox
from marli.errors import ConfigError

_TEST_UID_BASE = 1_000_000 + os.getpid() * 16


@pytest.fixture
def sandbox_host(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    if landlock.abi_version() < 4:
        pytest.skip("real sandbox requires Linux x86_64 with Landlock ABI >= 4")
    require_root()
    # Independent of pytest's private 0700 basetemp. TMPDIR permits a builder
    # with workspace-only write permission to keep test artifacts in its checkout.
    root = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp")))
    root.chmod(0o711)
    # Test lock directories are private, so their uid pools must also be
    # disjoint from production and from concurrent pytest processes/worktrees.
    monkeypatch.setattr(
        SubprocessSandbox.__init__,
        "__kwdefaults__",
        {
            **SubprocessSandbox.__init__.__kwdefaults__,
            "uid_range": (_TEST_UID_BASE, _TEST_UID_BASE + 15),
        },
    )
    monkeypatch.setenv("MARLI_SANDBOX_ROOT", str(root))
    monkeypatch.setattr(sandbox_module, "_LOCK_ROOT", root / "locks")
    monkeypatch.setattr(sandbox_module, "_FALLBACK_LOCK_ROOT", root / "fallback-locks")
    try:
        yield root
    finally:
        shutil.rmtree(root)


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
    return sorted(sandbox_module._uid_pids(uid))


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
    assert limits == [[4 * 1024**3] * 2, [8] * 2, [256 * 1024**2] * 2, [256] * 2, [64] * 2]


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
    assert {p.name for p in workdir.iterdir()} == {"tmp", sandbox_module._MARKER}
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
        if uid is not None and await asyncio.to_thread(uid_processes, uid):
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
    uid = _TEST_UID_BASE + 15
    first = SubprocessSandbox(uid_range=(uid, uid))
    second = SubprocessSandbox(uid_range=(uid, uid))
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


@pytest.mark.parametrize("path", ["/etc/rp_environment", "/etc/environment", "/etc/shadow"])
async def test_nonallowlisted_etc_files_are_unreadable(
    sandbox: SubprocessSandbox, path: str
) -> None:
    if not Path(path).exists():
        pytest.skip(f"no existing probe at {path}")
    # Never read or print secrets, even if the protection regresses.
    result = await sandbox.exec(
        ["python3", "-c", f"import os; fd = os.open({path!r}, os.O_RDONLY); os.close(fd)"],
        timeout_s=3,
    )
    assert result.exit_code != 0
    assert "PermissionError: [Errno 13]" in result.stderr
    assert result.stdout == ""


async def test_etc_allowlist_and_devices(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            """
import os, subprocess
for path in ('/etc/hostname', '/etc/passwd', '/etc/group', '/etc/ld.so.cache'):
    if os.path.exists(path):
        with open(path, 'rb') as f: f.read(1)
with open('/dev/zero', 'r+b', buffering=0) as f:
    assert f.read(3) == b'\0' * 3
    assert f.write(b'x') == 1
with open('/dev/urandom', 'rb') as f: assert len(f.read(1)) == 1
subprocess.run(['true'], stdout=subprocess.DEVNULL, check=True)
print('ok')
""".replace("\0", "\\0"),
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "ok\n"
    result = await sandbox.exec("echo ignored > /dev/null; printf ok", timeout_s=3)
    assert result.exit_code == 0 and result.stdout == "ok"
    result = await sandbox.exec(["python3", "-c", "import os; os.listdir('/dev/shm')"], timeout_s=3)
    assert result.exit_code != 0 and "PermissionError" in result.stderr


async def test_host_reads_and_cross_episode_access_are_denied(
    sandbox: SubprocessSandbox, sandbox_host: Path
) -> None:
    other = SubprocessSandbox()
    await other.start()
    try:
        await other.write_file("private", "other episode")
        paths = [
            "/root",
            "/workspace",
            "/proc/1/environ",
            "/proc/1/cmdline",
            "/proc/cpuinfo",
            "/sys/devices/system/cpu/online",
            "/var/log",
            "/run",
            "/tmp",
            "/home",
            "/opt",
            str(other.workdir / "private"),
            str(Path(__file__).resolve().parents[1] / "src/marli/__init__.py"),
        ]
        paths = [p for p in paths if Path(p).exists()]
        result = await sandbox.exec(
            [
                "python3",
                "-c",
                f"""
import errno, os
for path in {paths!r}:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        assert e.errno == errno.EACCES, (path, e.errno)
    else:
        os.close(fd)
        raise AssertionError('host path was accessible: ' + path)
""",
            ],
            timeout_s=3,
        )
        assert result.exit_code == 0, result.stderr
        result = await sandbox.exec("ln /etc/passwd hardlink", timeout_s=3)
        assert result.exit_code != 0
    finally:
        await other.close()


@pytest.mark.parametrize(
    "arguments",
    [
        "socket.AF_INET, socket.SOCK_STREAM, 0",  # TCP
        "socket.AF_INET6, socket.SOCK_STREAM, 0",
        "socket.AF_INET, socket.SOCK_STREAM, 262",  # MPTCP
        "socket.AF_INET, socket.SOCK_STREAM, 132",  # SCTP
        "socket.AF_INET, socket.SOCK_DGRAM, 0",  # UDP, including DNS
        "socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP",
        "socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP",
        "socket.AF_NETLINK, socket.SOCK_RAW, 0",
        "socket.AF_PACKET, socket.SOCK_RAW, 0",
        "socket.AF_UNIX, socket.SOCK_STREAM, 0",  # abstract and filesystem sockets
        "40, socket.SOCK_STREAM, 0",  # vsock
    ],
)
async def test_seccomp_denies_all_socket_domains(
    sandbox: SubprocessSandbox, arguments: str
) -> None:
    result = await sandbox.exec(
        ["python3", "-c", f"import socket; socket.socket({arguments})"],
        timeout_s=3,
    )
    assert result.exit_code != 0
    assert "PermissionError: [Errno 13]" in result.stderr


@pytest.mark.skipif(
    os.environ.get("MARLI_TEST_NO_NETWORK") == "1", reason="builder forbids host network use"
)
@pytest.mark.parametrize("protocol", [0, 262])
async def test_tcp_and_mptcp_never_reach_host_listener(
    sandbox: SubprocessSandbox, protocol: int
) -> None:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = await sandbox.exec(
            [
                "python3",
                "-c",
                f"import socket; s=socket.socket(socket.AF_INET, "
                f"socket.SOCK_STREAM, {protocol}); s.connect(('127.0.0.1', {port}))",
            ],
            timeout_s=3,
        )
        assert result.exit_code != 0 and "PermissionError: [Errno 13]" in result.stderr
        listener.setblocking(False)
        with pytest.raises(BlockingIOError):
            listener.accept()


async def test_unix_socket_creation_denied_for_both_address_types(
    sandbox: SubprocessSandbox,
) -> None:
    for address in ("\0marli-abstract-probe", str(sandbox.workdir / "sock")):
        result = await sandbox.exec(
            [
                "python3",
                "-c",
                f"import socket; s = socket.socket(socket.AF_UNIX); s.bind({address!r})",
            ],
            timeout_s=3,
        )
        assert result.exit_code != 0 and "PermissionError: [Errno 13]" in result.stderr
    assert not (sandbox.workdir / "sock").exists()


async def test_socketpair_and_asyncio_work(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            """
import asyncio, errno, socket
left, right = socket.socketpair()
with left, right:
    assert left.family == socket.AF_UNIX
    left.sendall(b'pair works')
    assert right.recv(10) == b'pair works'
try:
    socket.socketpair(socket.AF_INET)
except OSError as e:
    assert e.errno == errno.EACCES
else:
    raise AssertionError('non-Unix socketpair allowed')
async def main():
    await asyncio.sleep(0)
    print('asyncio works')
asyncio.run(main())
""",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == "asyncio works\n"


async def test_seccomp_denies_privileged_syscalls(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            """
import ctypes, errno
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
numbers = (425, 426, 427, 101, 165, 166, 272, 308, 321, 298,
           250, 248, 249, 310, 311, 246, 175, 313)
for number in numbers:
    ctypes.set_errno(0)
    result = libc.syscall(number, 0, 0, 0, 0, 0, 0)
    assert result == -1 and ctypes.get_errno() == errno.EPERM, number
""",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr


@pytest.mark.parametrize("abi", ["x32", "i386"])
async def test_seccomp_kills_alternate_syscall_abis(sandbox: SubprocessSandbox, abi: str) -> None:
    import signal

    code = "import ctypes; ctypes.CDLL(None).syscall(0x40000000)"
    if abi == "i386":
        code = (
            "import ctypes, mmap; "
            "m=mmap.mmap(-1, 4096, prot=mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC); "
            "m.write(bytes.fromhex('b814000000cd80c3')); "
            "ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.addressof(ctypes.c_char.from_buffer(m)))()"
        )
    result = await sandbox.exec(["python3", "-c", code], timeout_s=3)
    assert result.exit_code == -signal.SIGSYS


async def test_seccomp_startup_is_fail_closed(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unavailable(*args: str) -> tuple[int, str]:
        return 1, "seccomp unavailable"

    monkeypatch.setattr(sandbox_module, "_helper", unavailable)
    instance = SubprocessSandbox()
    with pytest.raises(ConfigError, match="cannot install.*seccomp"):
        await instance.start()
    assert not hasattr(instance, "workdir")
    degraded = SubprocessSandbox(require_seccomp=False)
    with pytest.warns(RuntimeWarning, match="seccomp.*degraded"):
        await degraded.start()
    assert not degraded._seccomp
    await degraded.close()


async def test_output_cap_keeps_both_ends_of_each_stream(sandbox_host: Path) -> None:
    instance = SubprocessSandbox(max_output_bytes=1024)
    await instance.start()
    try:
        result = await instance.exec(
            [
                "python3",
                "-c",
                """
import os
for fd in (1, 2):
    os.write(fd, b'H' * 512 + b'M' * 2048 + b'T' * 512)
""",
            ],
            timeout_s=3,
        )
        assert result.exit_code == 0, result.stderr
        assert result.truncated
        assert result.stdout == result.stderr == "H" * 512 + "T" * 512
    finally:
        await instance.close()


async def test_yes_output_does_not_grow_parent_rss(sandbox: SubprocessSandbox) -> None:
    def rss() -> int:
        lines = Path("/proc/self/status").read_text().splitlines()
        return int(next(line for line in lines if line.startswith("VmRSS:")).split()[1]) * 1024

    before = rss()
    peaks = [before]
    running = asyncio.create_task(sandbox.exec("yes output-flood", timeout_s=2))
    while not running.done():
        peaks.append(rss())
        await asyncio.sleep(0.01)
    result = await running
    assert result.truncated
    assert len(result.stdout.encode()) <= sandbox.max_output_bytes
    assert len(result.stderr.encode()) <= sandbox.max_output_bytes
    assert max(*peaks, rss()) - before < 50 * 1024**2


async def test_oom_preference_and_core_limit(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            """
import resource
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
assert open('/proc/self/oom_score_adj').read().strip() == '1000'
""",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr


async def test_aggregate_memory_stops_six_allocating_children(sandbox_host: Path) -> None:
    # Each child attempts 1 GiB, growing gradually. A scaled aggregate budget
    # verifies the same boundary without committing multiple GiB on a shared pod.
    instance = SubprocessSandbox(max_memory_bytes=192 * 1024**2)
    await instance.start()
    uid = instance.uid
    try:
        result = await instance.exec(
            [
                "python3",
                "-c",
                """
import os, time
for _ in range(6):
    if os.fork() == 0:
        blocks = []
        for _ in range(128):
            blocks.append(bytearray(8 * 1024**2))
            time.sleep(.08)
        os._exit(0)
time.sleep(30)
""",
            ],
            timeout_s=10,
        )
        assert result.exit_code != 0 and not result.timed_out
        assert "max_memory_bytes exceeded" in result.stderr
        assert uid is not None and not uid_processes(uid)
        assert (await instance.exec("true", timeout_s=3)).exit_code == 0
    finally:
        await instance.close()


async def test_disk_watch_stops_many_files_below_fsize(sandbox_host: Path) -> None:
    # Eight 200 MiB files total 1.6 GiB, BELOW the 2 GiB default; use a lower
    # aggregate quota so this regression actually crosses its configured bound.
    instance = SubprocessSandbox(max_disk_bytes=512 * 1024**2)
    await instance.start()
    try:
        result = await instance.exec(
            [
                "python3",
                "-c",
                """
import time
chunk = b'x' * 1024**2
for i in range(8):
    with open(f'file{i}', 'wb') as f:
        for _ in range(200):
            f.write(chunk)
            time.sleep(.001)
time.sleep(30)
""",
            ],
            timeout_s=10,
        )
        assert result.exit_code != 0 and not result.timed_out
        assert "max_disk_bytes exceeded" in result.stderr
        await instance.reset()
        assert (await instance.exec("true", timeout_s=3)).exit_code == 0
    finally:
        await instance.close()


async def test_disk_checked_after_fast_exec_and_before_next_exec(sandbox_host: Path) -> None:
    instance = SubprocessSandbox(max_disk_bytes=1024**2)
    await instance.start()
    try:
        result = await instance.exec(
            ["python3", "-c", "open('fast', 'wb').truncate(2 * 1024**2)"],
            timeout_s=3,
        )
        assert result.exit_code != 0 and "max_disk_bytes exceeded" in result.stderr
        result = await instance.exec("echo should-not-run", timeout_s=3)
        assert result.exit_code != 0 and result.stdout == ""
    finally:
        await instance.close()


async def test_default_root_and_launcher_path(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = sandbox_host / "default"
    monkeypatch.delenv("MARLI_SANDBOX_ROOT")
    monkeypatch.setattr(sandbox_module, "_DEFAULT_ROOT", root)
    instance = SubprocessSandbox()
    await instance.start()
    try:
        assert instance.workdir.parent == root
        assert root.stat().st_mode & 0o777 == 0o711
        assert root.stat().st_uid == 0
        code, detail = await sandbox_module._helper("--probe-seccomp")
        assert code == 0, detail
        assert json.loads(detail)["launcher"] == str(
            Path(__file__).resolve().parents[1] / "src/marli/envs/sandbox/_launch.py"
        )
    finally:
        await instance.close()


async def test_private_ancestor_rejected_with_name(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = sandbox_host / "private"
    private.mkdir(mode=0o700)
    root = private / "searchable"
    root.mkdir(mode=0o711)
    monkeypatch.setenv("MARLI_SANDBOX_ROOT", str(root))
    instance = SubprocessSandbox()
    with pytest.raises(ConfigError, match=f"not searchable.*{private}"):
        await instance.start()
    assert list(root.iterdir()) == []


async def test_nonroot_requires_explicit_degraded_mode(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    with pytest.raises(ConfigError, match="dedicated uid"):
        await SubprocessSandbox().start()
    instance = SubprocessSandbox(require_dedicated_uid=False)
    with pytest.warns(RuntimeWarning, match="dedicated uid.*degraded"):
        await instance.start()
    try:
        assert instance.uid is None
    finally:
        await instance.close()


async def test_bash_has_no_dotfile_state(sandbox: SubprocessSandbox) -> None:
    await sandbox.write_file(".bash_profile", "echo profile-state\n")
    await sandbox.write_file(".bashrc", "echo rc-state\n")
    result = await sandbox.exec("echo clean", timeout_s=3)
    assert result.exit_code == 0 and result.stdout == "clean\n" and result.stderr == ""


async def test_parent_subreaper_state_is_unchanged(sandbox_host: Path) -> None:
    import ctypes

    state = ctypes.c_int()
    libc = ctypes.CDLL(None)
    assert libc.prctl(37, ctypes.byref(state), 0, 0, 0) == 0
    before = state.value
    instance = SubprocessSandbox()
    await instance.start()
    await instance.close()
    assert libc.prctl(37, ctypes.byref(state), 0, 0, 0) == 0
    assert state.value == before


def _uid_lock_worker(root: str, uid: int, connection: Any) -> None:
    os.environ["MARLI_SANDBOX_ROOT"] = root
    sandbox_module._LOCK_ROOT = Path(root) / "locks"

    async def hold() -> None:
        instance = SubprocessSandbox(uid_range=(uid, uid))
        try:
            await instance.start()
        except ConfigError as exc:
            connection.send(("error", str(exc)))
            return
        try:
            connection.send(("started", instance.uid))
            connection.recv()
        finally:
            await instance.close()

    asyncio.run(hold())
    connection.close()


async def test_uid_lock_excludes_another_spawned_process(sandbox_host: Path) -> None:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    parent_a, child_a = context.Pipe()
    parent_b, child_b = context.Pipe()
    uid = _TEST_UID_BASE + 14
    first = context.Process(target=_uid_lock_worker, args=(str(sandbox_host), uid, child_a))
    second = context.Process(target=_uid_lock_worker, args=(str(sandbox_host), uid, child_b))
    first.start()
    try:
        assert await asyncio.to_thread(parent_a.poll, 10)
        assert parent_a.recv() == ("started", uid)
        second.start()
        assert await asyncio.to_thread(parent_b.poll, 10)
        status, message = parent_b.recv()
        assert status == "error" and "uid pool exhausted" in message
        await asyncio.to_thread(second.join, 10)
        assert second.exitcode == 0
    finally:
        parent_a.send("close")
        await asyncio.to_thread(first.join, 10)
        for process in (first, second):
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join()
        for connection in (parent_a, child_a, parent_b, child_b):
            connection.close()
    assert first.exitcode == 0
    instance = SubprocessSandbox(uid_range=(uid, uid))
    await instance.start()
    await instance.close()


async def test_stale_sweep_preserves_live_and_unmarked_directories(sandbox_host: Path) -> None:
    stale = SubprocessSandbox()
    live = SubprocessSandbox()
    await stale.start()
    await live.start()
    stale_path = stale.workdir
    stale._release_uid()  # simulate an exited trainer, with no remaining child processes
    stale._started = False
    unmarked = sandbox_host / "unmarked"
    unmarked.mkdir()
    (unmarked / "keep").write_text("valuable")
    new = SubprocessSandbox()
    try:
        await new.start()
        assert not stale_path.exists()
        assert live.workdir.exists()
        assert (unmarked / "keep").read_text() == "valuable"
    finally:
        await live.close()
        await new.close()


async def test_stale_sweep_tolerates_directory_removed_concurrently(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = SubprocessSandbox()
    await stale.start()
    stale_path = stale.workdir
    stale._release_uid()
    stale._started = False
    original = sandbox_module._uid_pids

    def closed_meanwhile(uid: int) -> list[int]:
        # Another process's close() finished its rmtree after we read the marker.
        if stale_path.exists():
            shutil.rmtree(stale_path)
        return original(uid)

    monkeypatch.setattr(sandbox_module, "_uid_pids", closed_meanwhile)
    new = SubprocessSandbox()
    try:
        await new.start()
        assert not stale_path.exists() and new.workdir.exists()
    finally:
        monkeypatch.setattr(sandbox_module, "_uid_pids", original)
        await new.close()


async def test_concurrent_commands_do_not_starve_the_default_executor(
    sandbox_host: Path,
) -> None:
    # Each command's resource watchdog used to hold an executor thread until the
    # command's cleanup finished, while that cleanup awaited to_thread: with more
    # concurrent commands than executor threads, every episode froze.
    import concurrent.futures

    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=2)
    )
    boxes = [SubprocessSandbox() for _ in range(4)]
    for box in boxes:
        await box.start()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(box.exec("sleep 0.3; echo ok", timeout_s=10) for box in boxes)),
            timeout=20,
        )
        assert [(r.exit_code, r.stdout.strip()) for r in results] == [(0, "ok")] * 4
    finally:
        for box in boxes:
            await box.close()


async def test_proc_scans_do_not_block_event_loop(
    sandbox: SubprocessSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    main_thread = threading.get_ident()
    original = sandbox_module._uid_usage

    def checked(uid: int) -> dict[int, int]:
        assert threading.get_ident() != main_thread
        return original(uid)

    monkeypatch.setattr(sandbox_module, "_uid_usage", checked)
    assert (await sandbox.exec("true", timeout_s=3)).exit_code == 0
    await sandbox.close()


async def test_close_during_start_releases_directory_and_uid(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered = asyncio.Event()
    resume = asyncio.Event()
    helper = sandbox_module._helper

    async def delayed_probe(*args: str) -> tuple[int, str]:
        if args == ("--probe-seccomp",):
            entered.set()
            await resume.wait()
        return await helper(*args)

    monkeypatch.setattr(sandbox_module, "_helper", delayed_probe)
    instance = SubprocessSandbox()
    starting = asyncio.create_task(instance.start())
    await entered.wait()
    closing = asyncio.create_task(instance.close())
    await asyncio.sleep(0)
    resume.set()
    await starting
    await closing
    assert instance.uid is None
    assert not instance.workdir.exists()


async def test_preexisting_disk_error_respects_small_output_cap(sandbox_host: Path) -> None:
    instance = SubprocessSandbox(max_output_bytes=16, max_disk_bytes=1024)
    await instance.start()
    try:
        await instance.write_file("large", "x" * 2048)
        result = await instance.exec("true", timeout_s=3)
        assert result.exit_code != 0
        assert result.truncated
        assert len(result.stderr.encode()) <= 16
    finally:
        await instance.close()


def test_uid_lock_directory_falls_back_only_when_unavailable(
    sandbox_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    original = Path.mkdir

    def denied(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == sandbox_module._LOCK_ROOT:
            raise OSError(errno.EROFS, "read-only test location")
        original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    root = sandbox_module._lock_directory()
    assert root == sandbox_module._FALLBACK_LOCK_ROOT
    assert root.stat().st_uid == 0 and root.stat().st_mode & 0o777 == 0o700


async def test_file_size_limit_is_enforced(sandbox_host: Path) -> None:
    instance = SubprocessSandbox(limits=ResourceLimits(file_size_bytes=1024**2))
    await instance.start()
    try:
        result = await instance.exec(
            ["python3", "-c", "open('big', 'wb').write(b'x' * (2 * 1024**2))"], timeout_s=3
        )
        assert result.exit_code != 0
        assert (instance.workdir / "big").stat().st_size <= 1024**2
    finally:
        await instance.close()


async def test_multiprocessing_semaphores_are_unavailable(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        ["python3", "-c", "import multiprocessing; multiprocessing.Lock()"], timeout_s=3
    )
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr


@pytest.mark.parametrize("kind", ["SOCK_DGRAM", "SOCK_RAW", "SOCK_SEQPACKET"])
async def test_socketpairs_cannot_create_retargetable_datagram_sockets(
    sandbox: SubprocessSandbox, kind: str
) -> None:
    # AF_UNIX datagram socketpair ends can disconnect with AF_UNSPEC and then
    # connect/sendto a host's named (including abstract) listener, bypassing a
    # filter that denies socket() but admits every Unix socketpair type.
    result = await sandbox.exec(
        ["python3", "-c", f"import socket; socket.socketpair(socket.AF_UNIX, socket.{kind})"],
        timeout_s=3,
    )
    assert result.exit_code != 0
    assert "PermissionError: [Errno 13]" in result.stderr


async def test_nonblocking_cloexec_stream_socketpair_works(sandbox: SubprocessSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            """
import socket
left, right = socket.socketpair(
    socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_NONBLOCK | socket.SOCK_CLOEXEC
)
with left, right:
    left.sendall(b'pair works')
    assert right.recv(10) == b'pair works'
""",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0, result.stderr
