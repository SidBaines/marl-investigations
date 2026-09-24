"""Pin Docker's boundary using real CLI processes without a Docker daemon."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from marli.envs.sandbox import DockerPool, DockerSandbox, DockerSandboxConfig, make_sandbox
from marli.envs.sandbox import docker as module
from marli.errors import BackendError, ConfigError


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    cli = tmp_path / "docker"
    cli.write_text(
        f"#!{sys.executable}\n" + Path(__file__).with_name("_fake_docker.py").read_text()
    )
    cli.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    # No real containers or other tests' registrations can be swept by this fixture.
    monkeypatch.setattr(module, "_CONTAINERS", set())
    yield state
    module._sweep_containers()


@pytest.fixture
async def sandbox(fake_docker: Path) -> AsyncIterator[DockerSandbox]:
    instance = DockerSandbox()
    await instance.start()
    try:
        yield instance
    finally:
        await instance.close()


def calls(state: Path) -> list[list[str]]:
    return [json.loads(line) for line in (state / "argv.jsonl").read_text().splitlines()]


def workspace(state: Path, sandbox: DockerSandbox) -> Path:
    assert sandbox.container_id is not None
    return state / sandbox.container_id / "workspace"


async def test_run_exact_argv_start_and_close_idempotent(fake_docker: Path) -> None:
    sandbox = DockerSandbox()
    await sandbox.start()
    await sandbox.start()
    assert calls(fake_docker)[0] == [
        "run",
        "-d",
        "--rm",
        "--init",
        "--entrypoint",
        "",
        "--network",
        "none",
        "--cpus",
        "1.0",
        "--memory",
        "2048m",
        "--memory-swap",
        "2048m",
        "--pids-limit",
        "256",
        "--user",
        "65534:65534",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=256m",
        "--tmpfs",
        "/workspace:rw,exec,mode=1777,size=2048m",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--workdir",
        "/workspace",
        "--label",
        f"marli.sandbox={sandbox._label}",
        "python:3.12-slim",
        "sleep",
        "infinity",
    ]
    assert sandbox.workdir == Path("/workspace")
    container = sandbox.container_id
    await sandbox.close()
    await sandbox.close()
    assert calls(fake_docker).count(["rm", "-f", container]) == 1
    assert not module._CONTAINERS
    with pytest.raises(RuntimeError, match="closed"):
        await sandbox.start()


@pytest.mark.parametrize(
    "args",
    [
        ["--privileged"],
        ["--privileged=true"],
        ["-v", "/:/host"],
        ["-v/:/host"],
        ["--volume=/:/host"],
        ["--mount", "type=bind,src=/,dst=/host"],
        ["--volumes-from", "other"],
        ["--network", "host"],
        ["--net=host"],
        ["--pid", "host"],
        ["--pid=host"],
        ["--cap-add=ALL"],
        ["--security-opt=seccomp=unconfined"],
        ["--user=0"],
        ["--read-only=false"],
        ["--entrypoint=evil"],
        ["--pids-limit=-1"],
        ["--memory=0"],
        ["--"],
        ["image"],
        ["--label=marli.sandbox=other"],
        ["--hostname"],
    ],
)
def test_dangerous_extra_args_are_rejected(args: list[str]) -> None:
    with pytest.raises(ConfigError):
        DockerSandboxConfig(extra_run_args=args)


@pytest.mark.parametrize(
    "cfg",
    [
        {"network": "host", "allow_network": True},
        {"network": "bridge"},
        {"network": "container:other", "allow_network": True},
        {"cpus": 0},
        {"cpus": float("nan")},
        {"pids": 0},
        {"memory_mb": -1},
        {"max_output_bytes": 0},
        {"workdir": "/"},
        {"workdir": "relative"},
        {"workdir": "/work/../etc"},
        {"workdir": "/work:ro"},
        {"image": "--privileged"},
    ],
)
def test_invalid_config(cfg: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        DockerSandboxConfig(**cfg)


async def test_config_and_safe_escape_hatch(fake_docker: Path) -> None:
    config = DockerSandboxConfig(
        image="repo:task",
        cpus=0.5,
        memory_mb=128,
        pids=32,
        workdir="/task",
        read_only_root=False,
        network="bridge",
        allow_network=True,
        docker=shutil.which("docker") or "docker",
        extra_run_args=["--platform=linux/amd64", "--label", "study=test"],
    )
    sandbox = DockerSandbox(config)
    await sandbox.start()
    try:
        argv = calls(fake_docker)[0]
        assert "--read-only" not in argv
        assert argv[argv.index("--network") + 1] == "bridge"
        assert argv[argv.index("--cpus") + 1] == "0.5"
        assert argv[-6:] == [
            "--platform=linux/amd64",
            "--label",
            "study=test",
            "repo:task",
            "sleep",
            "infinity",
        ]
        await sandbox.write_file("nested/file", "hello")
        assert await sandbox.read_file("nested/file") == "hello"
    finally:
        await sandbox.close()


async def test_exec_argv_stdin_exit_code_and_stateless_shell(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec(
        [
            "python3",
            "-c",
            "import sys; print(sys.stdin.read()); print('err', file=sys.stderr); sys.exit(7)",
        ],
        timeout_s=5,
        stdin="héllo",
    )
    assert (result.exit_code, result.stdout, result.stderr) == (7, "héllo\n", "err\n")
    assert not result.timed_out and not result.truncated and result.duration_s > 0
    await sandbox.write_file(".bashrc", "echo bad")
    await sandbox.exec("mkdir sub; cd sub; export MARLI_TEST_LOCAL=bad", timeout_s=3)
    result = await sandbox.exec('test -d sub && printf "${MARLI_TEST_LOCAL-clean}"', timeout_s=3)
    assert result.stdout == "clean" and result.exit_code == 0
    await sandbox.exec(["printf", "%s", "$(echo no); literal"], timeout_s=3)
    await sandbox.exec("printf persistent > data", timeout_s=3)
    assert await sandbox.read_file("data") == "persistent"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
async def test_invalid_timeout(sandbox: DockerSandbox, timeout: float) -> None:
    with pytest.raises(ValueError):
        await sandbox.exec("true", timeout_s=timeout)


async def test_capped_head_tail_and_overflow(fake_docker: Path) -> None:
    sandbox = DockerSandbox(DockerSandboxConfig(max_output_bytes=100))
    await sandbox.start()
    try:
        result = await sandbox.exec(
            [
                "python3",
                "-c",
                "import sys; sys.stdout.write('a'*100+'z'*100); sys.stderr.write('b'*100+'y'*100)",
            ],
            timeout_s=5,
        )
        assert result.exit_code == 0
        assert result.stdout == "a" * 50 + "z" * 50
        assert result.stderr == "b" * 50 + "y" * 50
        assert result.truncated
        result = await sandbox.exec(
            ["python3", "-c", "import os\nwhile True: os.write(1, b'x'*65536)"],
            timeout_s=5,
        )
        assert result.exit_code != 0 and not result.timed_out and result.truncated
        assert len(result.stdout.encode()) == 100
        assert len(result.stderr.encode()) <= 100
        assert "max_output_bytes" in result.stderr
    finally:
        await sandbox.close()


async def test_timeout_kills_children_and_allows_next_command(
    sandbox: DockerSandbox,
    fake_docker: Path,
) -> None:
    result = await sandbox.exec(
        "sleep 30 & echo $! > child; wait",
        timeout_s=0.2,
    )
    assert result.timed_out and result.exit_code is None
    pid = int((workspace(fake_docker, sandbox) / "child").read_text())
    assert_dead(pid)
    assert (await sandbox.exec("echo alive", timeout_s=3)).stdout == "alive\n"


def assert_dead(pid: int) -> None:
    status = Path(f"/proc/{pid}/status")
    assert not status.exists() or "\nState:\tZ" in status.read_text()


async def test_cancellation_stops_container_command(
    sandbox: DockerSandbox,
    fake_docker: Path,
) -> None:
    task = asyncio.create_task(sandbox.exec("echo $$ > child; sleep 30", timeout_s=40))
    target = workspace(fake_docker, sandbox) / "child"
    for _ in range(100):
        if target.exists():
            break
        await asyncio.sleep(0.02)
    assert target.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_dead(int(target.read_text()))
    assert (await sandbox.exec("true", timeout_s=3)).exit_code == 0


@pytest.mark.parametrize("rel", ["../escape", "a/../../escape", "/etc/passwd", "", ".", "a/../b"])
async def test_lexical_path_confinement(sandbox: DockerSandbox, rel: str) -> None:
    with pytest.raises(ValueError):
        await sandbox.read_file(rel)
    with pytest.raises(ValueError):
        await sandbox.write_file(rel, "bad")


async def test_symlinks_and_nonfiles_are_rejected(
    sandbox: DockerSandbox,
    fake_docker: Path,
    tmp_path: Path,
) -> None:
    external = tmp_path / "valuable"
    external.write_text("keep")
    root = workspace(fake_docker, sandbox)
    (root / "link").symlink_to(external)
    (root / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    (root / "dangling").symlink_to(tmp_path / "missing")
    os.mkfifo(root / "fifo")
    for rel in ("link", "dirlink/valuable", "dangling", "fifo"):
        with pytest.raises(ValueError):
            await sandbox.read_file(rel)
        with pytest.raises(ValueError):
            await sandbox.write_file(rel, "bad")
    assert external.read_text() == "keep"
    assert not (tmp_path / "missing").exists()


async def test_copy_roundtrip_overwrite_reset(sandbox: DockerSandbox, fake_docker: Path) -> None:
    await sandbox.write_file("nested/file with spaces", "héllo\n")
    assert await sandbox.read_file("nested/file with spaces") == "héllo\n"
    await sandbox.exec("printf updated >> 'nested/file with spaces'", timeout_s=3)
    assert await sandbox.read_file("nested/file with spaces") == "héllo\nupdated"
    await sandbox.write_file("nested/file with spaces", "new")
    assert await sandbox.read_file("nested/file with spaces") == "new"
    await sandbox.write_file(".hidden", "yes")
    await sandbox.reset()
    assert list(workspace(fake_docker, sandbox).iterdir()) == []
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file("missing")
    assert any(call[0] == "cp" for call in calls(fake_docker))


async def test_reset_interrupts_running_exec(sandbox: DockerSandbox, fake_docker: Path) -> None:
    task = asyncio.create_task(sandbox.exec("echo $$ > child; sleep 30", timeout_s=40))
    root = workspace(fake_docker, sandbox)
    for _ in range(100):
        if (root / "child").exists():
            break
        await asyncio.sleep(0.02)
    pid = int((root / "child").read_text())
    await asyncio.wait_for(sandbox.reset(), 5)
    await task
    assert_dead(pid)
    assert list(root.iterdir()) == []


async def test_exit_sweep_only_owned_labels(fake_docker: Path) -> None:
    first, second = DockerSandbox(), DockerSandbox()
    await first.start()
    await second.start()
    foreign = fake_docker / ("f" * 32)
    foreign.mkdir()
    (foreign / "metadata.json").write_text(json.dumps({"labels": ["marli.sandbox=foreign"]}))
    module._sweep_containers()
    assert foreign.exists()
    assert not (fake_docker / str(first.container_id)).exists()
    assert not (fake_docker / str(second.container_id)).exists()
    assert not module._CONTAINERS
    assert all("label=marli.sandbox=" in call[-1] for call in calls(fake_docker) if call[0] == "ps")


async def test_failed_and_cancelled_start_are_swept(fake_docker: Path) -> None:
    (fake_docker / "fail-exec").touch()
    with pytest.raises(BackendError):
        await DockerSandbox().start()
    (fake_docker / "fail-exec").unlink()
    assert not module._CONTAINERS
    (fake_docker / "delay-run").write_text("1")
    sandbox = DockerSandbox()
    task = asyncio.create_task(sandbox.start())
    for _ in range(100):
        if any(path.is_dir() for path in fake_docker.iterdir()):
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not module._CONTAINERS
    assert not any(path.is_dir() for path in fake_docker.iterdir())


async def test_pool_prewarms_blocks_and_resets(fake_docker: Path) -> None:
    pool = DockerPool(DockerSandboxConfig(), size=2)
    await pool.start()
    assert len([call for call in calls(fake_docker) if call[0] == "run"]) == 2
    first, second = await pool.acquire(), await pool.acquire()
    assert first is not second
    pending = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.05)
    assert not pending.done()
    await first.write_file("dirty", "yes")
    await pool.release(first)
    reused = await asyncio.wait_for(pending, 5)
    assert reused is first
    with pytest.raises(FileNotFoundError):
        await reused.read_file("dirty")
    await pool.release(reused)
    await pool.release(second)
    with pytest.raises(ValueError):
        await pool.release(second)
    await pool.close()
    await pool.close()
    assert not module._CONTAINERS


async def test_pool_retires_unhealthy_container(fake_docker: Path) -> None:
    pool = DockerPool(DockerSandboxConfig(), size=1)
    sandbox = await pool.acquire()
    await sandbox.close()
    with pytest.raises(RuntimeError):
        await pool.release(sandbox)
    replacement = await pool.acquire()
    assert replacement is not sandbox
    await pool.release(replacement)
    await pool.close()


async def test_factory(fake_docker: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import marli.envs.sandbox as package

    sandbox = await make_sandbox("docker", memory_mb=128)
    assert isinstance(sandbox, DockerSandbox) and sandbox.config.memory_mb == 128
    await sandbox.close()
    pool = DockerPool(DockerSandboxConfig(), 1)
    sandbox = await make_sandbox("docker", pool=pool)
    await pool.release(sandbox)
    await pool.close()
    with pytest.raises(ConfigError):
        await make_sandbox("unknown")
    with pytest.raises(ConfigError):
        await make_sandbox("docker", bad_key=True)
    with pytest.raises(ConfigError):
        await make_sandbox("subprocess", pool=pool)
    with pytest.raises(ConfigError):
        await make_sandbox("docker", pool=pool, image="ignored")

    class FakeSubprocess:
        def __init__(self, **cfg: Any) -> None:
            self.cfg = cfg
            self.started = False

        async def start(self) -> None:
            self.started = True

    monkeypatch.setattr(package, "SubprocessSandbox", FakeSubprocess)
    local = await make_sandbox("subprocess", max_output_bytes=7)
    assert local.started and local.cfg == {"max_output_bytes": 7}


@pytest.fixture
async def real_docker() -> AsyncIterator[DockerSandbox]:
    if os.environ.get("MARLI_TEST_DOCKER") != "1":
        pytest.skip("opt in with MARLI_TEST_DOCKER=1; requires a cached python:3.12-slim image")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker unavailable")
    try:
        subprocess.run([docker, "info"], capture_output=True, timeout=10, check=True)
        subprocess.run(
            [docker, "image", "inspect", "python:3.12-slim"],
            capture_output=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("docker daemon or cached image unavailable")
    sandbox = DockerSandbox(DockerSandboxConfig(memory_mb=128, pids=32))
    await sandbox.start()
    try:
        yield sandbox
    finally:
        await sandbox.close()


@pytest.mark.docker
async def test_real_docker_network_and_persistent_files(real_docker: DockerSandbox) -> None:
    result = await real_docker.exec(
        ["python3", "-c", "import socket; socket.create_connection(('192.0.2.1', 80), timeout=.2)"],
        timeout_s=3,
    )
    assert result.exit_code != 0 and not result.timed_out
    await real_docker.write_file("hello", "hello")
    assert (await real_docker.exec("chmod 600 hello", timeout_s=3)).exit_code == 0
    assert (await real_docker.exec("echo world >> hello", timeout_s=3)).exit_code == 0
    assert await real_docker.read_file("hello") == "helloworld\n"
    result = await real_docker.exec("id -u; touch /root/forbidden", timeout_s=3)
    assert result.stdout == "65534\n" and result.exit_code != 0
    await real_docker.reset()
    assert (await real_docker.exec("test ! -e hello", timeout_s=3)).exit_code == 0


@pytest.mark.docker
async def test_real_docker_timeout_fork_loop(real_docker: DockerSandbox) -> None:
    result = await real_docker.exec(
        [
            "python3",
            "-c",
            """
import os, time
while True:
    try:
        if os.fork() == 0:
            time.sleep(30)
            os._exit(0)
    except OSError:
        time.sleep(.01)
""",
        ],
        timeout_s=0.5,
    )
    assert result.timed_out
    assert (await real_docker.exec("echo alive", timeout_s=3)).stdout == "alive\n"
    result = await real_docker.exec(
        [
            "python3",
            "-c",
            "from pathlib import Path; "
            "print(sum(p.name.isdecimal() and 'python' in (p/'comm').read_text() "
            "for p in Path('/proc').iterdir()))",
        ],
        timeout_s=3,
    )
    assert result.stdout == "1\n"


@pytest.mark.docker
async def test_real_docker_memory_limit(real_docker: DockerSandbox) -> None:
    result = await real_docker.exec(["python3", "-c", "x = bytearray(512 * 1024**2)"], timeout_s=5)
    assert result.exit_code != 0 and not result.timed_out
    assert (await real_docker.exec("echo alive", timeout_s=3)).stdout == "alive\n"


async def test_exit_137_is_not_a_timeout(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("exit 137", timeout_s=3)
    assert result.exit_code == 137 and not result.timed_out


async def test_host_watchdog_kills_hung_cli(sandbox: DockerSandbox, fake_docker: Path) -> None:
    (fake_docker / "hang-exec").touch()
    result = await sandbox.exec("true", timeout_s=0.2)
    assert result.timed_out and result.exit_code is None
    assert result.duration_s < 4
    assert_dead(int((fake_docker / "hung-client").read_text()))
    (fake_docker / "hang-exec").unlink()
    assert (await sandbox.exec("echo recovered", timeout_s=3)).stdout == "recovered\n"


async def test_close_removes_container_when_exec_is_broken(fake_docker: Path) -> None:
    sandbox = DockerSandbox()
    await sandbox.start()
    cid = sandbox.container_id
    (fake_docker / "fail-exec").touch()
    await sandbox.close()
    assert not (fake_docker / str(cid)).exists()
    await sandbox.close()


async def test_pool_close_wakes_waiter_and_closes_leases(fake_docker: Path) -> None:
    pool = DockerPool(DockerSandboxConfig(), 1)
    sandbox = await pool.acquire()
    pending = asyncio.create_task(pool.acquire())
    await asyncio.sleep(0.05)
    await pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        await pending
    with pytest.raises(RuntimeError, match="not running"):
        await sandbox.exec("true", timeout_s=3)
    assert not module._CONTAINERS


def test_cleanup_script_repeats_scan_and_preserves_keeper(tmp_path: Path) -> None:
    # Execute the actual shell logic against synthetic proc entries. The shell
    # function replaces kill, so this test never signals any host process.
    proc = tmp_path / "proc"
    for pid in (1, 2, 10, 11, 13):
        entry = proc / str(pid)
        entry.mkdir(parents=True)
        (entry / "stat").write_text(f"{pid} (name with ) spaces) S 0\n")
    (proc / "13" / "stat").write_text("13 (fake) Z\nname) S 0\n")
    script = module._STOP_COMMANDS.replace("/proc/", str(proc) + "/")
    mocked_kill = """
kill() {
    printf '%s\n' "$2" >> "$MARLI_KILLS"
    printf '%s (dead) Z 0\n' "$2" > "$MARLI_PROC/$2/stat"
    if [ "$2" = 10 ]; then
        mkdir -p "$MARLI_PROC/12"
        printf '12 (forked) S 0\n' > "$MARLI_PROC/12/stat"
    fi
}
"""
    record = tmp_path / "killed"
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", mocked_kill + script, "test", "2", ""],
        env={**os.environ, "MARLI_KILLS": str(record), "MARLI_PROC": str(proc)},
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stderr == ""
    assert record.read_text().splitlines() == ["10", "11", "13", "12"]


@pytest.mark.docker
async def test_real_docker_detached_child_is_killed(real_docker: DockerSandbox) -> None:
    result = await real_docker.exec(
        [
            "python3",
            "-c",
            """
import os, time
if os.fork() == 0:
    os.setsid()
    with open('child', 'w') as f:
        f.write(str(os.getpid()))
    os.close(1)
    os.close(2)
    time.sleep(30)
else:
    while not os.path.exists('child'):
        time.sleep(.01)
""",
        ],
        timeout_s=3,
    )
    assert result.exit_code == 0
    pid = (await real_docker.read_file("child")).strip()
    result = await real_docker.exec(f"test ! -d /proc/{pid}", timeout_s=3)
    assert result.exit_code == 0
