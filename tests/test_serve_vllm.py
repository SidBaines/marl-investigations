"""Exercise real process ownership using only loopback HTTP and CPU children."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from marli.errors import BackendError, ConfigError
from marli.model import load_model
from marli.policy.vllm import read_server_json
from marli.serve import vllm as serving
from marli.serve._supervise import Supervisor, process_alive
from marli.serve.vllm import Server, VLLMServeConfig, launch_args
from marli.verbs import run_verb

ROOT = Path(__file__).resolve().parents[1]
requires_loopback = pytest.mark.skipif(
    os.environ.get("MARLI_TEST_NO_NETWORK") == "1", reason="builder forbids host network use"
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_server(tmp_path: Path, *, detach: bool = False) -> Supervisor:
    port = free_port()
    models = tmp_path / "v1" / "models"
    models.parent.mkdir(parents=True, exist_ok=True)
    models.write_text('{"data": []}')
    return Supervisor(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(tmp_path),
        ],
        env={"PYTHONUNBUFFERED": "1"},
        health_url=f"http://127.0.0.1:{port}/v1/models",
        log=tmp_path / "server.log",
        ready_timeout_s=5,
        grace_s=0.2,
        detach=detach,
    )


@requires_loopback
async def test_supervisor_ready_tee_stop(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    old = signal.getsignal(signal.SIGTERM)
    child = http_server(tmp_path)
    try:
        await child.start()
        assert process_alive(child.pid)
        assert os.getpgid(child.pid) == child.pid
    finally:
        child.stop()
    assert not process_alive(child.pid)
    assert "GET /v1/models" in child.log.read_text()
    assert "GET /v1/models" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == old
    child.stop()


@requires_loopback
async def test_supervisor_crash_includes_last_50_lines(tmp_path: Path) -> None:
    child = Supervisor(
        [
            sys.executable,
            "-c",
            "for i in range(70): print(f'crash-line-{i:02d}')\nraise SystemExit(7)",
        ],
        log=tmp_path / "crash.log",
        health_url=f"http://127.0.0.1:{free_port()}/",
        ready_timeout_s=5,
        grace_s=0.1,
    )
    with pytest.raises(BackendError, match="code 7") as error:
        await child.start()
    tail = str(error.value).split("Last 50 log lines", 1)[1]
    assert "crash-line-69" in tail and "crash-line-20" in tail
    assert "crash-line-19" not in tail
    assert not process_alive(child.pid)


@requires_loopback
async def test_supervisor_deadline_and_missing_python(tmp_path: Path) -> None:
    child = Supervisor(
        [sys.executable, "-c", "import time; print('starting', flush=True); time.sleep(30)"],
        log=tmp_path / "timeout.log",
        health_url=f"http://127.0.0.1:{free_port()}/",
        ready_timeout_s=0.3,
        grace_s=0.1,
    )
    with pytest.raises(BackendError, match="deadline.*\n.*\nstarting"):
        await child.start()
    assert not process_alive(child.pid)
    child = Supervisor(
        [str(tmp_path / "missing-python")],
        log=tmp_path / "missing.log",
        health_url="http://127.0.0.1:1/",
    )
    with pytest.raises(BackendError, match="No such file"):
        await child.start()


@requires_loopback
async def test_stop_kills_stubborn_descendant(tmp_path: Path) -> None:
    port = free_port()
    pid_file = tmp_path / "worker.pid"
    code = f"""
import os, signal, time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path({str(pid_file)!r}).write_text(str(os.getpid()))
    while True: time.sleep(1)
HTTPServer(('127.0.0.1', {port}), SimpleHTTPRequestHandler).serve_forever()
"""
    child = Supervisor(
        [sys.executable, "-c", code],
        log=tmp_path / "group.log",
        health_url=f"http://127.0.0.1:{port}/",
        ready_timeout_s=5,
        grace_s=0.1,
    )
    try:
        await child.start()
        worker = int(pid_file.read_text())
        assert process_alive(worker)
    finally:
        child.stop()
    deadline = time.monotonic() + 3
    while process_alive(worker) and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert not process_alive(worker)
    assert not process_alive(child.pid)


def python_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}


@pytest.mark.parametrize("mode", ["exit", "term", "detach"])
@requires_loopback
def test_parent_exit_cleanup_and_detached_survival(tmp_path: Path, mode: str) -> None:
    port = free_port()
    pid_file = tmp_path / "child.pid"
    code = f"""
import asyncio, time
from pathlib import Path
from marli.serve._supervise import Supervisor
child = Supervisor(
    {[sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]!r},
    log=Path({str(tmp_path / "parent.log")!r}),
    health_url='http://127.0.0.1:{port}/', ready_timeout_s=5, grace_s=0.1,
    detach={mode == "detach"!r},
)
asyncio.run(child.start())
if {mode == "detach"!r}: child.release()
Path({str(pid_file)!r}).write_text(str(child.pid))
if {mode == "term"!r}: time.sleep(30)
"""
    parent = subprocess.Popen([sys.executable, "-c", code], env=python_env())
    child_pid = None
    try:
        deadline = time.monotonic() + 8
        while not pid_file.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        if mode == "term":
            parent.terminate()
        parent.wait(timeout=5)
        assert process_alive(child_pid) == (mode == "detach")
        assert parent.returncode == (143 if mode == "term" else 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid is not None and process_alive(child_pid):
            os.killpg(child_pid, signal.SIGKILL)


def test_vllm_argv_env_and_runtime_hash() -> None:
    cfg = VLLMServeConfig(
        model="qwen3_5_4b",
        python="/opt/vllm/bin/python",
        cuda_visible_devices="0",
        learner_ranks=[16, 32],
        extra_args=["--dtype", "bfloat16"],
    )
    model, argv, env = launch_args(cfg)
    assert argv == [
        "/opt/vllm/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen3.5-4B",
        "--served-model-name",
        "qwen3_5_4b",
        "--generation-config",
        "vllm",
        "--max-model-len",
        str(model.max_ctx),
        "--gpu-memory-utilization",
        "0.85",
        "--enable-lora",
        "--max-loras",
        "4",
        "--max-lora-rank",
        "32",
        "--tensor-parallel-size",
        "1",
        "--seed",
        "0",
        "--port",
        "8000",
        "--host",
        "127.0.0.1",
        "--dtype",
        "bfloat16",
    ]
    assert env == {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True", "CUDA_VISIBLE_DEVICES": "0"}
    _, argv, env = launch_args(
        replace(cfg, enable_lora=False, learner_ranks=[], cuda_visible_devices=None)
    )
    assert "--enable-lora" not in argv and "--max-loras" not in argv
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "--enable-sleep-mode" not in argv and "VLLM_SERVER_DEV_MODE" not in env
    _, argv, env = launch_args(
        replace(cfg, tensor_parallel_size=2, enable_sleep_mode=True, cuda_visible_devices="0,1")
    )
    tail = argv[argv.index("--tensor-parallel-size") :]
    assert tail[:5] == ["--tensor-parallel-size", "2", "--enable-sleep-mode", "--seed", "0"]
    assert env == {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "VLLM_SERVER_DEV_MODE": "1",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    from marli.config import config_hash

    assert config_hash(cfg) == config_hash(replace(cfg, detach=True, ready_timeout_s=30))


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"model": "missing"}, "missing"),
        ({"max_model_len": 10000000}, "model.max_ctx"),
        ({"learner_ranks": [64]}, "max_lora_rank"),
        ({"learner_ranks": [16, 16], "max_loras": 2}, "snapshot"),
        ({"port": 0}, "port"),
        ({"gpu_memory_utilization": 1.1}, "utilization"),
        ({"enable_sleep_mode": 1}, "enable_sleep_mode"),
    ],
)
def test_validation_before_launch(overrides: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        launch_args(
            VLLMServeConfig(**{"model": "qwen3_5_4b", "python": sys.executable, **overrides})
        )


@requires_loopback
async def test_server_roundtrip_and_serve_detach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = http_server(tmp_path)
    port = int(child.health_url.split(":")[2].split("/")[0])
    monkeypatch.setattr(
        serving,
        "launch_args",
        lambda cfg: (
            load_model(cfg.model),
            child.argv,
            child.env,
        ),
    )
    try:
        result = await run_verb(
            "serve vllm",
            VLLMServeConfig(
                model="qwen3_5_4b", python=sys.executable, port=port, detach=True, ready_timeout_s=5
            ),
            out=tmp_path / "serve",
        )
        server = Server.load(result.manifest)
        assert read_server_json(result.manifest) == (server.base_url, {"qwen3_5_4b": "qwen3_5_4b"})
        assert server.adapters == [] and server.log == "server.log"
        assert server.max_loras == 4 and server.max_lora_rank == 32
        assert server.tensor_parallel_size == 1 and server.enable_sleep_mode is False
        assert process_alive(server.pid)
        assert server.config_hash == result.config_hash
    finally:
        if (tmp_path / "serve/server.json").exists():
            os.killpg(Server.load(tmp_path / "serve").pid, signal.SIGKILL)


@requires_loopback
async def test_cli_status_and_stop(tmp_path: Path) -> None:
    child = http_server(tmp_path)
    await child.start()
    server = Server(
        root=tmp_path,
        base_url=child.health_url.removesuffix("/v1/models"),
        models=["base"],
        pid=child.pid,
        log="server.log",
    )
    server.save()
    try:
        for verb in ("status", "stop", "status"):
            out = tmp_path / f"{verb}-{time.monotonic_ns()}"
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-m",
                    "marli.cli.main",
                    "serve",
                    verb,
                    f"server_json={server.manifest_path}",
                    "timeout_s=0.1",
                    "--out",
                    str(out),
                ],
                env=python_env(),
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, result.stderr + result.stdout
            assert len(result.stdout.splitlines()) == 1
            data = json.loads(result.stdout)
            assert data["ok"] is True and data["pid"] == child.pid
            expected = verb == "status" and child.process.poll() is None
            assert data["running"] == expected and data["healthy"] == expected
            assert Path(data["manifest"]).is_file()
    finally:
        child.stop()
    assert server.file("log").exists()


def test_serve_and_help_import_no_heavy_dependencies() -> None:
    code = """
import sys
import marli.serve
from marli.cli.main import main
try: main(['--help'])
except SystemExit as e: assert e.code == 0
assert not {'torch', 'peft', 'vllm', 'transformers', 'datasets'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], env=python_env(), capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@requires_loopback
def test_foreground_manifest_ready_then_signal_stops_group(tmp_path: Path, signum: int) -> None:
    port = free_port()
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1/models").write_text("{}")
    out = tmp_path / "foreground"
    code = f"""
import asyncio
import marli.serve.vllm as serving
from marli.model import load_model
from marli.verbs import run_verb
serving.launch_args = lambda cfg: (load_model(cfg.model),
    {
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            "127.0.0.1",
            "--directory",
            str(tmp_path),
        ]!r
    }, {{}})
cfg = serving.VLLMServeConfig(model='qwen3_5_4b', python={sys.executable!r},
                             port={port}, ready_timeout_s=5)
result = asyncio.run(run_verb('serve vllm', cfg, out={str(out)!r}))
assert result.handle.pid > 1
"""
    parent = subprocess.Popen([sys.executable, "-c", code], env=python_env())
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not (out / "server.json").exists() and time.monotonic() < deadline:
            assert parent.poll() is None
            time.sleep(0.02)
        server = Server.load(out)
        child_pid = server.pid
        assert process_alive(child_pid) and parent.poll() is None
        parent.send_signal(signum)
        parent.wait(timeout=10)
        assert parent.returncode == 0
        assert not process_alive(child_pid)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid is not None and process_alive(child_pid):
            os.killpg(child_pid, signal.SIGKILL)


@requires_loopback
async def test_foreground_reports_post_ready_crash(tmp_path: Path) -> None:
    child = http_server(tmp_path)
    await child.start()
    os.killpg(child.pid, signal.SIGKILL)
    with pytest.raises(BackendError, match="server exited"):
        await child.wait()
    assert not process_alive(child.pid)


@requires_loopback
async def test_cancelled_start_cleans_up(tmp_path: Path) -> None:
    child = Supervisor(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        log=tmp_path / "cancel.log",
        health_url=f"http://127.0.0.1:{free_port()}/",
        grace_s=0.1,
    )
    task = asyncio.create_task(child.start())
    while child.process is None:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not process_alive(child.pid)


def test_server_path_puts_the_vllm_venv_bin_first():
    from marli.serve.vllm import venv_path

    assert venv_path("/opt/vllm/bin/python", "/usr/bin:/opt/vllm/bin:/bin") == (
        "/opt/vllm/bin:/usr/bin:/bin"
    )
    assert venv_path("/opt/vllm/bin/python", "") == "/opt/vllm/bin"
