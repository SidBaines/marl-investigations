"""The live dashboard serves only its page and snapshot, behind a key off loopback."""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_dashboard import dash

from marli.dashboard.serve import COOKIE, KEY_ENV, DashboardServer, access_key, is_loopback
from marli.verbs import run_verb


def get(
    server: DashboardServer, path: str, headers: dict[str, str] | None = None, method: str = "GET"
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


@pytest.fixture
def site(tmp_path: Path) -> Path:
    (tmp_path / "standalone.html").write_text("<!doctype html><p>page</p>")
    (tmp_path / "snapshot.json").write_text('{"refresh": 1}')
    for private in ("cache.json", "config.yaml", "serve.json", "dashboard.json"):
        (tmp_path / private).write_text("private")
    return tmp_path


@pytest.fixture
def keyed(site: Path) -> Iterator[DashboardServer]:
    server = DashboardServer(site, "127.0.0.1", 0, "s3cret-key")
    server.start()
    try:
        yield server
    finally:
        server.close()


def test_loopback_and_key_choice() -> None:
    assert is_loopback("127.0.0.1") and is_loopback("127.0.0.2") and is_loopback("::1")
    assert is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("10.1.2.3") and not is_loopback("pod")
    assert access_key("127.0.0.1", {KEY_ENV: "ignored"}) is None
    assert access_key("0.0.0.0", {KEY_ENV: " stable "}) == "stable"
    fresh = {access_key("0.0.0.0", {}) for _ in range(5)}
    assert len(fresh) == 5 and all(key and len(key) >= 24 for key in fresh)


def test_loopback_server_needs_no_key_and_serves_only_the_allowlist(site: Path) -> None:
    server = DashboardServer(site, "127.0.0.1", 0, None)
    server.start()
    try:
        assert server.url() == f"http://127.0.0.1:{server.port}/"
        status, headers, body = get(server, "/")
        assert status == 200 and body == b"<!doctype html><p>page</p>"
        assert headers["Content-Type"].startswith("text/html")
        assert headers["Cache-Control"] == "no-store"
        assert get(server, "/snapshot.json?t=123")[2] == b'{"refresh": 1}'
        assert get(server, "/standalone.html")[0] == 200
        for path in (
            "/cache.json",
            "/config.yaml",
            "/serve.json",
            "/dashboard.json",
            "/../snapshot.json",
            "/%2e%2e/config.yaml",
            "//etc/passwd",
            "/index.html",
        ):
            assert get(server, path)[0] == 404, path
        status, _, body = get(server, "/", method="HEAD")
        assert status == 200 and body == b""
        (site / "snapshot.json").unlink()
        assert get(server, "/snapshot.json")[0] == 503
    finally:
        server.close()


def test_key_sets_cookie_and_redirects_then_cookie_grants_access(keyed: DashboardServer) -> None:
    assert keyed.url() == f"http://127.0.0.1:{keyed.port}/?key=s3cret-key"
    assert keyed.url(with_key=False) == f"http://127.0.0.1:{keyed.port}/"
    for path, headers in (
        ("/", {}),
        ("/snapshot.json", {}),
        ("/?key=wrong", {}),
        ("/", {"Cookie": f"{COOKIE}=wrong"}),
        ("/", {"Cookie": "garbage;;=="}),
    ):
        status, _, body = get(keyed, path, headers)
        assert status == 403 and b"s3cret" not in body
    status, headers, _ = get(keyed, "/?key=s3cret-key")
    assert status == 303 and headers["Location"] == "/"
    cookie = headers["Set-Cookie"]
    assert cookie.startswith(f"{COOKIE}=s3cret-key;")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Secure" not in cookie
    _, proxied, _ = get(keyed, "/?key=s3cret-key", {"X-Forwarded-Proto": "https"})
    assert "Secure" in proxied["Set-Cookie"]
    jar = {"Cookie": f"{COOKIE}=s3cret-key"}
    assert get(keyed, "/", jar)[0] == 200
    assert get(keyed, "/snapshot.json?t=9", jar)[2] == b'{"refresh": 1}'
    assert get(keyed, "/cache.json", jar)[0] == 404


async def test_verb_serves_while_refreshing_and_keeps_the_key_out_of_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(KEY_ENV, "verb-key-123")
    (tmp_path / "experiments").mkdir()
    out = tmp_path / "dash"
    cfg = dash(tmp_path, serve_port=0, serve_host="127.0.0.1", watch_s=0.05)
    task = asyncio.create_task(run_verb("dashboard", cfg, out=out))
    record = None
    for _ in range(200):
        await asyncio.sleep(0.05)
        if (out / "serve.json").exists():
            record = json.loads((out / "serve.json").read_text())
            break
    assert record is not None and record["url"].endswith(f":{record['port']}/")
    assert (out / "serve.json").stat().st_mode & 0o077 == 0
    conn = http.client.HTTPConnection("127.0.0.1", record["port"], timeout=10)
    conn.request("GET", "/snapshot.json")
    response = conn.getresponse()
    assert response.status == 200 and "runs" in json.loads(response.read())
    conn.close()
    assert not task.done()
    os.kill(os.getpid(), signal.SIGINT)
    result = await asyncio.wait_for(task, timeout=10)
    assert result.handle.refreshes >= 1
    assert result.handle.url == f"http://127.0.0.1:{record['port']}/"
    assert not (out / "serve.json").exists()
    assert any("dashboard live at" in message for message in caplog.messages)
    for name in ("dashboard.json", "config.yaml", "snapshot.json"):
        assert "verb-key-123" not in (out / name).read_text()


def test_render_opens_charted_train_runs_by_default() -> None:
    from marli.dashboard.render import render

    assert 'run.status === "running" || charted' in render({"runs": []})
