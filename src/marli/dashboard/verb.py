"""`marli dashboard`: observe run dirs and write snapshot.json + two HTML renderings.

With ``serve_port`` the verb also serves the live page over HTTP while it keeps
refreshing (see ``marli.dashboard.serve``): run it beside any training or eval
runs, with a per-study YAML for its charts, and open the printed URL.

A dashboard has no scientific identity: every setting is a runtime field, so the
config hash is constant and one ``--out`` is refreshed by every invocation (the
resume hook always reopens a finished dashboard). With ``watch_s > 0`` the verb
keeps rewriting its files (atomically) until SIGINT/SIGTERM, then records its
manifest. The fold cache lives in the out dir (``cache.json``), so repeated
refreshes parse only rows appended since the last one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from marli.config import runtime_field
from marli.dashboard.collect import SYSTEM_AGENT, collect
from marli.dashboard.render import render
from marli.dashboard.serve import DashboardServer, access_key, is_loopback
from marli.errors import ConfigError
from marli.handles import Handle, atomic_write_text, register_handle
from marli.rundir import RunDir

log = logging.getLogger("marli")


def _default_roots() -> list[str]:
    return ["experiments", "runs"]


@dataclass
class DashboardConfig:
    roots: list[str] = runtime_field(
        default_factory=_default_roots, help="directories scanned for run dirs (relative to cwd)"
    )
    annotations: str | None = runtime_field(
        None, help="JSON file shown verbatim at the top (e.g. pod ids, $/hr, spend)"
    )
    title: str = runtime_field("marli runs", help="page title")
    stale_s: float = runtime_field(180.0, help="an unowned or quiet run older than this is flagged")
    refresh_s: float = runtime_field(
        60.0, help="page poll interval for snapshot.json when served over http(s)"
    )
    probe_servers: bool = runtime_field(True, help="GET each live server's /metrics (2 s timeout)")
    gpus: bool = runtime_field(True, help="query nvidia-smi when present")
    watch_s: float = runtime_field(0.0, help="> 0: keep refreshing every watch_s s until SIGINT")
    max_refreshes: int | None = runtime_field(None, help="stop watching after N refreshes")
    grouped: dict[str, str] = runtime_field(
        default_factory=dict,
        help="train rl: grade component -> chart title; one chart per component with a line"
        " per agent plus the average, in mapping order (empty: no such charts)",
    )
    agent_labels: dict[str, str] = runtime_field(
        default_factory=dict,
        help="legend label per agent id in grouped charts (default: the id; _system is 'average')",
    )
    smooth_steps: int = runtime_field(
        5, help="grouped charts: trailing rolling-mean window in steps (1 = no smoothing)"
    )
    serve_port: int | None = runtime_field(
        None,
        help="serve the live page on this port (0 = any free port) and keep refreshing every"
        " watch_s (default refresh_s) until SIGINT; the URL is logged and written to serve.json",
    )
    serve_host: str = runtime_field(
        "127.0.0.1",
        help="bind address for serve_port; off loopback (e.g. 0.0.0.0 behind a pod's HTTPS proxy)"
        " every request needs the access key ($MARLI_DASHBOARD_KEY, else random per start)",
    )

    def __post_init__(self) -> None:
        if not self.roots or not all(isinstance(root, str) and root for root in self.roots):
            raise ConfigError("roots must be a nonempty list of paths")
        for name in ("stale_s", "refresh_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ConfigError(f"{name} must be a positive number")
        if (
            isinstance(self.watch_s, bool)
            or not isinstance(self.watch_s, (int, float))
            or (self.watch_s < 0)
        ):
            raise ConfigError("watch_s must be a non-negative number")
        if self.max_refreshes is not None and (
            type(self.max_refreshes) is not int or self.max_refreshes < 1
        ):
            raise ConfigError("max_refreshes must be a positive integer or None")
        for name in ("grouped", "agent_labels"):
            mapping = getattr(self, name)
            if not isinstance(mapping, dict) or not all(
                isinstance(key, str) and key and isinstance(value, str) and value.strip()
                for key, value in mapping.items()
            ):
                raise ConfigError(f"{name} must map nonempty strings to nonempty strings")
        if any("/" in component for component in self.grouped):
            raise ConfigError("grouped keys are grade component names and cannot contain '/'")
        if SYSTEM_AGENT in self.agent_labels:
            raise ConfigError(f"agent_labels cannot relabel {SYSTEM_AGENT} (always 'average')")
        if type(self.smooth_steps) is not int or self.smooth_steps < 1:
            raise ConfigError("smooth_steps must be a positive integer")
        if self.serve_port is not None and (
            type(self.serve_port) is not int or not 0 <= self.serve_port <= 65535
        ):
            raise ConfigError("serve_port must be an integer in [0, 65535] or None")
        if not isinstance(self.serve_host, str) or not self.serve_host.strip():
            raise ConfigError("serve_host must be a nonempty address")


@register_handle
@dataclass(frozen=True)
class Dashboard(Handle):
    KIND: ClassVar[str] = "dashboard"
    MANIFEST: ClassVar[str] = "dashboard.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("html", "standalone", "snapshot")

    html: str
    standalone: str
    snapshot: str
    n_runs: int
    n_attention: int
    refreshes: int
    generated_at: str | None
    url: str | None = None  # where the page was served, without the access key

    def summary(self) -> dict[str, Any]:
        return {
            "runs": self.n_runs,
            "attention": self.n_attention,
            "html": str(self.file("html")),
            "standalone": str(self.file("standalone")),
        }


def should_resume(handle: Dashboard, cfg: DashboardConfig) -> bool:
    """Every invocation refreshes: a dashboard is never done."""
    return True


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("version") == 1 else {}


def _annotations(path: str | None) -> Any:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"annotations_error": f"{type(exc).__name__} reading {path}"}


async def dashboard(cfg: DashboardConfig, run: RunDir) -> Dashboard:
    cache_path = run.path("cache.json")
    cache = _load_cache(cache_path)
    cache["version"] = 1
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[int] = []
    if cfg.watch_s > 0 or cfg.serve_port is not None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, stop.set)
                installed.append(sig)
    serving = cfg.serve_port is not None
    interval = float(cfg.watch_s if cfg.watch_s > 0 else cfg.refresh_s) if serving else cfg.watch_s
    server: DashboardServer | None = None
    serve_file = run.path("serve.json")
    refreshes = 0
    snapshot: dict[str, Any] = {}
    try:
        while True:
            snapshot = await asyncio.to_thread(
                collect,
                cfg.roots,
                cache=cache,
                cache_dir=run.out,
                stale_s=float(cfg.stale_s),
                probe_servers=cfg.probe_servers,
                gpus=cfg.gpus,
                annotations=_annotations(cfg.annotations),
                title=cfg.title,
                refresh_s=float(cfg.refresh_s),
                exclude=(run.out,),
                grouped=cfg.grouped,
                agent_labels=cfg.agent_labels,
                smooth_steps=cfg.smooth_steps,
            )
            refreshes += 1
            snapshot["refresh"] = refreshes
            text = json.dumps(snapshot, sort_keys=True, indent=1, allow_nan=False) + "\n"
            atomic_write_text(run.path("snapshot.json"), text)
            atomic_write_text(run.path("index.html"), render(snapshot))
            atomic_write_text(run.path("standalone.html"), render(snapshot, standalone=True))
            atomic_write_text(cache_path, json.dumps(cache, separators=(",", ":")) + "\n")
            if serving and server is None:
                server = _start_server(run.out, cfg, serve_file)
            if interval <= 0 or (cfg.max_refreshes and refreshes >= cfg.max_refreshes):
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval)
            if stop.is_set():
                break
    finally:
        # Close before removing the handlers, so a repeated signal cannot interrupt the cleanup.
        if server is not None:
            server.close()
            serve_file.unlink(missing_ok=True)
        for sig in installed:
            loop.remove_signal_handler(sig)
    return Dashboard(
        root=run.out,
        html="index.html",
        standalone="standalone.html",
        snapshot="snapshot.json",
        n_runs=len(snapshot.get("runs", [])),
        n_attention=len(snapshot.get("attention", [])),
        refreshes=refreshes,
        generated_at=snapshot.get("generated_at"),
        url=server.url(with_key=False) if server is not None else None,
    )


def _start_server(directory: Path, cfg: DashboardConfig, serve_file: Path) -> DashboardServer:
    host = cfg.serve_host.strip()
    assert cfg.serve_port is not None
    server = DashboardServer(directory, host, cfg.serve_port, access_key(host))
    server.start()
    record = {"url": server.url(), "host": host, "port": server.port, "pid": os.getpid()}
    fd = os.open(serve_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1)
        stream.write("\n")
    where = "" if is_loopback(host) else " (from another machine: this host's address or proxy)"
    log.warning("dashboard live at %s%s", server.url(), where)
    return server
