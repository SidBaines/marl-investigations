"""`marli dashboard`: observe run dirs and write snapshot.json + two HTML renderings.

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
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from marli.config import runtime_field
from marli.dashboard.collect import SYSTEM_AGENT, collect
from marli.dashboard.render import render
from marli.errors import ConfigError
from marli.handles import Handle, atomic_write_text, register_handle
from marli.rundir import RunDir


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
    if cfg.watch_s > 0:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, stop.set)
                installed.append(sig)
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
            if cfg.watch_s <= 0 or (cfg.max_refreshes and refreshes >= cfg.max_refreshes):
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=float(cfg.watch_s))
            if stop.is_set():
                break
    finally:
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
    )
