"""Collect a JSON-safe progress snapshot from run directories without disturbing them.

Observation only: nothing here takes a run's lock or writes inside a run dir.
Liveness comes from /proc/locks (whether some process holds the run's flock),
calibrated against a lock the collector holds itself, and falls back to file
freshness where locks are not visible. Episode files are folded incrementally:
a cache keyed by (path, inode) keeps the byte offset and running sums, so a
large ``episodes.jsonl`` is parsed only for rows appended since the last pass.

Only numbers, agent ids and run metadata enter the snapshot: prompts, tool
I/O, answers and error messages never leave a run dir (benchmark text must not
be republished).
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = 1
# Verbs whose run dirs are control records, not work worth listing.
HIDDEN_KINDS = frozenset({"serve status", "serve stop", "dashboard"})
# Subtrees that never hold run dirs of their own but can be huge.
PRUNE = frozenset(
    {
        ".marli",
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        "rollouts",
        "adapters",
        "states",
        "checkpoints",
        "hf",
        ".triton",
    }
)
MAX_DEPTH = 8
RECENT_ENDS = 256  # episode completion times kept per file for rate estimates
TRAIN_KEYS = ("loss", "grad_norm", "kl_sample_train", "n_datums", "n_tokens", "n_action_tokens")
SERVER_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:gpu_cache_usage_perc": "kv_usage",
}
SERVER_COUNTERS = {
    "vllm:generation_tokens_total": "gen_tokens_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:prefix_cache_queries_total": "prefix_queries_total",
    "vllm:prefix_cache_hits_total": "prefix_hits_total",
    "vllm:spec_decode_num_draft_tokens_total": "spec_draft_total",
    "vllm:spec_decode_num_accepted_tokens_total": "spec_accepted_total",
}
_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")


def _num(value: Any) -> float | None:
    """A finite float, or None for anything that is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# ---------------------------------------------------------------- locks


class LockView:
    """Answer "does some process hold this run's flock?" from /proc/locks.

    ``calibrate`` flocks a probe file on each filesystem it is given; if that
    lock is invisible (a FUSE or network filesystem), answers there are None.
    """

    def __init__(self) -> None:
        try:
            text = Path("/proc/locks").read_text()
        except OSError:
            text = None
        self._entries: set[tuple[str, int]] | None = None
        self._inodes: set[int] = set()
        self._trusted: dict[int, bool] = {}
        if text is not None:
            self._entries = set()
            self._load(text)

    def _load(self, text: str) -> None:
        assert self._entries is not None
        for line in text.splitlines():
            parts = line.split()
            if "FLOCK" not in parts or "->" in parts:
                continue  # "->" lines are waiters, not holders
            for part in parts:
                fields = part.split(":")
                if len(fields) == 3 and fields[2].isdigit():
                    self._entries.add((f"{fields[0]}:{fields[1]}".lower(), int(fields[2])))
                    self._inodes.add(int(fields[2]))
                    break

    def calibrate(self, directory: Path) -> None:
        """Hold a probe flock in ``directory`` and check that /proc/locks shows it."""
        if self._entries is None:
            return
        try:
            dev = directory.stat().st_dev
        except OSError:
            return
        if dev in self._trusted:
            return
        try:
            fd, name = tempfile.mkstemp(prefix=".marli-lockprobe-", dir=directory)
        except OSError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            text = Path("/proc/locks").read_text()
            probe = LockView.__new__(LockView)
            probe._entries, probe._inodes, probe._trusted = set(), set(), {}
            probe._load(text)
            self._trusted[dev] = probe._match(os.fstat(fd)) is True
        except OSError:
            self._trusted[dev] = False
        finally:
            os.close(fd)
            os.unlink(name)

    def _match(self, info: os.stat_result) -> bool:
        assert self._entries is not None
        key = (f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}", info.st_ino)
        # Some filesystems report a different device in /proc/locks than stat().
        return key in self._entries or info.st_ino in self._inodes

    def held(self, lock: Path) -> bool | None:
        if self._entries is None:
            return None
        try:
            info = lock.stat()
        except OSError:
            return None
        if self._trusted.get(info.st_dev) is False:
            return None
        return self._match(info)


# ---------------------------------------------------------------- incremental jsonl folds


def fold_jsonl(
    path: Path,
    cache: dict[str, Any],
    init: Callable[[], dict[str, Any]],
    fold: Callable[[dict[str, Any], dict[str, Any]], None],
) -> dict[str, Any] | None:
    """Fold complete rows appended since the cached offset; restart on a new inode.

    The returned entry carries ``state`` plus ``rows`` (all folded rows),
    ``parsed`` (rows folded in this call) and ``bad`` (unparseable complete rows).
    A torn final line is left for the next pass.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    key = str(path)
    entry = cache.get(key)
    if (
        not isinstance(entry, dict)
        or entry.get("ino") != info.st_ino
        or info.st_size < entry.get("offset", 0)
    ):
        entry = {"ino": info.st_ino, "offset": 0, "rows": 0, "bad": 0, "state": init()}
    entry["parsed"] = 0
    if info.st_size > entry["offset"]:
        try:
            with path.open("rb") as stream:
                stream.seek(entry["offset"])
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    entry["offset"] += len(line)
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        entry["bad"] += 1
                        continue
                    if isinstance(row, dict):
                        fold(entry["state"], row)
                        entry["rows"] += 1
                        entry["parsed"] += 1
        except OSError:
            pass
    entry["mtime"] = info.st_mtime
    cache[key] = entry
    return entry


def _episode_init() -> dict[str, Any]:
    return {"n": 0, "ok": 0, "gen_tokens": 0, "calls": 0, "grades": {}, "roles": {}, "ends": []}


def _episode_fold(state: dict[str, Any], row: dict[str, Any]) -> None:
    """Numbers only: grades, ok, call counts/tokens and completion time."""
    state["n"] += 1
    ok = bool(row.get("ok", True))
    state["ok"] += ok
    ends = []
    for call in row.get("calls") or ():
        if not isinstance(call, dict):
            continue
        state["calls"] += 1
        usage = call.get("usage") or {}
        state["gen_tokens"] += int(_num(usage.get("completion_tokens")) or 0)
        timing = call.get("timing") or {}
        start, latency = _num(timing.get("started_at")), _num(timing.get("latency_s"))
        if start:
            ends.append(start + (latency or 0.0))
    if ends:
        state["ends"] = [*state["ends"], max(ends)][-RECENT_ENDS:]
    for agent in row.get("agents") or ():
        if isinstance(agent, dict) and isinstance(agent.get("agent_id"), str):
            role = agent.get("role")
            state["roles"][agent["agent_id"]] = role if isinstance(role, str) else None
    if not ok:
        return
    for agent, components in (row.get("grades") or {}).items():
        if not isinstance(agent, str) or not isinstance(components, dict):
            continue
        sums = state["grades"].setdefault(agent, {})
        for name, value in components.items():
            number = _num(value)
            if isinstance(name, str) and number is not None:
                total = sums.setdefault(name, [0.0, 0])
                total[0] += number
                total[1] += 1


def _grade_means(state: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        agent: {name: round(s / n, 4) for name, (s, n) in sorted(components.items()) if n}
        for agent, components in sorted(state["grades"].items())
    }


def _rate_per_min(ends: list[float], now: float, window_s: float = 1800.0) -> float | None:
    recent = sorted(t for t in ends if now - t <= window_s)
    if len(recent) < 2 or recent[-1] - recent[0] <= 0:
        return None
    return (len(recent) - 1) / (recent[-1] - recent[0]) * 60.0


def _metrics_init() -> dict[str, Any]:
    return {"points": []}


def _metrics_fold(state: dict[str, Any], row: dict[str, Any]) -> None:
    step = _num(row.get("step"))
    if step is None:
        return
    point: dict[str, float] = {"step": step}

    def put(key: str, value: Any) -> None:
        number = _num(value)
        if number is not None:
            point[key] = number

    for key in (
        "accuracy",
        "total_gen",
        "calls",
        "cp_tokens",
        "peak_ctx",
        "n_agents",
        "sample_seconds",
        "train_seconds",
    ):
        put(key, row.get(key))
    if "sample_seconds" in point or "train_seconds" in point:
        point["step_seconds"] = point.get("sample_seconds", 0.0) + point.get("train_seconds", 0.0)
    for role, value in (row.get("reward_mean") or {}).items():
        put(f"reward/{role}", value)
    for agent, components in (row.get("grades") or {}).items():
        if isinstance(components, dict):
            for name, value in components.items():
                put(f"grades/{agent}/{name}", value)
    for learner, record in (row.get("learners") or {}).items():
        if isinstance(record, dict):
            for name in TRAIN_KEYS:
                put(f"learner/{learner}/{name}", record.get(name))
            for name, value in (record.get("metrics") or {}).items():
                put(f"learner/{learner}/{name}", value)
    for name, value in (row.get("credit") or {}).items():
        put(f"credit/{name}", value)
    put("spent_usd", (row.get("spend") or {}).get("spent_usd"))
    # A resume rewrites metrics.jsonl (new inode) so a step never repeats here.
    state["points"] = [*[p for p in state["points"] if p["step"] != step], point]


# ---------------------------------------------------------------- per-kind summaries


def _short_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= 120)


def _config_facts(config: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Selected scalar settings (never prompts or free text beyond short names)."""
    facts: dict[str, Any] = {}
    for key in keys:
        value: Any = config
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if _short_scalar(value) or (
            isinstance(value, dict) and all(_short_scalar(v) for v in value.values())
        ):
            facts[key] = value
    return facts


def _eval_rollout(run: dict[str, Any], path: Path, cache: dict[str, Any], now: float) -> None:
    config = _read_yaml(path / "config.yaml")
    manifest = run["manifest"] or {}
    progress = _read_json(path / "progress.json") or {}
    entry = fold_jsonl(path / "episodes.jsonl", cache, _episode_init, _episode_fold)
    state = entry["state"] if entry else _episode_init()
    meta = manifest.get("meta") or {}
    per_task = (
        config.get("episodes_per_task") if isinstance(config.get("episodes_per_task"), int) else 1
    )
    total = _num(progress.get("total"))
    if meta.get("n_tasks") is not None:
        total = float(meta["n_tasks"] * per_task)  # the whole task set, not only a pause point
    elif manifest:
        total = float(manifest.get("n", state["n"]))
    # progress.json counts distinct episode ids (retries append duplicate rows).
    done = _num(progress.get("done"))
    done = float(state["n"]) if done is None else done
    failures = _num(progress.get("failures"))
    failures = float(state["n"] - state["ok"]) if failures is None else failures
    if meta.get("paused"):
        run["status"] = "paused"
    run["progress"] = {"done": done, "total": total, "unit": "episodes", "failures": failures}
    rate = _rate_per_min(state["ends"], now)
    run["rate"] = {"value": rate, "unit": "ep/min"} if rate else None
    if run["status"] == "running" and rate and total and done is not None:
        stop = config.get("stop_after_tasks")
        target = min(total, stop * per_task) if isinstance(stop, int) else total
        run["eta_s"] = max(0.0, (target - done) / rate * 60.0)
    run["eval"] = {
        "grades": _grade_means(state),
        "roles": state["roles"],
        "gen_tokens": state["gen_tokens"],
        "calls": state["calls"],
        "gen_tokens_per_episode": round(state["gen_tokens"] / state["n"]) if state["n"] else None,
        "unparsed_rows": entry["bad"] if entry else 0,
        "paused_at_tasks": meta.get("n_tasks_sampled") if meta.get("paused") else None,
    }
    run["facts"] = _config_facts(
        config,
        ("protocol", "env", "episodes_per_task", "max_tasks", "stop_after_tasks", "concurrency"),
    )
    policies = config.get("policies")
    if isinstance(policies, dict):
        run["facts"]["policies"] = {
            name: spec.get("ref")
            for name, spec in policies.items()
            if isinstance(spec, dict) and isinstance(spec.get("ref"), str)
        }
    run["_refs"] = _server_refs(config)


def _train_rl(run: dict[str, Any], path: Path, cache: dict[str, Any], now: float) -> None:
    config = _read_yaml(path / "config.yaml")
    progress = _read_json(path / "progress.json") or {}
    steps = (
        config.get("steps") if isinstance(config.get("steps"), int) else _num(progress.get("steps"))
    )
    entry = fold_jsonl(path / "metrics.jsonl", cache, _metrics_init, _metrics_fold)
    points = sorted(entry["state"]["points"] if entry else [], key=lambda p: p["step"])
    completed = int(points[-1]["step"]) if points else int(_num(progress.get("step")) or -1)
    per_step = None
    if isinstance(config.get("batch_tasks"), int) and isinstance(config.get("group_size"), int):
        per_step = config["batch_tasks"] * config["group_size"]
    # Per-step grade curves from rollouts when metrics.jsonl has none (older layouts).
    step_files = sorted((path / "rollouts").glob("step_*/episodes.jsonl"))
    rollout_points = []
    current = None
    for file in step_files:
        try:
            step = int(file.parent.name.removeprefix("step_"))
        except ValueError:
            continue
        folded = fold_jsonl(file, cache, _episode_init, _episode_fold)
        if folded is None:
            continue
        state = folded["state"]
        if step > completed:
            rate = _rate_per_min(state["ends"], now)
            current = {
                "step": step,
                "episodes_done": state["n"],
                "episodes_total": per_step,
                "failures": state["n"] - state["ok"],
                "grades": _grade_means(state).get("_system", {}),
                "rate": rate,
            }
        elif not points:
            point: dict[str, float] = {"step": float(step)}
            for agent, components in _grade_means(state).items():
                for name, value in components.items():
                    point[f"grades/{agent}/{name}"] = value
            rollout_points.append(point)
    points = points or rollout_points
    curves: dict[str, list[list[float]]] = {}
    for point in points:
        for key, value in point.items():
            if key != "step":
                curves.setdefault(key, []).append([point["step"], value])
    run["progress"] = {
        "done": completed + 1,
        "total": steps,
        "unit": "steps",
        "failures": None,
    }
    durations = [p["step_seconds"] for p in points if "step_seconds" in p][-5:]
    mean_step = sum(durations) / len(durations) if durations else None
    run["rate"] = {"value": 3600.0 / mean_step, "unit": "steps/h"} if mean_step else None
    if run["status"] == "running" and mean_step and steps:
        remaining = steps - (completed + 1)
        elapsed = now - (entry["mtime"] if entry else now)
        run["eta_s"] = max(0.0, remaining * mean_step - min(elapsed, mean_step))
    run["train"] = {
        "curves": curves,
        "key_curves": _key_curves(curves),
        "latest": {k: v[-1][1] for k, v in curves.items()},
        "current_step": current,
        "mean_step_s": mean_step,
    }
    run["facts"] = _config_facts(
        config,
        (
            "run_name",
            "protocol",
            "env",
            "steps",
            "batch_tasks",
            "group_size",
            "checkpoint_every",
            "credit.reward_key",
            "credit.reward_target",
            "credit.baseline",
            "seating",
        ),
    )
    learners = config.get("learners")
    if isinstance(learners, dict):
        run["facts"]["learners"] = {
            name: " ".join(
                str(spec.get(k))
                for k in ("base_model", "backend", "rank", "learning_rate")
                if spec.get(k) is not None
            )
            for name, spec in learners.items()
            if isinstance(spec, dict)
        }
    spend = progress.get("spend") or {}
    if _num(spend.get("spent_usd")) is not None:
        run["facts"]["spent_usd"] = spend["spent_usd"]
    run["_refs"] = _server_refs(config)


def _key_curves(curves: dict[str, Any]) -> list[str]:
    """The small multiples shown first: rewards, team grades, off-policy gap, timing."""
    order: list[str] = []
    order += sorted(k for k in curves if k.startswith("reward/"))
    order += [k for k in ("accuracy",) if k in curves]
    order += sorted(k for k in curves if k.startswith("grades/_system/") and not k.endswith("/n"))
    order += sorted(
        k for k in curves if k.startswith("learner/") and k.endswith("/kl_sample_train")
    )
    order += sorted(k for k in curves if k.startswith("learner/") and k.endswith("/grad_norm"))
    order += [
        k
        for k in (
            "step_seconds",
            "sample_seconds",
            "train_seconds",
            "total_gen",
            "peak_ctx",
            "spent_usd",
        )
        if k in curves
    ]
    return order


def _data_run(run: dict[str, Any], path: Path) -> None:
    manifest = run["manifest"] or {}
    meta = manifest.get("meta") or {}
    facts = {
        k: manifest[k]
        for k in ("n", "n_failed", "source", "split", "answer_format", "kind")
        if isinstance(manifest.get(k), (int, float, str)) and len(str(manifest[k])) <= 80
    }
    # Counts sit at meta.counts (build, repos) or one level down (meta.filter.counts).
    for scope in (meta, *(v for v in meta.values() if isinstance(v, dict))):
        counts = scope.get("counts")
        if isinstance(counts, dict) and "counts" not in facts:
            facts["counts"] = {k: v for k, v in counts.items() if _num(v) is not None}
        for key in ("lo", "hi", "metric", "inclusive", "n_repos", "n_per_repo"):
            if isinstance(scope.get(key), (int, float, str)) and key not in facts:
                facts[key] = scope[key]
    run["facts"] = facts
    n = _num(manifest.get("n"))
    if n is not None:
        run["progress"] = {"done": n, "total": n, "unit": "items", "failures": None}


def _server_refs(config: Any) -> list[str]:
    """server.json paths a run's config points at (vllm:@... refs, local_server_json)."""
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and "server.json" in value:
            match = re.search(r"(/[^\s#@]*server\.json)", value)
            if match:
                found.append(str(Path(match.group(1)).resolve()))

    walk(config)
    return sorted(set(found))


# ---------------------------------------------------------------- servers and GPUs


def _pid_alive(pid: Any) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _probe_metrics(base_url: str, timeout_s: float = 2.0) -> dict[str, float]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base_url.rstrip("/") + "/metrics", timeout=timeout_s) as response:
        text = response.read(8_000_000).decode("utf-8", errors="replace")
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name, value = match.group(1), float(match.group(3))
        target = SERVER_GAUGES.get(name) or SERVER_COUNTERS.get(name)
        if target and math.isfinite(value):
            values[target] = values.get(target, 0.0) + value
    return values


def _server(
    path: Path, manifest: dict[str, Any], cache: dict[str, Any], probe: bool, now: float
) -> dict[str, Any]:
    base_url = manifest.get("base_url") if isinstance(manifest.get("base_url"), str) else None
    adapters = [a.get("name") for a in manifest.get("adapters") or () if isinstance(a, dict)]
    server: dict[str, Any] = {
        "base_url": base_url,
        "models": [m for m in manifest.get("models") or () if isinstance(m, str)],
        "adapters": [a for a in adapters if isinstance(a, str)][-6:],
        "n_adapters": len(adapters),
        "pid": manifest.get("pid"),
        "alive": _pid_alive(manifest.get("pid")),
        "metrics": None,
    }
    if probe and base_url and server["alive"] is not False:
        try:
            values = _probe_metrics(base_url)
        except (OSError, ValueError) as exc:
            server["probe_error"] = type(exc).__name__
        else:
            previous = (cache.get("probes") or {}).get(base_url)
            if isinstance(previous, dict) and 0 < now - previous.get("t", 0) < 3600:
                dt = now - previous["t"]
                for key in ("gen_tokens_total", "prompt_tokens_total"):
                    delta = values.get(key, 0.0) - previous.get(key, 0.0)
                    if delta >= 0:
                        values[key.replace("_total", "_per_s")] = round(delta / dt, 1)
            cache.setdefault("probes", {})[base_url] = {"t": now, **values}
            if values.get("spec_draft_total"):
                values["spec_accept_rate"] = round(
                    values.get("spec_accepted_total", 0.0) / values["spec_draft_total"], 3
                )
            if values.get("prefix_queries_total"):
                values["prefix_hit_rate"] = round(
                    values.get("prefix_hits_total", 0.0) / values["prefix_queries_total"], 3
                )
            server["metrics"] = values
            server["alive"] = True if server["alive"] is None else server["alive"]
    server["up"] = bool(server["metrics"]) or server["alive"] is True
    return server


def _gpus() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_used_mib": float(parts[2]),
                    "mem_total_mib": float(parts[3]),
                    "util": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return gpus


# ---------------------------------------------------------------- discovery and assembly


def find_runs(roots: Iterable[Path]) -> list[Path]:
    """Directories holding ``.marli/run.json``, pruning heavy subtrees."""
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        base = len(root.parts)
        for directory, subdirs, _ in os.walk(root, followlinks=False):
            current = Path(directory)
            if (current / ".marli" / "run.json").is_file():
                found.append(current)
            if len(current.parts) - base >= MAX_DEPTH:
                subdirs[:] = []
            else:
                subdirs[:] = sorted(d for d in subdirs if d not in PRUNE)
    return sorted(set(found))


def _display(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _study(path: Path) -> str:
    parts = path.parts
    if "experiments" in parts:
        index = len(parts) - 1 - parts[::-1].index("experiments")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name or "runs"


def _last_activity(path: Path, kind: str) -> float | None:
    candidates = [path / "progress.json", path / ".marli" / "run.json"]
    candidates += [path / name for name in ("episodes.jsonl", "metrics.jsonl", "server.log")]
    if kind == "train rl":
        steps = sorted((path / "rollouts").glob("step_*"))
        if steps:
            candidates.append(steps[-1] / "episodes.jsonl")
    times = [t for t in (_mtime(p) for p in candidates) if t is not None]
    return max(times) if times else None


def collect(
    roots: Iterable[str | Path],
    *,
    cache: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
    stale_s: float = 180.0,
    probe_servers: bool = True,
    gpus: bool = True,
    annotations: Any = None,
    title: str = "marli runs",
    refresh_s: float = 60.0,
    exclude: Iterable[Path] = (),
) -> dict[str, Any]:
    """Snapshot every run under ``roots``; ``cache`` is updated in place."""
    cache = cache if cache is not None else {}
    files = cache.setdefault("files", {})
    now = time.time()
    cwd = Path.cwd()
    root_paths = [Path(root).resolve() for root in roots]
    locks = LockView()
    if cache_dir is not None:
        locks.calibrate(cache_dir)
    excluded = {Path(p).resolve() for p in exclude}
    runs: list[dict[str, Any]] = []
    servers: list[dict[str, Any]] = []
    for path in find_runs(root_paths):
        if path in excluded:
            continue
        record = _read_json(path / ".marli" / "run.json") or {}
        kind = record.get("kind") if isinstance(record.get("kind"), str) else "unknown"
        if kind in HIDDEN_KINDS:
            continue
        manifest_name = record.get("manifest") if isinstance(record.get("manifest"), str) else None
        manifest = _read_json(path / manifest_name) if manifest_name else None
        manifest = manifest if isinstance(manifest, dict) else None
        last = _last_activity(path, kind)
        held = locks.held(path / ".marli" / "lock")
        idle = now - last if last is not None else None
        if manifest is not None:
            status = "complete"
        elif held is True:
            # A live owner: a learner step writes nothing for many minutes, so only a
            # much longer silence suggests a hung trainer.
            limit = stale_s * (10 if kind == "train rl" else 1)
            status = "stalled" if idle is not None and idle > limit else "running"
        elif held is False:
            status = "stopped"
        else:
            status = "running" if idle is not None and idle <= stale_s else "stalled"
        run: dict[str, Any] = {
            "path": _display(path, cwd),
            "study": _study(path),
            "name": path.name,
            "kind": kind,
            "status": status,
            "lock_held": held,
            "config_hash": (record.get("config_hash") or "")[:12] or None,
            "started_at": record.get("started_at"),
            "updated_at": _iso(last),
            "updated_ts": last,
            "idle_s": round(idle, 1) if idle is not None else None,
            "manifest": manifest,
            "progress": None,
            "rate": None,
            "eta_s": None,
            "facts": {},
        }
        if kind == "serve vllm":
            if manifest is not None:
                server = _server(path, manifest, cache, probe_servers, now)
                server.update(
                    path=run["path"],
                    study=run["study"],
                    name=path.name,
                    manifest_path=str((path / manifest_name).resolve()),
                )
                servers.append(server)
            continue
        try:
            if kind == "eval rollout":
                _eval_rollout(run, path, files, now)
            elif kind == "train rl":
                _train_rl(run, path, files, now)
            elif manifest is not None:
                _data_run(run, path)
        except Exception as exc:  # one odd run dir must not blank the whole page
            run["collect_error"] = type(exc).__name__
        run.pop("manifest", None)
        runs.append(run)
    # Drop cache entries for files that no longer exist.
    for key in [k for k in files if not os.path.exists(k)]:
        del files[key]
    gpu_rows = _gpus() if gpus else []
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "title": title,
        "generated_at": _iso(now),
        "generated_ts": now,
        "host": socket.gethostname(),
        "cwd": str(cwd),
        "roots": [_display(root, cwd) for root in root_paths],
        "refresh_s": refresh_s,
        "stale_s": stale_s,
        "gpus": gpu_rows,
        "servers": servers,
        "runs": runs,
        "annotations": annotations,
    }
    snapshot["attention"] = _attention(snapshot)
    for run in runs:
        run.pop("_refs", None)
    snapshot["counts"] = _counts(runs)
    return sanitize(snapshot)


def sanitize(value: Any) -> Any:
    """Strict JSON: non-finite floats become None; tuples become lists."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def _counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    return counts


def _attention(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """What needs a human: quiet or stopped work, dead servers in use, idle GPUs."""
    items: list[dict[str, Any]] = []
    running = [r for r in snapshot["runs"] if r["status"] == "running"]
    for run in snapshot["runs"]:
        if run["status"] == "stalled":
            items.append(
                {
                    "level": "warn",
                    "path": run["path"],
                    "text": f"{run['kind']} quiet for {int(run['idle_s'] or 0) // 60} min",
                }
            )
        elif run["status"] == "stopped":
            items.append(
                {
                    "level": "bad",
                    "path": run["path"],
                    "text": f"{run['kind']} stopped before completing (resumable)",
                }
            )
        elif run["status"] == "paused":
            items.append(
                {
                    "level": "info",
                    "path": run["path"],
                    "text": "rollout paused; rerun without stop_after_tasks to finish",
                }
            )
        if run.get("collect_error"):
            items.append(
                {
                    "level": "warn",
                    "path": run["path"],
                    "text": f"could not summarise ({run['collect_error']})",
                }
            )
        failures = (run.get("progress") or {}).get("failures")
        if failures:
            items.append(
                {"level": "warn", "path": run["path"], "text": f"{int(failures)} failed episodes"}
            )
    in_use = {ref for run in running for ref in run.get("_refs") or ()}
    for server in snapshot["servers"]:
        if server["manifest_path"] in in_use and not server["up"]:
            items.append(
                {
                    "level": "bad",
                    "path": server["path"],
                    "text": "server used by a running run is down",
                }
            )
    gpus = snapshot["gpus"]
    if running and gpus and all(g["util"] == 0 for g in gpus):
        items.append(
            {
                "level": "warn",
                "path": None,
                "text": f"all {len(gpus)} GPUs idle while {len(running)} runs are running",
            }
        )
    order = {"bad": 0, "warn": 1, "info": 2}
    return sorted(items, key=lambda item: order[item["level"]])
