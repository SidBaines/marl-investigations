"""Readers turn a harness's own outputs into the common rows (see spec.py).

Upstream harness directories carry a ``harness.json`` sidecar written by the study's
``run.sh`` beside the harness outputs: ``{"repo", "commit", "patch_sha256", "settings",
...}``. Readers return it unchanged so the verb can check the pin and record what ran.
Readers never copy prompts, replies or transcripts into rows.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marli.errors import ConfigError
from marli.eval.external.spec import ExternalSuite
from marli.handles import sha256_file

HARNESS_SIDECAR = "harness.json"


@dataclass
class ReaderResult:
    rows: list[dict[str, Any]]
    harness: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)  # ingested file -> sha256


def row(
    sample: str,
    condition: str,
    metrics: dict[str, float | None],
    *,
    parse_failures: int = 0,
    error: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sample": sample,
        "condition": condition,
        "metrics": metrics,
        "parse_failures": parse_failures,
        "error": error,
        "detail": detail or {},
    }


def _sidecar(directory: Path) -> dict[str, Any]:
    path = directory / HARNESS_SIDECAR
    if not path.is_file():
        raise ConfigError(f"{directory}: missing {HARNESS_SIDECAR} (written by the study run.sh)")
    return json.loads(path.read_text(encoding="utf-8"))


def _outputs(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ConfigError(f"harness output not found: {path}")
    found = sorted(p for p in path.glob(pattern) if p.name != HARNESS_SIDECAR)
    if not found:
        raise ConfigError(f"{path}: no harness outputs matching {pattern!r}")
    return found


# --- Inspect ------------------------------------------------------------------------


def inspect_log(path: Path, suite: ExternalSuite) -> ReaderResult:
    """marli Inspect tasks: score value = metric value or "unparsed"; metadata names it.

    ``path`` is a cell's log directory (every ``*.eval`` log in it is read).
    """
    from inspect_ai.log import read_eval_log

    rows: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    harness: dict[str, Any] = {}
    for log_path in _outputs(path, "*.eval"):
        log = read_eval_log(str(log_path))
        files[str(log_path)] = sha256_file(log_path)
        if log.status != "success":
            raise ConfigError(f"{log_path}: Inspect log status {log.status!r}: {log.error}")
        harness = {
            "inspect_ai": log.eval.packages.get("inspect_ai"),
            "task": log.eval.task,
            "task_version": log.eval.task_version,
            "task_args": log.eval.task_args,
            "model": log.eval.model,
            "model_generate_config": log.eval.model_generate_config.model_dump(exclude_none=True),
            "epochs": log.eval.config.epochs,
            "dataset_samples": log.eval.dataset.samples,
        }
        for sample in log.samples or []:
            sample_id = f"{sample.id}#e{sample.epoch}" if sample.epoch > 1 else str(sample.id)
            if sample.error is not None or not sample.scores:
                metadata = sample.metadata or {}
                condition = str(metadata.get("condition") or metadata.get("game") or "unknown")
                message = sample.error.message if sample.error is not None else "no score"
                rows.append(row(sample_id, condition, {}, error=message))
                continue
            (score,) = sample.scores.values()
            meta = score.metadata or {}
            value = score.value if isinstance(score.value, (int, float)) else None
            rows.append(
                row(
                    sample_id,
                    str(meta["condition"]),
                    {str(meta["metric"]): None if value is None else float(value)},
                    parse_failures=int(value is None),
                    detail={"reasoning_in_content": bool(meta.get("reasoning_in_content"))},
                )
            )
    return ReaderResult(rows, harness, files)


# --- HiddenBench (Li, Naito & Shirado, ICML 2026) -------------------------------------


def _accuracy(votes: list[dict[str, Any]], answer: str) -> tuple[float, float]:
    """Upstream metrics.py: average rule and majority rule (> half correct)."""
    if not votes:
        return 0.0, 0.0
    correct = sum(vote.get("vote") == answer for vote in votes)
    return correct / len(votes), float(correct > len(votes) / 2)


def hiddenbench(path: Path, suite: ExternalSuite) -> ReaderResult:
    """Upstream result JSONs (``hiddenbench eval --output``), merged like its ``score``.

    Group rows reproduce upstream ``score_results`` per run (pre/post, average and
    majority rule). Seat rows give each seat's own vote correctness, so mixed-seat
    cells can be read per seat. A scenario that raised (the marli patch records it
    instead of aborting the evaluation) is an error row, excluded from the rates.
    """
    files = _outputs(path, "*.json")
    harness = _sidecar(path if path.is_dir() else path.parent)
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for result_path in files:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metadata.append(data.get("metadata", {}))
        for run in data.get("runs", []):
            profile = str(run.get("profile") or data.get("metadata", {}).get("profile"))
            sample = str(run["task_id"])
            if run.get("error"):
                rows.append(row(sample, profile, {}, error=str(run["error"])))
                continue
            answer = run["correct_answer"]
            pre_avg, pre_maj = _accuracy(run.get("initial_votes", []), answer)
            post_avg, post_maj = _accuracy(run.get("final_votes", []), answer)
            seats = run.get("seat_models") or []
            rows.append(
                row(
                    sample,
                    profile,
                    {
                        "pre_average": pre_avg,
                        "post_average": post_avg,
                        "pre_majority": pre_maj,
                        "post_majority": post_maj,
                    },
                    detail={"seed": run.get("seed"), "seat_models": seats},
                )
            )
            initial = {vote["agent"]: vote for vote in run.get("initial_votes", [])}
            final = {vote["agent"]: vote for vote in run.get("final_votes", [])}
            for index, agent in enumerate(sorted(final, key=lambda name: int(name.split()[-1]))):
                rows.append(
                    row(
                        sample,
                        f"{profile}/seat{index}",
                        {
                            "pre_correct": float(initial.get(agent, {}).get("vote") == answer),
                            "post_correct": float(final[agent].get("vote") == answer),
                        },
                        detail={"model": seats[index] if index < len(seats) else None},
                    )
                )
    requests = set()
    for meta in metadata:
        request = dict(meta.get("request_kwargs") or {})
        if meta.get("temperature") is not None:
            request["temperature"] = meta["temperature"]
        requests.add(json.dumps(request, sort_keys=True))
    if len(requests) > 1:
        raise ConfigError(f"{path}: result files were sampled with different settings")
    harness = {
        **harness,
        # What the patched harness says it sent: its --request-kwargs plus --temperature.
        "request": json.loads(requests.pop()) if requests else None,
        "seat_assignments": sorted(
            {
                tuple(item["detail"]["seat_models"])
                for item in rows
                if item["detail"].get("seat_models")
            }
        ),
        "rounds": sorted({meta.get("rounds") for meta in metadata}),
        "seeds": [meta.get("seed") for meta in metadata],
        "result_files": len(files),
    }
    return ReaderResult(rows, harness, {str(p): sha256_file(p) for p in files})


# --- FAIRGAME volunteer's dilemma (Buscemi et al.) ------------------------------------


def _strategies(value: str) -> list[str]:
    parsed = ast.literal_eval(value) if value else []
    if not isinstance(parsed, list):
        raise ConfigError(f"unexpected FAIRGAME strategies cell: {value[:80]!r}")
    return [str(item) for item in parsed]


def fairgame_volunteer(path: Path, suite: ExternalSuite) -> ReaderResult:
    """FAIRGAME results CSVs, one game per row, plus per-game parse-failure logs.

    The sidecar names the volunteer label (the template's strategy2) and the games the
    study launched (``games``); a launched game with no CSV row is an error row.
    Group rows: volunteering rate over all decisions, the share of played rounds in
    which someone volunteered, and round 1's volunteering rate (before any history).
    Seat rows: each seat's volunteering rates.
    """
    directory = path if path.is_dir() else path.parent
    harness = _sidecar(directory)
    volunteer = harness["labels"]["volunteer"]
    prefix = harness.get("model_prefix", "")  # e.g. litellm:hosted_vllm/
    files = _outputs(path, "*.csv")
    failures: Counter[str] = Counter()
    for log in directory.glob("*.parse_failures.jsonl"):
        game = log.name.removesuffix(".parse_failures.jsonl")
        failures[game] += sum(1 for line in log.read_text(encoding="utf-8").splitlines() if line)
    rows: list[dict[str, Any]] = []
    seen = set()
    for csv_path in files:
        game = csv_path.stem
        with csv_path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        if len(records) != 1:
            raise ConfigError(f"{csv_path}: expected one game per CSV, found {len(records)}")
        (record,) = records
        seen.add(game)
        n_agents = sum(1 for key in record if key.endswith("_strategies"))
        choices = [_strategies(record[f"agent{i + 1}_strategies"]) for i in range(n_agents)]
        played = int(record["played_rounds"])
        if any(len(choice) != played for choice in choices):
            raise ConfigError(f"{csv_path}: strategy lists do not match played_rounds")
        decisions = [[c == volunteer for c in choice] for choice in choices]
        rounds = list(zip(*decisions, strict=True))
        rows.append(
            row(
                game,
                "group",
                {
                    "volunteer_rate": sum(map(sum, decisions)) / (n_agents * played),
                    "safe_rate": sum(any(r) for r in rounds) / played,
                    "round1_volunteer_rate": sum(rounds[0]) / n_agents,
                },
                parse_failures=failures[game],
                detail={"played_rounds": played, "max_rounds": int(record["max_rounds"])},
            )
        )
        for index, seat in enumerate(decisions):
            rows.append(
                row(
                    game,
                    f"seat{index}",
                    {
                        "volunteer_rate": sum(seat) / played,
                        "round1_volunteer": float(seat[0]),
                    },
                    detail={"model": record[f"agent{index + 1}_llm"].removeprefix(prefix)},
                )
            )
    harness = {
        **harness,
        "seat_assignments": sorted(
            {
                tuple(
                    item["detail"]["model"]
                    for item in rows
                    if item["sample"] == game and item["condition"].startswith("seat")
                )
                for game in seen
            }
        ),
    }
    for game in harness.get("games", []):
        if game not in seen:
            rows.append(row(game, "group", {}, error="game produced no results (see its log)"))
    return ReaderResult(rows, harness, {str(p): sha256_file(p) for p in files})
