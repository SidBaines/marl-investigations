"""The dashboard observes real run dirs, folds rows incrementally and never leaks text."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_data_filter import episode as filter_episode
from test_data_filter import save_episodes
from test_eval_rollout import rollout_config
from test_train_loop import FakeSetup, make_taskset
from test_train_loop import config as train_config
from test_train_loop import fake_setup as fake_setup

from marli.cli.main import main
from marli.dashboard import collect as collect_module
from marli.dashboard.collect import collect, fold_jsonl
from marli.dashboard.render import render
from marli.dashboard.verb import DashboardConfig
from marli.data.filter import FilterConfig
from marli.envs.base import Task
from marli.errors import ConfigError
from marli.policy.scripted import ScriptedPolicy, Turn, from_callable
from marli.render.fake import FakeRenderer
from marli.rundir import RunDir, RunStatus
from marli.serve.vllm import Server
from marli.tasks.taskset import TaskSet, write_tasks
from marli.verbs import run_verb

PROMPT_SENTINEL = "ZZSENTINEL-PROMPT-7f3a"
ANSWER_SENTINEL = "ZZSENTINEL-ANSWER-91c2"
__all__ = ["fake_setup"]


def sentinel_policy() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy(
        "sentinel",
        renderer,
        from_callable(
            lambda ctx: Turn(tool_calls=(("submit", {"answer": f"5 {ANSWER_SENTINEL}"}),)),
            renderer,
        ),
    )


def sentinel_taskset(root: Path, n: int = 3) -> TaskSet:
    tasks = [Task(f"t{i}", f"What is 2 + 3? {PROMPT_SENTINEL}", answer=5) for i in range(n)]
    write_tasks(root, tasks)
    handle = TaskSet(
        root=root,
        tasks="tasks.jsonl",
        source="arithmetic",
        split="test",
        kind="math",
        n=n,
        answer_format="integer",
        commit_text=False,
    )
    handle.save()
    return handle


def dash(tmp_path: Path, **overrides: Any) -> DashboardConfig:
    return replace(
        DashboardConfig(roots=[str(tmp_path / "experiments")], probe_servers=False, gpus=False),
        **overrides,
    )


def by_name(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {run["name"]: run for run in snapshot["runs"]}


@pytest.fixture
def lab(tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One study with a complete, a paused and a stopped rollout, a train run and a filter."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "experiments" / "2026-01-01_demo" / "out"
    tasks = sentinel_taskset(tmp_path / "tasks")
    policies = {
        "test": replace(
            rollout_config(tasks).policies["test"], ref="scripted:test_dashboard:sentinel_policy"
        )
    }
    cfg = rollout_config(tasks, policies=policies, episodes_per_task=2)
    import asyncio

    asyncio.run(run_verb("eval rollout", cfg, out=out / "complete"))
    asyncio.run(run_verb("eval rollout", replace(cfg, stop_after_tasks=1), out=out / "paused"))
    original = RunDir.append_row

    def crash(run: RunDir, rel: str, row: Any) -> None:
        if (
            rel == "episodes.jsonl"
            and (run.out / rel).exists()
            and (len((run.out / rel).read_text().splitlines()) >= 2)
        ):
            raise RuntimeError("simulated crash")
        original(run, rel, row)

    with monkeypatch.context() as patch:
        patch.setattr(RunDir, "append_row", crash)
        with pytest.raises(RuntimeError):
            asyncio.run(run_verb("eval rollout", replace(cfg, concurrency=1), out=out / "stopped"))
    asyncio.run(
        run_verb(
            "train rl", train_config(make_taskset(tmp_path / "train_tasks")), out=out / "train"
        )
    )
    ftasks = TaskSet.load(tasks.manifest_path)
    saved = save_episodes(
        tmp_path / "filter_eps",
        [filter_episode(f"t{i}", j, float(j % 2)) for i in range(3) for j in range(2)],
        ftasks,
    )
    asyncio.run(
        run_verb(
            "data filter",
            FilterConfig(tasks=str(tasks.root), episodes=str(saved), metric="correct"),
            out=out / "pool",
        )
    )
    return tmp_path


async def test_classification_progress_curves_and_safety(lab: Path) -> None:
    result = await run_verb("dashboard", dash(lab), out=lab / "dash")
    assert result.status is RunStatus.FRESH
    snapshot = json.loads((lab / "dash" / "snapshot.json").read_text())
    runs = by_name(snapshot)
    assert set(runs) == {"complete", "paused", "stopped", "train", "pool"}
    assert {run["study"] for run in runs.values()} == {"2026-01-01_demo"}
    complete, paused, stopped = runs["complete"], runs["paused"], runs["stopped"]
    assert complete["status"] == "complete"
    assert complete["progress"] == {"done": 6, "total": 6, "unit": "episodes", "failures": 0}
    assert complete["eval"]["grades"]["_system"]["correct"] == 0.0  # "5 <sentinel>" is wrong
    assert paused["status"] == "paused" and paused["progress"]["done"] == 2
    assert paused["progress"]["total"] == 6 and paused["eval"]["paused_at_tasks"] == 1
    assert stopped["status"] == "stopped" and stopped["lock_held"] is False
    assert stopped["progress"]["done"] == 2
    train = runs["train"]
    assert train["status"] == "complete" and train["progress"]["done"] == 2
    assert train["progress"]["total"] == 2
    curves = train["train"]["curves"]
    assert [point[0] for point in curves["accuracy"]] == [0, 1]
    assert curves["learner/a/kl_sample_train"] == [[0, 0.0], [1, 0.0]]
    assert "grades/_system/correct" in train["train"]["key_curves"]
    assert train["rate"]["unit"] == "steps/h" and train["train"]["mean_step_s"] > 0
    assert runs["pool"]["facts"]["n"] >= 0 and "counts" in runs["pool"]["facts"]
    kinds = {item["path"]: item["level"] for item in snapshot["attention"]}
    assert kinds[stopped["path"]] == "bad" and kinds[paused["path"]] == "info"
    # Benchmark text stays in the run dirs.
    assert PROMPT_SENTINEL in (lab / "tasks" / "tasks.jsonl").read_text()
    raw = (lab / "experiments/2026-01-01_demo/out/complete/episodes.jsonl").read_text()
    assert ANSWER_SENTINEL in raw
    for name in ("snapshot.json", "index.html", "standalone.html", "cache.json"):
        text = (lab / "dash" / name).read_text()
        assert PROMPT_SENTINEL not in text and ANSWER_SENTINEL not in text, name


async def test_refresh_reopens_and_render_contract(lab: Path) -> None:
    first = await run_verb("dashboard", dash(lab), out=lab / "dash")
    again = await run_verb(
        "dashboard", dash(lab, title="Other title", stale_s=60.0), out=lab / "dash"
    )
    assert again.status is RunStatus.RESUME and again.config_hash == first.config_hash
    fragment = (lab / "dash" / "index.html").read_text()
    standalone = (lab / "dash" / "standalone.html").read_text()
    assert fragment.startswith("<title>Other title</title>\n<style>")
    for tag in ("<!doctype", "<html", "<head", "<body"):
        assert tag not in fragment.lower()
    assert standalone.startswith("<!doctype html>") and "<body>" in standalone
    snapshot = json.loads((lab / "dash" / "snapshot.json").read_text())
    assert "</script" not in fragment.split('id="snapshot">', 1)[1].split("</script>", 1)[0]
    embedded = json.loads(fragment.split('id="snapshot">', 1)[1].split("</script>", 1)[0])
    assert embedded["runs"] == snapshot["runs"]
    for run in snapshot["runs"]:
        assert run["name"] in fragment


def test_fold_is_incremental_and_tolerates_torn_lines(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    rows = [{"ok": True, "grades": {"_system": {"score": i}}, "calls": []} for i in range(3)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cache: dict[str, Any] = {}
    fold = collect_module._episode_fold
    init = collect_module._episode_init
    entry = fold_jsonl(path, cache, init, fold)
    assert entry is not None and entry["parsed"] == 3 and entry["state"]["n"] == 3
    with path.open("a") as stream:
        stream.write(json.dumps(rows[0]) + "\n" + '{"ok": true, "gra')
    entry = fold_jsonl(path, cache, init, fold)
    assert entry["parsed"] == 1 and entry["state"]["n"] == 4  # torn tail left for later
    with path.open("a") as stream:
        stream.write('des": {"_system": {"score": 10}}}\nnot json\n')
    entry = fold_jsonl(path, cache, init, fold)
    assert entry["parsed"] == 1 and entry["bad"] == 1 and entry["state"]["n"] == 5
    assert entry["state"]["grades"]["_system"]["score"] == [13.0, 5]
    assert fold_jsonl(path, cache, init, fold)["parsed"] == 0
    # A rewrite (new inode, e.g. compaction or resume) starts over.
    replacement = tmp_path / "new.jsonl"
    replacement.write_text(json.dumps(rows[1]) + "\n")
    os.replace(replacement, path)
    entry = fold_jsonl(path, cache, init, fold)
    assert entry["parsed"] == 1 and entry["state"]["n"] == 1


def test_servers_are_probed_safely_and_flagged_when_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs"
    server_dir = root / "serve"
    with RunDir(server_dir, kind="serve vllm", manifest_name="server.json", config_hash="h") as run:
        run.finalize(
            Server(
                root=server_dir,
                base_url="http://127.0.0.1:1",
                models=["m"],
                pid=2**22 + 12345,
                log="server.log",
            )
        )
    user = root / "rollout"
    with RunDir(user, kind="eval rollout", manifest_name="episodes.json", config_hash="h") as run:
        (user / "config.yaml").write_text(
            f"policies:\n  q:\n    ref: vllm:@{server_dir / 'server.json'}#m\n"
        )
        run.write_progress({"done": 0, "total": 4, "failures": 0})
        snapshot = collect([str(root)], probe_servers=True, gpus=False)
    (server,) = snapshot["servers"]
    assert server["alive"] is False and server["up"] is False and server["metrics"] is None
    (rollout,) = snapshot["runs"]
    assert rollout["status"] == "running" and rollout["lock_held"] is True
    assert any(
        item["text"].startswith("server used by a running run") for item in snapshot["attention"]
    )


def test_watch_mode_stops_after_max_refreshes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    out = tmp_path / "dash"
    args = [
        "dashboard",
        "roots=[experiments]",
        "probe_servers=false",
        "gpus=false",
        "watch_s=0.05",
        "max_refreshes=3",
        "--out",
        str(out),
    ]
    assert main(args) == 0
    snapshot = json.loads((out / "snapshot.json").read_text())
    assert snapshot["refresh"] == 3 and snapshot["runs"] == []
    assert json.loads((out / "dashboard.json").read_text())["refreshes"] == 3


async def test_watch_mode_finalizes_on_sigint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "dash"
    task = asyncio.create_task(run_verb("dashboard", dash(tmp_path, watch_s=0.05), out=out))
    for _ in range(200):
        await asyncio.sleep(0.05)
        path = out / "snapshot.json"
        if path.exists() and json.loads(path.read_text())["refresh"] >= 2:
            break
    assert not task.done()
    os.kill(os.getpid(), signal.SIGINT)
    result = await asyncio.wait_for(task, timeout=10)
    assert result.handle.refreshes >= 2 and (out / "dashboard.json").is_file()


def test_cli_prints_one_json_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    main(["dashboard", "roots=[missing]", "probe_servers=false", "gpus=false", "--out", "d"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["ok"] is True and result["runs"] == 0 and result["kind"] == "dashboard"
    snapshot = json.loads((tmp_path / "d" / "snapshot.json").read_text())
    assert snapshot["runs"] == [] and snapshot["attention"] == []


@pytest.mark.parametrize(
    "overrides",
    [{"roots": []}, {"stale_s": 0}, {"watch_s": -1}, {"max_refreshes": 0}, {"refresh_s": True}],
)
def test_invalid_config(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        replace(DashboardConfig(), **overrides)


def test_render_escapes_script_breakout() -> None:
    snapshot = {"title": "<b>t</b>", "generated_ts": 0, "runs": [{"name": "</script><x>"}]}
    page = render(snapshot)
    assert page.startswith("<title>&lt;b&gt;t&lt;/b&gt;</title>")
    assert "</script><x>" not in page
