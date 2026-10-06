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
from marli.config import compose
from marli.dashboard import collect as collect_module
from marli.dashboard.collect import collect, fold_jsonl, trailing_mean
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
    [
        {"roots": []},
        {"stale_s": 0},
        {"watch_s": -1},
        {"max_refreshes": 0},
        {"refresh_s": True},
        {"grouped": ["score"]},
        {"grouped": {"score": ""}},
        {"grouped": {"": "Score"}},
        {"grouped": {"a/score": "Score"}},
        {"grouped": {"score": 1}},
        {"agent_labels": {"contrib0": " "}},
        {"agent_labels": {"_system": "team"}},
        {"smooth_steps": 0},
        {"smooth_steps": 2.5},
        {"smooth_steps": True},
        {"split_by": "a/b", "grouped": {"score": "Score"}},
        {"split_by": "", "grouped": {"score": "Score"}},
        {"split_by": "knew"},  # nothing grouped to split
        {"split_by": "knew", "grouped": {"score": "Score"}, "split_labels": ["only one"]},
        {"split_by": "knew", "grouped": {"score": "Score"}, "split_labels": ["yes", " "]},
        {"split_agents": [""]},
        {"serve_port": -1},
        {"serve_port": 70000},
        {"serve_port": True},
        {"serve_port": 8.5},
        {"serve_host": ""},
        {"serve_host": " "},
    ],
)
def test_invalid_config(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        replace(DashboardConfig(), **overrides)


def test_render_escapes_script_breakout() -> None:
    snapshot = {"title": "<b>t</b>", "generated_ts": 0, "runs": [{"name": "</script><x>"}]}
    page = render(snapshot)
    assert page.startswith("<title>&lt;b&gt;t&lt;/b&gt;</title>")
    assert "</script><x>" not in page


# ---------------------------------------------------------------- grouped per-agent charts

RELAY_SCORES = {"contrib0": [0, 1, 2, 3], "contrib2": [4, 0, 2, 6], "contrib10": [1, 1, 1, 1]}
RELAY_CREDIT = [
    {"zero_variance_groups": 1, "n_groups": 4},
    {"zero_variance_groups": 0, "n_groups": 0},  # no groups: no share
    {"zero_variance_groups": 2},  # n_groups missing: no share
    {"zero_variance_groups": 3, "n_groups": 3, "action_tokens": {"policy": 9}},
]


def relay_row(step: int) -> dict[str, Any]:
    """A metrics.jsonl row; contrib2 has no ``probed`` grade."""
    grades: dict[str, dict[str, float]] = {
        agent: {"score": scores[step], "n": 4} for agent, scores in RELAY_SCORES.items()
    }
    grades["contrib0"]["probed"] = step / 4
    grades["contrib10"]["probed"] = 1 - step / 4
    grades["_system"] = {
        "score": sum(scores[step] for scores in RELAY_SCORES.values()) / 3,
        "probed": 0.5,
        "n": 12,
    }
    return {
        "step": step,
        "accuracy": 0.5,
        "reward_mean": {"contributor": step / 10},
        "grades": grades,
        "credit": RELAY_CREDIT[step] if step < len(RELAY_CREDIT) else {},
        "sample_seconds": 10.0,
        "train_seconds": 5.0,
    }


def relay_run(tmp_path: Path, steps: int = 4, name: str = "relay") -> Path:
    path = tmp_path / "experiments" / "2026-01-01_relay" / "out" / name
    (path / ".marli").mkdir(parents=True)
    record = {"kind": "train rl", "manifest": "checkpoint.json", "config_hash": "c" * 64}
    (path / ".marli" / "run.json").write_text(json.dumps(record))
    (path / "config.yaml").write_text("steps: 10\nbatch_tasks: 2\ngroup_size: 2\n")
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(relay_row(i)) + "\n" for i in range(steps))
    )
    return path


GROUPED = {"score": "Score (average = team score)", "probed": "Chose review", "absent": "Nobody"}


def test_grouped_config_from_yaml_keeps_order(tmp_path: Path) -> None:
    path = tmp_path / "dash.yaml"
    path.write_text(
        "grouped:\n  submitted: Chose submit\n  probed: Chose review\n"
        "agent_labels:\n  contrib0: 1st\nsmooth_steps: 3\n"
    )
    cfg = compose(DashboardConfig, path, overrides=["grouped.ran_ci=Ran CI", "agent_labels.c1=2nd"])
    assert list(cfg.grouped) == ["submitted", "probed", "ran_ci"]
    assert cfg.agent_labels == {"contrib0": "1st", "c1": "2nd"} and cfg.smooth_steps == 3
    assert DashboardConfig().grouped == {} and DashboardConfig().smooth_steps == 5


async def test_grouped_charts_one_series_per_agent_then_average(tmp_path: Path) -> None:
    relay_run(tmp_path)
    cfg = dash(tmp_path, grouped=GROUPED, agent_labels={"contrib0": "1st"}, smooth_steps=3)
    await run_verb("dashboard", cfg, out=tmp_path / "dash")
    snapshot = json.loads((tmp_path / "dash" / "snapshot.json").read_text())
    assert snapshot["smooth_steps"] == 3
    groups = by_name(snapshot)["relay"]["train"]["groups"]
    # Mapping order, not alphabetical; a component nobody has is left out.
    assert [(g["component"], g["title"]) for g in groups] == [
        ("score", "Score (average = team score)"),
        ("probed", "Chose review"),
    ]
    score, probed = groups
    assert [(s["agent"], s["label"], s["slot"]) for s in score["series"]] == [
        ("contrib0", "1st", 0),
        ("contrib2", "contrib2", 1),
        ("contrib10", "contrib10", 2),
        ("_system", "average", None),
    ]
    # contrib2 has no probed grade; contrib10 keeps its colour slot anyway.
    assert [(s["agent"], s["slot"]) for s in probed["series"]] == [
        ("contrib0", 0),
        ("contrib10", 2),
        ("_system", None),
    ]
    series = {s["agent"]: s for s in score["series"]}
    assert series["contrib0"]["points"] == [[0, 0], [1, 1], [2, 2], [3, 3]]
    assert [v for _, v in series["contrib0"]["smooth"]] == [0, 0.5, 1, 2]
    assert [v for _, v in series["contrib2"]["smooth"]] == pytest.approx([4, 2, 2, 8 / 3], abs=1e-6)
    assert [step for step, _ in series["contrib2"]["smooth"]] == [0, 1, 2, 3]
    assert [v for _, v in series["_system"]["points"]] == pytest.approx(
        [5 / 3, 2 / 3, 5 / 3, 10 / 3]
    )
    for name in ("snapshot.json", "standalone.html"):
        text = (tmp_path / "dash" / name).read_text()
        assert "Chose review" in text and "1st" in text and "average" in text


def test_trailing_mean_windows() -> None:
    points = [[0, 4.0], [1, 0.0], [2, 2.0], [5, 6.0]]
    assert trailing_mean(points, 1) == points
    assert trailing_mean(points, 3) == [[0, 4.0], [1, 2.0], [2, 2.0], [5, pytest.approx(8 / 3)]]
    assert trailing_mean(points, 10) == [[0, 4.0], [1, 2.0], [2, 2.0], [5, 3.0]]
    assert trailing_mean([], 3) == []


def test_smooth_window_one_is_the_raw_series(tmp_path: Path) -> None:
    relay_run(tmp_path)
    snapshot = collect(
        [str(tmp_path / "experiments")],
        probe_servers=False,
        gpus=False,
        grouped=GROUPED,
        smooth_steps=1,
    )
    for group in by_name(snapshot)["relay"]["train"]["groups"]:
        for series in group["series"]:
            assert series["smooth"] == series["points"]
    with pytest.raises(ValueError):
        collect([str(tmp_path / "experiments")], gpus=False, grouped=GROUPED, smooth_steps=0)


def test_no_signal_share_is_derived_and_listed_after_rewards(tmp_path: Path) -> None:
    relay_run(tmp_path)
    train = by_name(collect([str(tmp_path / "experiments")], probe_servers=False, gpus=False))[
        "relay"
    ]["train"]
    # Steps 1 (n_groups 0) and 2 (n_groups missing) have no share.
    assert train["curves"]["credit/no_signal_share"] == [[0, 0.25], [3, 1.0]]
    assert train["key_curves"][:3] == ["reward/contributor", "credit/no_signal_share", "accuracy"]
    assert "groups" not in train  # grouped is empty: the old structure


def split_episode(grades: dict[str, dict[str, Any]], ok: bool = True) -> str:
    row = {
        "ok": ok,
        "grades": {**grades, "_system": {"score": 9, "knew": 1}},
        "transcript": f"{PROMPT_SENTINEL} {ANSWER_SENTINEL}",  # text that must stay here
    }
    return json.dumps(row) + "\n"


SPLIT_GROUPED = {"knew": "Started knowing", "score": "Score (avg = team)", "probed": "Chose review"}


def split_rollouts(path: Path) -> Path:
    """Rollouts for completed steps 0 and 1 and the sampling step 9; returns step 1's file."""
    rollouts = path / "rollouts"
    for step in (0, 1, 9):
        (rollouts / f"step_{step:05d}").mkdir(parents=True)
    (rollouts / "step_00000" / "episodes.jsonl").write_text(
        split_episode(
            {
                "contrib0": {"knew": 0, "score": 0, "probed": 1},  # not in split_agents
                "contrib1": {"knew": 1, "score": 1, "probed": 0},
                "contrib2": {"knew": 0, "score": 0, "probed": 1},
            }
        )
        + split_episode({"contrib1": {"knew": 1, "score": 0, "probed": 0}}, ok=False)
    )
    step1 = rollouts / "step_00001" / "episodes.jsonl"
    step1.write_text(
        split_episode(
            {
                "contrib1": {"knew": 0, "score": 0, "probed": 0},
                "contrib2": {"knew": 1, "score": 1, "probed": 1},
            }
        )
        + split_episode(
            {
                "contrib1": {"knew": 1, "score": 0, "probed": 1},
                "contrib2": {"knew": 1, "score": 0, "probed": 1, "n": 4},
                # No split grade ("none" side); a text grade is not a number.
                "contrib3": {"score": 1, "probed": ANSWER_SENTINEL},
            }
        )
    )
    # A step still sampling never enters the cube.
    (rollouts / "step_00009" / "episodes.jsonl").write_text(
        split_episode({"contrib1": {"knew": 1, "score": 1, "probed": 1}})
    )
    return step1


def pool(
    cube: dict[str, Any], agents: list[str], sides: list[str], component: str, window: int
) -> tuple[list[list[float]], list[list[float]]]:
    """What the page draws from the cube: per-step means and the window's sum over sum."""
    rows = []
    for i, step in enumerate(cube["steps"]):
        total = n = 0
        for agent in agents:
            for side in sides:
                cell = cube["cells"].get(agent, {}).get(side)
                if cell and component in cell["sum"]:
                    total += cell["sum"][component][i]
                    n += cell.get("n_by", {}).get(component, cell["n"])[i]
        if n:
            rows.append((step, total, n))
    raw = [[step, total / n] for step, total, n in rows]
    smooth = []
    for i, (step, _, _) in enumerate(rows):
        recent = rows[max(0, i - window + 1) : i + 1]
        smooth.append([step, sum(r[1] for r in recent) / sum(r[2] for r in recent)])
    return raw, smooth


async def test_explore_cube_splits_each_agent_and_pools_sum_over_sum(tmp_path: Path) -> None:
    path = relay_run(tmp_path)  # metrics rows for steps 0-3: steps 0-3 are complete
    step1 = split_rollouts(path)
    cfg = dash(
        tmp_path,
        grouped=SPLIT_GROUPED,
        agent_labels={"contrib1": "2nd"},
        smooth_steps=2,
        split_by="knew",
        split_labels=["Knew", "Did not"],
        split_agents=["contrib1", "contrib2", "contrib9"],
    )
    await run_verb("dashboard", cfg, out=tmp_path / "dash")
    snapshot = json.loads((tmp_path / "dash" / "snapshot.json").read_text())
    train = by_name(snapshot)["relay"]["train"]
    # Split agents that have no turns are dropped; ids, not labels (the cube is keyed by id).
    assert train["split"] == {
        "by": "knew",
        "title": "Started knowing",
        "labels": ["Knew", "Did not"],
        "agents": ["contrib1", "contrib2"],
    }
    cube = train["explore"]
    assert cube["steps"] == [0, 1]  # step 9 is still sampling
    assert cube["components"] == ["knew", "score", "probed"]  # grouped order
    assert cube["titles"] == {"knew": "Started knowing", "score": "Score", "probed": "Chose review"}
    # One colour slot per agent over the grouped charts' agents (contrib10) and the cube's.
    assert cube["agents"] == [
        {"id": "contrib0", "label": "contrib0", "slot": 0},
        {"id": "contrib1", "label": "2nd", "slot": 1},
        {"id": "contrib2", "label": "contrib2", "slot": 2},
        {"id": "contrib3", "label": "contrib3", "slot": 3},
    ]
    slots = {s["agent"]: s["slot"] for g in train["groups"] for s in g["series"]}
    assert slots == {"contrib0": 0, "contrib2": 2, "contrib10": 4, "_system": None}
    # Every agent's ok turns, by its own side; _system and the failed episode are left out.
    cells = cube["cells"]
    assert cells["contrib0"] == {
        "0": {"n": [1, 0], "sum": {"knew": [0, 0], "score": [0, 0], "probed": [1, 0]}}
    }
    assert cells["contrib1"] == {
        "1": {"n": [1, 1], "sum": {"knew": [1, 1], "score": [1, 0], "probed": [0, 1]}},
        "0": {"n": [0, 1], "sum": {"knew": [0, 0], "score": [0, 0], "probed": [0, 0]}},
    }
    assert cells["contrib2"]["1"] == {
        "n": [0, 2],
        "sum": {"knew": [0, 2], "score": [0, 1], "probed": [0, 2]},
    }
    # No split grade: the "none" side; the text grade is not counted, so probed has no sums.
    assert cells["contrib3"] == {"none": {"n": [0, 1], "sum": {"score": [0, 1]}}}
    # Pooling the split agents reproduces sum over sum: (1 + 1) / (1 + 3), not mean(1, 1/3).
    split_agents = train["split"]["agents"]
    raw, smooth = pool(cube, split_agents, ["1"], "score", 2)
    assert raw == [[0, 1.0], [1, pytest.approx(1 / 3)]] and smooth == [[0, 1.0], [1, 0.5]]
    assert pool(cube, split_agents, ["1"], "probed", 2)[1] == [[0, 0.0], [1, 0.75]]
    assert pool(cube, split_agents, ["0"], "probed", 2)[0] == [[0, 1.0], [1, 0.0]]
    # "All rollouts" pools every side: contrib3's "none" turn counts there.
    assert pool(cube, ["contrib2", "contrib3"], ["1", "0", "none"], "score", 1)[0] == [
        [0, 0.0],
        [1, pytest.approx(2 / 3)],
    ]
    assert train["current_step"]["episodes_done"] == 1  # the main fold is untouched
    for name in ("snapshot.json", "index.html", "standalone.html", "cache.json"):
        text = (tmp_path / "dash" / name).read_text()
        assert PROMPT_SENTINEL not in text and ANSWER_SENTINEL not in text, name

    # Rows appended later are folded on the next pass from the cached offset.
    with step1.open("a") as stream:
        stream.write(split_episode({"contrib2": {"knew": 0, "score": 1, "probed": 1}}))
    await run_verb("dashboard", cfg, out=tmp_path / "dash")
    snapshot = json.loads((tmp_path / "dash" / "snapshot.json").read_text())
    cube = by_name(snapshot)["relay"]["train"]["explore"]
    assert cube["cells"]["contrib2"]["0"]["n"] == [1, 1]
    assert pool(cube, split_agents, ["0"], "score", 1)[0] == [[0, 0.0], [1, 0.5]]


def test_cube_counts_a_component_graded_on_fewer_turns() -> None:
    fold = collect_module._cube_fold(("score", "probed"), None)
    state: dict[str, Any] = {}
    rows = [
        {"grades": {"a": {"score": 0.25, "probed": 1}, "b": {"score": 1}}},
        {"grades": {"a": {"score": 0.5, "probed": True}, "_system": {"score": 3}}},  # bool: no
        {"ok": False, "grades": {"a": {"score": 1, "probed": 1}}},
        {"grades": {"a": "not a dict", "b": {"other": 2}}},
    ]
    for row in rows:
        fold(state, row)
    cube = collect_module.explore_cube([(4, state)], ["score", "probed"])
    assert cube["steps"] == [4] and cube["agents"] == ["a", "b"]
    # Without split_by every turn is on the "none" side; probed was graded on 1 of a's 2 turns.
    assert cube["cells"]["a"] == {
        "none": {"n": [2], "sum": {"score": [0.75], "probed": [1]}, "n_by": {"probed": [1]}}
    }
    assert cube["cells"]["b"] == {"none": {"n": [2], "sum": {"score": [1]}, "n_by": {"score": [1]}}}


def test_grouped_without_split_by_has_a_cube_but_no_split(tmp_path: Path) -> None:
    split_rollouts(relay_run(tmp_path))
    snapshot = collect(
        [str(tmp_path / "experiments")], probe_servers=False, gpus=False, grouped=SPLIT_GROUPED
    )
    train = by_name(snapshot)["relay"]["train"]
    assert "split" not in train
    cube = train["explore"]
    assert cube["components"] == ["knew", "score", "probed"] and cube["steps"] == [0, 1]
    assert {side for sides in cube["cells"].values() for side in sides} == {"none"}
    assert cube["cells"]["contrib1"]["none"]["n"] == [1, 2]


def test_no_signal_share_skipped_without_group_counts(tmp_path: Path) -> None:
    path = relay_run(tmp_path)
    rows = [relay_row(i) for i in range(3)]
    rows[0]["credit"] = {"zero_variance_groups": 2, "n_groups": 0}
    rows[1]["credit"] = {"zero_variance_groups": 2}
    rows[2].pop("credit")
    (path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    train = by_name(collect([str(tmp_path / "experiments")], probe_servers=False, gpus=False))[
        "relay"
    ]["train"]
    assert "credit/no_signal_share" not in train["curves"]
    assert "credit/no_signal_share" not in train["key_curves"]


def test_grouped_charts_follow_appended_rows_incrementally(tmp_path: Path) -> None:
    path = relay_run(tmp_path, steps=3)
    cache: dict[str, Any] = {}
    kwargs: dict[str, Any] = {"probe_servers": False, "gpus": False, "grouped": GROUPED}
    roots = [str(tmp_path / "experiments")]
    first = by_name(collect(roots, cache=cache, **kwargs))["relay"]["train"]
    assert len(first["groups"][0]["series"][0]["points"]) == 3
    with (path / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(relay_row(3)) + "\n")
    second = by_name(collect(roots, cache=cache, **kwargs))["relay"]["train"]
    entry = cache["files"][str((path / "metrics.jsonl").resolve())]
    assert entry["parsed"] == 1 and entry["rows"] == 4
    assert [len(s["points"]) for s in second["groups"][0]["series"]] == [4, 4, 4, 4]
    assert second["curves"]["credit/no_signal_share"] == [[0, 0.25], [3, 1.0]]
    # Derived values never leak into the cached fold state.
    assert all("credit/no_signal_share" not in p for p in entry["state"]["points"])


def test_render_draws_grouped_charts_and_raw_toggle(tmp_path: Path) -> None:
    relay_run(tmp_path)
    snapshot = collect(
        [str(tmp_path / "experiments")],
        probe_servers=False,
        gpus=False,
        grouped=GROUPED,
        agent_labels={"contrib0": "1st", "contrib2": "3rd"},
    )
    for page in (render(snapshot), render(snapshot, standalone=True)):
        embedded = json.loads(page.split('id="snapshot">', 1)[1].split("</script>", 1)[0])
        groups = by_name(embedded)["relay"]["train"]["groups"]
        assert [g["title"] for g in groups] == ["Score (average = team score)", "Chose review"]
        assert [s["label"] for s in groups[0]["series"]] == ["1st", "3rd", "contrib10", "average"]
        for text in ("Score (average = team score)", "Chose review", '"1st"', '"3rd"'):
            assert text in page
        assert "show raw per-step values" in page and "marli-dash-raw" in page
        assert "Share of groups with no learning signal (all playthroughs tied)" in page
        # Six series colours, each with light and both dark-mode definitions.
        for slot in range(1, 7):
            assert page.count(f"--series-{slot}:") == 3
    plain = collect([str(tmp_path / "experiments")], probe_servers=False, gpus=False)
    assert "groups" not in by_name(plain)["relay"]["train"]
    assert "explore" not in by_name(plain)["relay"]["train"]


def test_render_carries_the_explorer_and_its_cube(tmp_path: Path) -> None:
    split_rollouts(relay_run(tmp_path))
    snapshot = collect(
        [str(tmp_path / "experiments")],
        probe_servers=False,
        gpus=False,
        grouped=SPLIT_GROUPED,
        split_by="knew",
        split_labels=["Knew", "Did not"],
    )
    for page in (render(snapshot), render(snapshot, standalone=True)):
        embedded = json.loads(page.split('id="snapshot">', 1)[1].split("</script>", 1)[0])
        train = by_name(embedded)["relay"]["train"]
        assert train["explore"] == by_name(snapshot)["relay"]["train"]["explore"]
        assert train["split"]["labels"] == ["Knew", "Did not"]
        # The explorer's controls, persistence and the per-agent split charts.
        for text in (
            "function explorer(run)",
            "marli-dash-explore:",
            "Left axis (solid)",
            "Right axis (dashed)",
            "all rollouts",
            "Whole run, one curve (agent, average and filter choices do not apply)",
            "Average line",
            "function splitCharts(train)",
            "Agents in each group per step",
            ".explorer {",
            ".pair-grid {",
        ):
            assert text in page, text
        # Every browser-storage access is wrapped: the page must work without it.
        uses = [line for line in page.splitlines() if "localStorage" in line]
        assert len(uses) >= 6 and all("try {" in line and "catch (e)" in line for line in uses)
