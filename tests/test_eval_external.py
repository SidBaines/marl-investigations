"""External evals: registry, decision parsing, settings, readers, statistics and the verb.

Upstream harness outputs here are synthetic (upstream schemas, invented tasks): no
benchmark text enters the repository. Inspect runs use Inspect's offline mock model or
the fake OpenAI-compatible server in ``_fake_openai.py``; nothing touches the network.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _fake_openai import FakeOpenAIServer, scripted_reply

from marli.errors import BackendError, ConfigError, MarliError
from marli.eval.external.parsing import parse_amount, parse_option, strip_reasoning
from marli.eval.external.readers import fairgame_volunteer, hiddenbench, row
from marli.eval.external.report import summarize
from marli.eval.external.serving import request_fields, resolve_endpoint, resolve_settings
from marli.eval.external.spec import SUITES, ExternalSuite
from marli.eval.external.verb import (
    ExternalCell,
    ExternalEval,
    ExternalEvalConfig,
    focal_rows,
    should_resume,
)
from marli.rundir import RunStatus
from marli.verbs import run_verb

ROOT = Path(__file__).resolve().parents[1]
HB_COMMIT = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
FG_COMMIT = "fc302a642c6f7cc0c439c2ae957a45f5954f4525"


# --- registry ------------------------------------------------------------------------


def test_registry_entries_load_and_are_listed() -> None:
    suites = SUITES.load_all()
    assert set(suites) == {"fairgame_volunteer", "hiddenbench", "li_shirado_games", "planted_help"}
    assert suites["hiddenbench"].source["commit"] == HB_COMMIT
    assert suites["fairgame_volunteer"].source["commit"] == FG_COMMIT
    assert suites["li_shirado_games"].task_args["trials"] == 100
    from marli.registry import catalog_names

    assert catalog_names("external_evals") == sorted(suites)


def _suite(**overrides: Any) -> ExternalSuite:
    fields: dict[str, Any] = {
        "name": "t",
        "kind": "upstream",
        "description": "d",
        "citation": "c",
        "license": "l",
        "source": {"repo": "r", "commit": "a" * 40},
        "reader": "m:f",
        "metrics": [{"name": "x", "kind": "binary", "pair": "none", "label": "x"}],
    }
    return ExternalSuite(**{**fields, **overrides})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "docker"}, "kind"),
        ({"source": {"repo": "r", "commit": "abc"}}, "40-character"),
        ({"generation": {"temprature": 1.0}}, "unknown generation keys"),
        ({"generation": {"chat_template_kwargs": {}}}, "thinking"),
        ({"thinking": "maybe"}, "thinking"),
        ({"metrics": [{"name": "x", "kind": "count", "pair": "none"}]}, "metric kind"),
        ({"kind": "inspect", "source": {"package": "p", "version": "1"}}, "task"),
    ],
)
def test_suite_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _suite(**overrides)


# --- decision parsing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Option A", "A"),
        ("**B**", "B"),
        ("I choose **Option B**.\n\nReasoning: choosing Option A risks exploitation.", "B"),
        ("# Option A\n\nI give 100 points (Option A). Option B would keep them.", "A"),
        ("Option B: Do nothing.\n\nOption A costs 6 points.", "B"),
        ("If I choose Option A I may lose.\nFinal answer: **Option B**", "B"),
        ("While Option B is the safer choice, I prefer cooperation: I choose Option A.", "A"),
        ("**Answer**: [Option A]", "A"),
        ("Thus, **Option B** is the more rational choice.", "B"),
        ("Both Option A and Option B have merits.", None),
        ("What would you choose?", None),
        ("", None),
    ],
)
def test_parse_option(text: str, expected: str | None) -> None:
    assert parse_option(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("50", 50),
        ("I give **50 points** to my partner.", 50),
        ("I'll decide to give 40 points to my partner.\nI'll keep 60 points.", 40),
        ("My minimum acceptable offer is **30**. Offers below 30 are rejected.", 30),
        ("**Final Answer:**\n\\boxed{20}", 20),
        ("I would give 120 points.", None),
        ("Giving 0 or 50 are both defensible; it depends on your values.", None),
    ],
)
def test_parse_amount(text: str, expected: int | None) -> None:
    assert parse_amount(text) == expected


def test_strip_reasoning_flags_leaked_thinking() -> None:
    assert strip_reasoning("weighing it up</think>\n\nOption B") == ("Option B", True)
    assert strip_reasoning("<think>still thinking") == ("", True)
    assert strip_reasoning("Option A") == ("Option A", False)


# --- settings ------------------------------------------------------------------------


def test_settings_precedence_and_thinking_switch() -> None:
    suite = _suite(generation={"temperature": 0.7})
    model = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"x": 1}}
    settings = resolve_settings(suite, model, {"max_tokens": 64})
    assert settings == {
        "temperature": 0.7,  # the benchmark's own setting wins over the model card
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 64,
        "chat_template_kwargs": {"x": 1, "enable_thinking": True},
    }
    assert request_fields(settings) == {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 64,
        "extra_body": {"top_k": 20, "chat_template_kwargs": {"x": 1, "enable_thinking": True}},
    }
    with pytest.raises(ConfigError, match="thinking"):
        resolve_settings(suite, {"chat_template_kwargs": {"enable_thinking": False}}, {})
    with pytest.raises(ValueError, match="unknown generation keys"):
        resolve_settings(suite, {"top_kk": 1}, {})


def test_endpoints_accept_vllm_refs_only(tmp_path: Path) -> None:
    server = tmp_path / "server.json"
    server.write_text(json.dumps({"base_url": "http://h:1/v1/", "models": ["m"], "hf_id": "x/y"}))
    endpoint = resolve_endpoint(f"vllm:@{server}#m")
    assert (endpoint.base_url, endpoint.model, endpoint.hf_id) == ("http://h:1", "m", "x/y")
    with pytest.raises(ConfigError, match="not listed"):
        resolve_endpoint(f"vllm:@{server}#other")
    with pytest.raises(ConfigError, match="vllm refs"):
        resolve_endpoint("api:openai/gpt-4o")
    assert resolve_endpoint("inspect:mockllm/model").provider == "mockllm"


# --- statistics ----------------------------------------------------------------------


def _metric_suite(kind: str, pair: str) -> ExternalSuite:
    return _suite(metrics=[{"name": "x", "kind": kind, "pair": pair, "label": "x"}])


def test_summary_counts_and_independent_gain() -> None:
    base = [row(f"t{i}", "pd", {"x": float(i < 5)}) for i in range(10)]
    base.append(row("t10", "pd", {"x": None}, parse_failures=1))
    base.append(row("t11", "pd", {}, error="boom"))
    cand = [row(f"t{i}", "pd", {"x": 1.0}) for i in range(10)]
    results = summarize({"base": base, "cand": cand}, _metric_suite("binary", "none"), "base")
    by_cell = {item["cell"]: item for item in results}
    assert by_cell["base"]["n"] == 10 and by_cell["base"]["missing"] == 1
    assert by_cell["base"]["parse_failures"] == 1 and by_cell["base"]["errors"] == 1
    assert by_cell["base"]["mean"] == 0.5 and by_cell["base"]["ci_method"] == "wilson"
    gain = by_cell["cand"]["gain"]
    assert gain["test"] == "permutation" and gain["difference"] == pytest.approx(0.5)
    assert gain["low"] > 0 and gain["p"] < 0.05


def test_summary_pairs_repeated_samples() -> None:
    base = [row(f"t{i}", "hidden", {"x": 0.25}) for i in range(8) for _ in range(2)]
    cand = [row(f"t{i}", "hidden", {"x": 0.75}) for i in range(8) for _ in range(2)]
    results = summarize({"base": base, "cand": cand}, _metric_suite("share", "sample"), "base")
    cand_result = next(item for item in results if item["cell"] == "cand")
    assert cand_result["ci_method"] == "sample_bootstrap" and cand_result["n_samples"] == 8
    gain = cand_result["gain"]
    assert gain["test"] == "sign_flip" and gain["n_pairs"] == 8
    assert gain["difference"] == pytest.approx(0.5)


def test_summary_mcnemar_for_single_binary_pairs() -> None:
    base = [row(f"t{i}", "c", {"x": 0.0}) for i in range(6)]
    cand = [row(f"t{i}", "c", {"x": 1.0}) for i in range(6)]
    gain = summarize({"b": base, "c": cand}, _metric_suite("binary", "sample"), "b")[1]["gain"]
    assert gain["test"] == "mcnemar" and gain["p"] == pytest.approx(2 / 64)


def test_focal_rows_regroup_mixed_and_homogeneous_cells() -> None:
    mixed = [
        row("t0", "hidden/seat0", {"x": 1.0}, detail={"model": "team"}),
        row("t0", "hidden/seat1", {"x": 0.0}, detail={"model": "base"}),
        row("t0", "hidden", {"x": 0.5}),
    ]
    grouped = focal_rows(mixed, "team")
    assert sorted(item["condition"] for item in grouped) == [
        "hidden",
        "hidden/focal",
        "hidden/others",
    ]
    homogeneous = [row("g", f"seat{i}", {"x": 1.0}, detail={"model": "base"}) for i in range(3)]
    conditions = [item["condition"] for item in focal_rows(homogeneous, "base")]
    assert conditions.count("focal") == 3 and conditions.count("others") == 3


# --- readers -------------------------------------------------------------------------


def _hb_run(task: int, votes: tuple[list[str], list[str]], seats: list[str]) -> dict[str, Any]:
    initial, final = votes
    return {
        "task_id": task,
        "scenario": f"synthetic_{task}",
        "profile": "hidden",
        "correct_answer": "X",
        "seed": task,
        "seat_models": seats,
        "initial_votes": [{"agent": f"Person {i + 1}", "vote": v} for i, v in enumerate(initial)],
        "final_votes": [{"agent": f"Person {i + 1}", "vote": v} for i, v in enumerate(final)],
    }


def _write_hiddenbench(
    directory: Path, runs: list[dict[str, Any]], request: dict[str, Any]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "benchmark": "HiddenBench",
        "profile": "hidden",
        "rounds": 15,
        "duplications": 1,
        "seed": 0,
        "seat_models": runs[0].get("seat_models"),
        "request_kwargs": {k: v for k, v in request.items() if k != "temperature"},
        "temperature": request.get("temperature"),
    }
    (directory / "hidden_s0.json").write_text(json.dumps({"metadata": metadata, "runs": runs}))
    (directory / "harness.json").write_text(
        json.dumps({"repo": "r", "commit": HB_COMMIT, "patch_sha256": "p", "request": request})
    )


def test_hiddenbench_reader_reproduces_upstream_scores(tmp_path: Path) -> None:
    seats = ["team", "base", "base", "base"]
    runs = [
        _hb_run(1, (["X", "Y", "Y", "Y"], ["X", "X", "X", "Y"]), seats),
        _hb_run(2, (["Y", "Y", "Y", "Y"], ["Y", "Y", "X", "Y"]), seats),
        {"task_id": 3, "scenario": "s3", "profile": "hidden", "seed": 3, "error": "ValueError: x"},
    ]
    _write_hiddenbench(tmp_path / "cell", runs, {"temperature": 1.0})
    result = hiddenbench(tmp_path / "cell", SUITES.load("hiddenbench"))
    groups = [item for item in result.rows if item["condition"] == "hidden"]
    assert groups[0]["metrics"] == {
        "pre_average": 0.25,
        "post_average": 0.75,
        "pre_majority": 0.0,
        "post_majority": 1.0,
    }
    assert groups[1]["metrics"]["post_average"] == 0.25
    assert groups[2]["error"] == "ValueError: x"
    seat0 = [item for item in result.rows if item["condition"] == "hidden/seat0"]
    assert seat0[0]["metrics"] == {"pre_correct": 1.0, "post_correct": 1.0}
    assert seat0[0]["detail"]["model"] == "team"
    assert result.harness["request"] == {"temperature": 1.0}
    assert result.harness["seat_assignments"] == [("team", "base", "base", "base")]


def _write_fairgame(directory: Path, games: dict[str, list[list[str]]], request: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for game, choices in games.items():
        record = {"game_id": game, "played_rounds": len(choices[0]), "max_rounds": 10}
        for index, strategy in enumerate(choices):
            record[f"agent{index + 1}_llm"] = f"litellm:hosted_vllm/m{index}"
            record[f"agent{index + 1}_strategies"] = repr(strategy)
        with (directory / f"{game}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
    (directory / "seed0.parse_failures.jsonl").write_text('{"agent": "agent1"}\n' * 2)
    sidecar = {
        "repo": "r",
        "commit": FG_COMMIT,
        "labels": {"do_nothing": "OptionA", "volunteer": "OptionB"},
        "games": [*games, "seed9"],
        "model_prefix": "litellm:hosted_vllm/",
        "request": request,
    }
    (directory / "harness.json").write_text(json.dumps(sidecar))


def test_fairgame_reader_rates_and_missing_games(tmp_path: Path) -> None:
    games = {
        "seed0": [["OptionB", "OptionA"], ["OptionA", "OptionA"], ["OptionA", "OptionA"]],
    }
    _write_fairgame(tmp_path / "cell", games, {})
    result = fairgame_volunteer(tmp_path / "cell", SUITES.load("fairgame_volunteer"))
    group = next(item for item in result.rows if item["condition"] == "group" and not item["error"])
    assert group["metrics"] == {
        "volunteer_rate": pytest.approx(1 / 6),
        "safe_rate": 0.5,
        "round1_volunteer_rate": pytest.approx(1 / 3),
    }
    assert group["parse_failures"] == 2
    seat0 = next(item for item in result.rows if item["condition"] == "seat0")
    assert seat0["metrics"] == {"volunteer_rate": 0.5, "round1_volunteer": 1.0}
    assert seat0["detail"]["model"] == "m0"
    assert any(item["error"] for item in result.rows if item["sample"] == "seed9")
    assert result.harness["seat_assignments"] == [("m0", "m1", "m2")]


# --- the verb: upstream suites ---------------------------------------------------------


async def test_verb_ingests_upstream_and_checks_the_pin(tmp_path: Path) -> None:
    request = {
        "temperature": 1.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    runs_base = [_hb_run(i, (["Y"] * 4, ["Y", "Y", "X", "Y"]), ["b"] * 4) for i in range(6)]
    runs_team = [_hb_run(i, (["Y"] * 4, ["X", "X", "X", "Y"]), ["t"] * 4) for i in range(6)]
    _write_hiddenbench(tmp_path / "base", runs_base, request)
    _write_hiddenbench(tmp_path / "team", runs_team, request)
    cfg = ExternalEvalConfig(
        suite="hiddenbench",
        baseline="base",
        model_generation={"temperature": 1.0},
        cells=[
            ExternalCell("base", results=str(tmp_path / "base"), seats=["b"] * 4, focal="b"),
            ExternalCell("team", results=str(tmp_path / "team"), seats=["t"] * 4, focal="t"),
        ],
    )
    result = await run_verb("eval external", cfg, out=tmp_path / "out")
    handle = result.handle
    assert isinstance(handle, ExternalEval) and handle.n_cells == 2
    results = [json.loads(line) for line in handle.file("results").read_text().splitlines()]
    post = next(
        item for item in results if item["cell"] == "team" and item["metric"] == "post_average"
    )
    assert post["mean"] == 0.75 and post["gain"]["difference"] == pytest.approx(0.5)
    assert {item["condition"] for item in results} == {"hidden", "hidden/focal", "hidden/others"}
    assert "post_average" in handle.file("markdown").read_text()
    assert handle.meta["cells"]["team"]["harness"]["commit"] == HB_COMMIT
    # Unchanged outputs: complete. Changed outputs: the run reopens and rereads them.
    assert (await run_verb("eval external", cfg, out=tmp_path / "out")).status is RunStatus.COMPLETE
    assert not should_resume(handle, cfg)
    _write_hiddenbench(tmp_path / "team", runs_team[:3], request)
    assert should_resume(ExternalEval.load(tmp_path / "out"), cfg)
    assert (await run_verb("eval external", cfg, out=tmp_path / "out")).status is RunStatus.RESUME


@pytest.mark.parametrize("problem", ["commit", "request", "seats"])
async def test_verb_rejects_harness_drift(tmp_path: Path, problem: str) -> None:
    request = {
        "temperature": 1.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }
    runs = [_hb_run(0, (["Y"] * 4, ["Y"] * 4), ["b"] * 4)]
    _write_hiddenbench(tmp_path / "base", runs, request)
    if problem == "commit":
        sidecar = json.loads((tmp_path / "base/harness.json").read_text())
        (tmp_path / "base/harness.json").write_text(json.dumps({**sidecar, "commit": "f" * 40}))
    generation = {"temperature": 0.5 if problem == "request" else 1.0}
    seats = ["t"] * 4 if problem == "seats" else ["b"] * 4
    cfg = ExternalEvalConfig(
        suite="hiddenbench",
        model_generation=generation,
        cells=[ExternalCell("base", results=str(tmp_path / "base"), seats=seats)],
    )
    with pytest.raises(
        ConfigError, match={"commit": "commit", "request": "sent", "seats": "seated"}[problem]
    ):
        await run_verb("eval external", cfg, out=tmp_path / "out")


def test_cell_validation() -> None:
    with pytest.raises(ConfigError, match="unique"):
        ExternalEvalConfig(suite="hiddenbench", cells=[ExternalCell("a"), ExternalCell("a")])
    with pytest.raises(ConfigError, match="baseline"):
        ExternalEvalConfig(suite="hiddenbench", cells=[ExternalCell("a")], baseline="b")
    with pytest.raises(ConfigError, match="label"):
        ExternalEvalConfig(suite="hiddenbench", cells=[ExternalCell("a/b")])


def test_verb_module_imports_no_harness() -> None:
    code = (
        "import sys, marli.eval.external.verb, marli.eval.external.readers;"
        "assert not {'inspect_ai', 'litellm', 'openai'} & set(sys.modules), sorted(sys.modules)"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# --- the verb: inspect suites (needs the [external] extra) ----------------------------


@pytest.fixture
def inspect_ai() -> Any:
    return pytest.importorskip("inspect_ai")


def test_game_prompts_are_the_papers(inspect_ai: Any) -> None:
    from marli.eval.external.tasks.econ_games import GAMES, SYSTEM_PROMPT, score_reply, trial_seed

    assert SYSTEM_PROMPT.startswith("You are participating in a study.")
    assert GAMES["prisoners_dilemma"].prompt.splitlines()[1].startswith("- Option A: Give 100")
    assert GAMES["dictator"].prompt.endswith(
        "How many points do you give to your partner? [0 to 100]"
    )
    assert score_reply("public_goods", "thinking</think>Option A")[0] == 1.0
    assert score_reply("dictator", "I give **25 points**.")[0] == 0.25
    assert score_reply("prisoners_dilemma", "no idea") == (
        None,
        {"decision": None, "reasoning_in_content": False},
    )
    assert trial_seed(0, "dictator", 1) != trial_seed(0, "dictator", 2)


async def test_verb_runs_inspect_against_a_fake_vllm_server(
    tmp_path: Path, inspect_ai: Any
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    models = {"base": scripted_reply(True), "selfish": scripted_reply(False)}
    with FakeOpenAIServer(models, roots={"selfish": str(adapter)}, parent="base") as fake:
        server = tmp_path / "server.json"
        server.write_text(json.dumps({"base_url": fake.base_url, "models": list(models)}))
        cfg = ExternalEvalConfig(
            suite="li_shirado_games",
            baseline="base",
            task_args={"trials": 4},
            model_generation={
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 256,
                "chat_template_kwargs": {"reasoning_effort": "medium"},
            },
            cells=[
                ExternalCell("base", policy=f"vllm:@{server}#base"),
                ExternalCell("selfish", policy=f"vllm:@{server}#selfish"),
            ],
        )
        result = await run_verb("eval external", cfg, out=tmp_path / "out")
        requests = list(fake.requests)
    handle = result.handle
    results = {
        (item["cell"], item["metric"]): item
        for item in map(json.loads, handle.file("results").read_text().splitlines())
    }
    assert results[("base", "cooperate")]["mean"] == 1.0
    assert results[("selfish", "cooperate")]["mean"] == 0.0
    assert results[("selfish", "give_share")]["gain"]["difference"] == pytest.approx(-0.5)
    assert len(requests) == 24 and {r["model"] for r in requests} == {"base", "selfish"}
    sent = requests[0]
    assert sent["top_k"] == 20 and sent["top_p"] == 0.95 and sent["max_tokens"] == 256
    assert sent["chat_template_kwargs"] == {"reasoning_effort": "medium", "enable_thinking": True}
    assert len({r["seed"] for r in requests if r["model"] == "base"}) == 12  # one seed per trial
    served = handle.meta["cells"]["selfish"]["served"]
    assert served["parent"] == "base" and "adapter_config.json" in served["adapter_sha256"]
    logs = list((tmp_path / "out/logs/base").glob("*.eval"))
    assert len(logs) == 1 and handle.meta["cells"]["base"]["harness"]["inspect_ai"]


async def test_verb_fails_before_sampling_an_unserved_adapter(
    tmp_path: Path, inspect_ai: Any
) -> None:
    with FakeOpenAIServer({"base": scripted_reply(True)}) as fake:
        server = tmp_path / "server.json"
        server.write_text(json.dumps({"base_url": fake.base_url, "models": ["base", "gone"]}))
        cfg = ExternalEvalConfig(
            suite="li_shirado_games", cells=[ExternalCell("x", policy=f"vllm:@{server}#gone")]
        )
        with pytest.raises(BackendError, match="not served"):
            await run_verb("eval external", cfg, out=tmp_path / "out")
        assert fake.requests == []


def test_cli_mock_run_prints_one_json_line(tmp_path: Path, inspect_ai: Any) -> None:
    config = tmp_path / "mock.yaml"
    config.write_text(
        "suite: li_shirado_games\nbaseline: a\ntask_args: {trials: 2}\n"
        "cells:\n  - {label: a, policy: 'inspect:mockllm/model'}\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "marli.cli.main",
            "eval",
            "external",
            str(config),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    (line,) = done.stdout.splitlines()
    summary = json.loads(line)
    assert summary["ok"] and summary["kind"] == "external_eval"
    # The mock model's default reply holds no decision: every trial is a counted parse failure.
    results = [json.loads(x) for x in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert all(item["n"] == 0 and item["missing"] == 2 for item in results)


def test_error_hierarchy_is_marli() -> None:
    assert issubclass(BackendError, MarliError)
