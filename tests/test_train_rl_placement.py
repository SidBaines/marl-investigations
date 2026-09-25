"""Learner placement is runtime-only, validated, and its timings reach metrics rows."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_train_loop import FakeSetup, config, make_taskset
from test_train_loop import fake_setup as fake_setup  # noqa: F401

from marli.config import config_hash
from marli.errors import ConfigError
from marli.train.backends.fake import FakeBackend
from marli.train.loop import local_backend_kwargs
from marli.train.rl import TrainRLConfig
from marli.verbs import run_verb


def test_placement_is_runtime_and_validated() -> None:
    cfg = TrainRLConfig()
    placed = replace(cfg, local_devices=["cuda:0", "cuda:1"], local_sleep_sampler=True)
    assert config_hash(cfg) == config_hash(placed)
    with pytest.raises(ConfigError, match="duplicate"):
        replace(cfg, local_devices=["cuda:0", "cuda:0"])
    with pytest.raises(ConfigError, match="local_sleep_sampler"):
        replace(cfg, local_sleep_sampler="yes")


def test_local_backend_kwargs_only_add_placement_when_set(tmp_path: Path) -> None:
    run = SimpleNamespace(path=lambda rel: tmp_path / rel)
    cfg = TrainRLConfig(local_server_json="server.json")
    assert local_backend_kwargs(cfg, run) == {
        "server_json": "server.json",
        "adapters_dir": str(tmp_path / "adapters"),
        "state_dir": str(tmp_path / "states"),
    }
    kwargs = local_backend_kwargs(
        replace(cfg, local_devices=["cuda:0", "cuda:1"], local_sleep_sampler=True), run
    )
    assert kwargs["devices"] == ["cuda:0", "cuda:1"] and kwargs["sleep_sampler"] is True


async def test_backend_step_metrics_reach_each_metrics_row(
    tmp_path: Path, fake_setup: FakeSetup, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    monkeypatch.setattr(
        FakeBackend, "pop_step_metrics", lambda self: {"sleep_s": 0.5}, raising=False
    )
    await run_verb("train rl", config(make_taskset(tmp_path / "tasks")), out=tmp_path / "train")
    lines = (tmp_path / "train/metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows and all(row["backend_metrics"] == {"fake": {"sleep_s": 0.5}} for row in rows)
