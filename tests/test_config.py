"""Configuration contracts inherited from scimt and used by later CLI tasks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, fields, make_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Literal

import pytest
from omegaconf import MISSING

from marli.config import (
    compose,
    config_hash,
    describe_config,
    doc_field,
    hashable_dict,
    input_field,
    parse,
    runtime_field,
    save,
    to_dict,
)
from marli.errors import (
    BackendError,
    BudgetExceededError,
    ConfigError,
    DirtyTreeError,
    HashMismatchError,
    MarliError,
)


class Optimizer(Enum):
    adam = "adam"
    sgd = "sgd"


@dataclass
class TrainConfig:
    lr: float = doc_field(0.0001, help="Learning rate.")
    steps: int = 10
    optimizer: Optimizer = Optimizer.adam
    workers: int = runtime_field(1, help="Concurrent workers.")
    tasks: str = input_field("tasks.json", help="Input task manifest.")

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("train.lr must be positive")


@dataclass
class Config:
    seed: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)
    label: str = "baseline"
    optional: str | None = None
    tags: list[str] = doc_field(default_factory=list, help="Study tags.")
    timeout: int = runtime_field(60)
    checkpoint: Path = input_field(Path("checkpoint.json"))


@dataclass
class RequiredConfig:
    name: str = doc_field(help="Run name.")
    task: str = MISSING


@dataclass
class NestedRequiredConfig:
    required: RequiredConfig = field(default_factory=lambda: RequiredConfig("???"))


@dataclass
class LiteralConfig:
    mode: Literal["single", "swarm"] = "single"


def test_error_exit_codes() -> None:
    classes = [
        MarliError,
        ConfigError,
        DirtyTreeError,
        HashMismatchError,
        BudgetExceededError,
        BackendError,
    ]
    assert [cls.exit_code for cls in classes] == [1, 2, 2, 3, 4, 5]
    assert all(issubclass(cls, MarliError) for cls in classes)
    assert issubclass(ConfigError, ValueError)


def test_compose_merge_order(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    first.write_text("seed: 1\nlabel: first\ntrain:\n  lr: 0.1\n  steps: 20\n")
    second = tmp_path / "second.yaml"
    second.write_text("seed: 2\nlabel: second\ntrain:\n  lr: 0.2\n")

    assert compose(Config).seed == 0
    assert compose(Config, first).seed == 1
    assert compose(Config, first, str(second)).seed == 2
    cfg = compose(Config, first, second, overrides=["seed=3", "train.lr=0.3"])
    assert cfg == Config(seed=3, label="second", train=TrainConfig(lr=0.3, steps=20))
    assert isinstance(cfg, Config)
    assert isinstance(cfg.train, TrainConfig)


@pytest.mark.parametrize("key", ["typo", "train.lrr"])
@pytest.mark.parametrize("source", ["yaml", "override"])
def test_unknown_keys_are_config_errors(tmp_path: Path, key: str, source: str) -> None:
    path = tmp_path / "unknown.yaml"
    path.write_text("train:\n  lrr: 1\n" if "." in key else "typo: 1\n")
    with pytest.raises(ConfigError) as caught:
        if source == "yaml":
            compose(Config, path)
        else:
            compose(Config, overrides=[f"{key}=1"])
    assert isinstance(caught.value, ValueError)
    assert key in str(caught.value)
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("source", ["yaml", "override"])
def test_type_validation_errors_name_key(tmp_path: Path, source: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("train:\n  lr: nope\n")
    with pytest.raises(ConfigError, match=r"train\.lr") as caught:
        if source == "yaml":
            compose(Config, path)
        else:
            compose(Config, overrides=["train.lr=nope"])
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize(
    ("cls", "overrides", "key"),
    [
        (RequiredConfig, [], "name"),
        (RequiredConfig, ["name=run"], "task"),
        (RequiredConfig, ["name=run", "task=???"], "task"),
        (NestedRequiredConfig, [], "required.name"),
    ],
)
def test_missing_mandatory_values(cls: type, overrides: list[str], key: str) -> None:
    with pytest.raises(ConfigError) as caught:
        compose(cls, overrides=overrides)
    assert key in str(caught.value)


def test_post_init_errors_are_wrapped() -> None:
    with pytest.raises(ConfigError, match=r"^train\.lr must be positive$") as caught:
        compose(Config, overrides=["train.lr=-1"])
    assert isinstance(caught.value.__cause__, ValueError)


def test_missing_yaml_stays_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        compose(Config, tmp_path / "absent.yaml")


@pytest.mark.parametrize("invalid", [dict, {}, Config()])
def test_compose_requires_dataclass_type(invalid: object) -> None:
    with pytest.raises(TypeError, match="dataclass type"):
        compose(invalid)


def test_enum_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "enum.yaml"
    path.write_text("train:\n  optimizer: sgd\n")
    assert compose(Config, path).train.optimizer is Optimizer.sgd


def test_literal_is_not_supported() -> None:
    # OmegaConf 2.3 rejects Literal; configs must use str/Enum plus validation.
    with pytest.raises(ConfigError):
        compose(LiteralConfig)


def test_save_round_trip(tmp_path: Path) -> None:
    cfg = Config(seed=42, train=TrainConfig(lr=0.2, optimizer=Optimizer.sgd), tags=["α"])
    path = tmp_path / "nested" / "config.yaml"
    assert save(cfg, str(path)) == path
    assert compose(Config, path) == cfg


@pytest.mark.parametrize("invalid", [Config, {}, 1])
def test_save_requires_dataclass_instance(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(TypeError, match="dataclass instance"):
        save(invalid, tmp_path / "bad.yaml")
    assert not (tmp_path / "bad.yaml").exists()


def test_parse_splits_args_and_defaults_to_sys_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 2\ntrain:\n  lr: 0.1\n")
    args = ["seed=3", str(path), "train.lr=0.4", "label=a=b"]
    expected = Config(seed=3, train=TrainConfig(lr=0.4), label="a=b")
    assert parse(Config, args) == expected
    monkeypatch.setattr(sys, "argv", ["marli", *args])
    assert parse(Config) == expected
    assert parse(Config, []) == Config()


def test_field_helpers() -> None:
    @dataclass
    class Fields:
        runtime: list[str] = runtime_field(default_factory=list, help="Runtime options.")
        input: str = input_field("manifest.json")
        documented: list[str] = doc_field(default_factory=list, help="Scientific options.")

    descriptors = fields(Fields)
    assert descriptors[0].metadata == {"runtime": True, "help": "Runtime options."}
    assert descriptors[1].metadata == {"input": True, "help": None}
    assert descriptors[2].metadata == {"help": "Scientific options."}
    first, second = Fields(), Fields()
    first.runtime.append("a")
    first.documented.append("b")
    assert second == Fields([], "manifest.json", [])
    with pytest.raises(ValueError):
        runtime_field([], default_factory=list)
    with pytest.raises(ValueError):
        doc_field([], default_factory=list, help="Invalid defaults.")


def test_plain_and_hashable_dicts_recurse_through_containers() -> None:
    @dataclass
    class ContainerConfig:
        single: TrainConfig = field(default_factory=TrainConfig)
        items: list[TrainConfig] = field(default_factory=lambda: [TrainConfig()])
        pair: tuple[Path, Optimizer] = (Path("relative/file"), Optimizer.sgd)
        mapping: dict[str, TrainConfig] = field(default_factory=lambda: {"nested": TrainConfig()})
        timeout: int = runtime_field(10)
        manifest: Path = input_field(Path("input.json"))

    cfg = ContainerConfig()
    full_train = {
        "lr": 0.0001,
        "steps": 10,
        "optimizer": "adam",
        "workers": 1,
        "tasks": "tasks.json",
    }
    hashed_train = {"lr": 0.0001, "steps": 10, "optimizer": "adam"}
    assert to_dict(cfg) == {
        "single": full_train,
        "items": [full_train],
        "pair": ["relative/file", "sgd"],
        "mapping": {"nested": full_train},
        "timeout": 10,
        "manifest": "input.json",
    }
    assert hashable_dict(cfg) == {
        "single": hashed_train,
        "items": [hashed_train],
        "pair": ["relative/file", "sgd"],
        "mapping": {"nested": hashed_train},
    }
    assert json.loads(json.dumps(to_dict(cfg))) == to_dict(cfg)
    assert cfg.single.workers == 1
    assert cfg.manifest == Path("input.json")


@pytest.mark.parametrize("invalid", [Config, {}])
def test_dict_helpers_require_instances(invalid: object) -> None:
    for convert in (to_dict, hashable_dict):
        with pytest.raises(TypeError, match="dataclass instance"):
            convert(invalid)


def test_hash_stability_and_scientific_sensitivity() -> None:
    cfg = Config()
    digest = config_hash(cfg)
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    assert digest == config_hash(Config())
    assert digest != config_hash(replace(cfg, seed=1))
    assert digest != config_hash(replace(cfg, train=TrainConfig(lr=0.2)))
    assert digest == config_hash(
        replace(
            cfg,
            timeout=100,
            checkpoint=Path("elsewhere.json"),
            train=TrainConfig(workers=8, tasks="moved/tasks.json"),
        )
    )
    assert config_hash(cfg, input_digests=["aaa", "bbb"]) == config_hash(
        cfg, input_digests=iter(["bbb", "aaa"])
    )
    assert config_hash(cfg, input_digests=["aaa"]) != config_hash(cfg, input_digests=["bbb"])


def test_hash_uses_canonical_json_and_ignores_field_order() -> None:
    first = make_dataclass("First", [("z", int, 2), ("a", str, "α")])
    second = make_dataclass("Second", [("a", str, "α"), ("z", int, 2)])
    expected = hashlib.sha256('{"config":{"a":"α","z":2},"inputs":["a","b"]}'.encode()).hexdigest()
    assert config_hash(first(), input_digests=["b", "a"]) == expected
    assert config_hash(second(), input_digests=["a", "b"]) == expected


def test_hash_is_stable_across_processes() -> None:
    code = (
        "from dataclasses import make_dataclass; "
        "from marli.config import config_hash; "
        "cls = make_dataclass('Config', [('z', int, 2), ('a', str, 'α')]); "
        "print(config_hash(cls()))"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    outputs = [
        subprocess.run(
            [sys.executable, "-c", code],
            env={**env, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2")
    ]
    cls = make_dataclass("LocalConfig", [("a", str, "α"), ("z", int, 2)])
    assert outputs == [config_hash(cls()), config_hash(cls())]


def test_describe_nested_config() -> None:
    schema = describe_config(Config)
    assert [row["name"] for row in schema] == [
        "seed",
        "train.lr",
        "train.steps",
        "train.optimizer",
        "train.workers",
        "train.tasks",
        "label",
        "optional",
        "tags",
        "timeout",
        "checkpoint",
    ]
    rows = {row["name"]: row for row in schema}
    assert rows["train.lr"] == {
        "name": "train.lr",
        "type": "float",
        "default": 0.0001,
        "required": False,
        "runtime": False,
        "input": False,
        "help": "Learning rate.",
    }
    assert rows["seed"]["type"] == "int"
    assert rows["optional"]["type"] == "str | None"
    assert rows["optional"]["default"] is None
    assert rows["optional"]["required"] is False
    assert rows["tags"]["type"] == "list[str]"
    assert rows["tags"]["default"] == []
    assert rows["train.optimizer"]["type"] == "Optimizer"
    assert rows["train.optimizer"]["default"] == "adam"
    assert rows["train.workers"]["runtime"] is True
    assert rows["train.tasks"]["input"] is True
    assert rows["train.tasks"]["help"] == "Input task manifest."
    assert rows["checkpoint"]["default"] == "checkpoint.json"
    assert rows["timeout"]["help"] is None
    json.dumps(schema)


def test_describe_required_fields() -> None:
    schema = describe_config(RequiredConfig)
    assert [row["name"] for row in schema] == ["name", "task"]
    assert all(row["required"] and row["default"] is None for row in schema)
    nested = describe_config(NestedRequiredConfig)
    assert [row["name"] for row in nested] == ["required.name", "required.task"]
    assert all(row["required"] and row["default"] is None for row in nested)


def test_describe_uses_nested_factory_defaults_and_parent_metadata() -> None:
    @dataclass
    class CustomDefaults:
        train: TrainConfig = runtime_field(default_factory=lambda: TrainConfig(lr=0.5))

    schema = describe_config(CustomDefaults)
    assert schema[0]["default"] == 0.5
    assert all(row["runtime"] for row in schema)


@pytest.mark.parametrize("invalid", [dict, Config()])
def test_describe_requires_dataclass_type(invalid: object) -> None:
    with pytest.raises(TypeError, match="dataclass type"):
        describe_config(invalid)
