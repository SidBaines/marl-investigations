"""Registry entries must have unambiguous, reproducible names."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import marli.registry as registry_module
from marli.errors import ConfigError
from marli.registry import FnRegistry, Registry, catalog_kinds, catalog_names


@dataclass
class Model:
    name: str
    width: int
    tags: list[str] = field(default_factory=list)
    options: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")


def identity(value: int) -> int:
    return value


def test_yaml_registry_happy_path_and_sorted_names(tmp_path: Path) -> None:
    (tmp_path / "zeta.yaml").write_text("name: zeta\nwidth: 8\n")
    (tmp_path / "alpha.yaml").write_text(
        "name: alpha\nwidth: 4\ntags: [small, base]\noptions: {layers: 2}\n"
    )
    (tmp_path / "ignored.yml").write_text("width: 16\n")
    (tmp_path / "directory.yaml").mkdir()
    registry = Registry("models", str(tmp_path), Model)
    assert registry.kind == "models"
    assert registry.directory == tmp_path
    assert registry.names() == ["alpha", "zeta"]
    assert registry.path("alpha") == tmp_path / "alpha.yaml"
    assert registry.load("alpha") == Model("alpha", 4, ["small", "base"], {"layers": 2})
    assert registry.load_all() == {
        "alpha": Model("alpha", 4, ["small", "base"], {"layers": 2}),
        "zeta": Model("zeta", 8),
    }
    assert list(registry.load_all()) == ["alpha", "zeta"]


def test_missing_directory_has_no_names(tmp_path: Path) -> None:
    registry = Registry("models", tmp_path / "absent", Model)
    assert registry.names() == []
    assert registry.load_all() == {}


@pytest.mark.parametrize("invalid", [dict, Model("a", 1)])
def test_registry_requires_dataclass_type(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(TypeError, match="dataclass type"):
        Registry("models", tmp_path, invalid)


def test_registry_rejects_name_mismatch(tmp_path: Path) -> None:
    (tmp_path / "original.yaml").write_text("name: twin\nwidth: 8\n")
    with pytest.raises(ConfigError, match="original.*twin.*filename stem 'original'"):
        Registry("models", tmp_path, Model).load("original")


def test_registry_injects_name(tmp_path: Path) -> None:
    (tmp_path / "original.yaml").write_text("width: 8\n")
    assert Registry("models", tmp_path, Model).load("original") == Model("original", 8)


def test_registry_custom_name_field(tmp_path: Path) -> None:
    @dataclass
    class Entry:
        slug: str

    registry = Registry("entries", tmp_path, Entry, name_field="slug")
    (tmp_path / "entry.yaml").write_text("{}\n")
    assert registry.load("entry") == Entry("entry")
    (tmp_path / "entry.yaml").write_text("slug: wrong\n")
    with pytest.raises(ConfigError, match="slug.*wrong.*entry"):
        registry.load("entry")


def test_registry_without_name_field(tmp_path: Path) -> None:
    @dataclass
    class Entry:
        value: int

    (tmp_path / "entry.yaml").write_text("value: 3\n")
    assert Registry("entries", tmp_path, Entry).load("entry") == Entry(3)


def test_registry_unknown_keys_are_sorted(tmp_path: Path) -> None:
    (tmp_path / "qwen3_8b.yaml").write_text("width: 8\nzoo: 0\nfoo: 1\n")
    with pytest.raises(ConfigError) as caught:
        Registry("models", tmp_path, Model).load("qwen3_8b")
    assert str(caught.value) == "models registry entry 'qwen3_8b': unknown keys ['foo', 'zoo']"


@pytest.mark.parametrize("key", ["name", "width"])
def test_registry_rejects_init_false_keys(tmp_path: Path, key: str) -> None:
    @dataclass
    class Entry:
        name: str = field(init=False, default="internal")
        width: int = field(init=False, default=8)

    path = tmp_path / "entry.yaml"
    path.write_text("{}\n")
    registry = Registry("entries", tmp_path, Entry)
    assert registry.load("entry") == Entry()
    path.write_text(f"{key}: 1\n")
    with pytest.raises(ConfigError) as caught:
        registry.load("entry")
    assert str(caught.value) == f"entries registry entry 'entry': unknown keys ['{key}']"


def test_registry_missing_required_fields(tmp_path: Path) -> None:
    (tmp_path / "entry.yaml").write_text("{}\n")
    with pytest.raises(ConfigError, match="entry.*missing required fields.*width"):
        Registry("models", tmp_path, Model).load("entry")


def test_registry_wraps_post_init_error(tmp_path: Path) -> None:
    (tmp_path / "entry.yaml").write_text("width: -1\n")
    with pytest.raises(ConfigError, match="entry.*width must be positive") as caught:
        Registry("models", tmp_path, Model).load("entry")
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("content", ["", "null\n", "[1, 2]\n", "a scalar\n", "3\n"])
def test_registry_requires_mapping(tmp_path: Path, content: str) -> None:
    (tmp_path / "entry.yaml").write_text(content)
    with pytest.raises(ConfigError, match="entry.*mapping"):
        Registry("models", tmp_path, Model).load("entry")


def test_registry_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "entry.yaml").write_text("width: [\n")
    with pytest.raises(ConfigError, match="entry.*invalid YAML"):
        Registry("models", tmp_path, Model).load("entry")


def test_registry_path_errors_list_available_names(tmp_path: Path) -> None:
    (tmp_path / "beta.yaml").write_text("width: 2\n")
    (tmp_path / "alpha.yaml").write_text("width: 1\n")
    registry = Registry("models", tmp_path, Model)
    with pytest.raises(ConfigError) as caught:
        registry.path("missing")
    assert "missing" in str(caught.value)
    assert "['alpha', 'beta']" in str(caught.value)


def test_function_registry_register_get_and_names() -> None:
    registry: FnRegistry[Callable[[int], int]] = FnRegistry("rewards")
    assert registry.names() == []
    assert registry.register("zeta")(identity) is identity

    @registry.register("alpha")
    def twice(value: int) -> int:
        return 2 * value

    assert registry.names() == ["alpha", "zeta"]
    assert registry.get("zeta") is identity
    assert registry.get("alpha") is twice
    assert registry.get("alpha")(3) == 6


def test_function_registry_duplicate_rejected() -> None:
    registry = FnRegistry("rewards")
    registry.register("same")(identity)
    with pytest.raises(ValueError, match="same.*already registered"):
        registry.register("same")(identity)
    assert registry.get("same") is identity


def test_function_registry_lambda_rejected() -> None:
    registry = FnRegistry("rewards")
    with pytest.raises(ValueError, match="named callable"):
        registry.register("anonymous")(lambda value: value)
    assert registry.names() == []


def test_function_registry_imports_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_import = registry_module.importlib.import_module

    def tracked_import(module: str) -> object:
        calls.append(module)
        return real_import(module)

    monkeypatch.setattr(registry_module.importlib, "import_module", tracked_import)
    registry = FnRegistry("rewards")
    registry.register("identity")(identity)
    assert registry.get("identity") is identity
    assert calls == []
    assert registry.get("json:dumps") is json.dumps
    assert calls == ["json"]
    assert registry.names() == ["identity"]


def test_function_registry_non_callable_import_rejected() -> None:
    with pytest.raises(ValueError, match="named callable"):
        FnRegistry("rewards").get("json:__name__")


@pytest.mark.parametrize(
    ("ref", "cause_type"),
    [
        ("no_such_module_xyz:f", ImportError),
        ("json:no_such_attr", AttributeError),
        (":", ValueError),
    ],
)
def test_function_registry_resolution_errors_are_config_errors(
    ref: str, cause_type: type[Exception]
) -> None:
    with pytest.raises(ConfigError) as caught:
        FnRegistry("rewards").get(ref)
    cause = caught.value.__cause__
    assert isinstance(cause, cause_type)
    assert str(caught.value) == f"cannot resolve rewards function {ref!r}: {cause}"


def test_function_registry_unknown_name_lists_choices() -> None:
    registry = FnRegistry("rewards")
    registry.register("identity")(identity)
    with pytest.raises(ConfigError, match="absent.*identity"):
        registry.get("absent")


def test_catalog_resolves_only_requested_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "small.yaml").write_text("width: 4\n")
    monkeypatch.setattr(
        sys.modules[__name__], "_test_models", Registry("models", tmp_path, Model), raising=False
    )
    functions = FnRegistry("rewards")
    functions.register("identity")(identity)
    monkeypatch.setattr(sys.modules[__name__], "_test_rewards", functions, raising=False)
    monkeypatch.setattr(
        registry_module,
        "CATALOG",
        {
            "rewards": f"{__name__}:_test_rewards",
            "models": f"{__name__}:_test_models",
            "unloaded": "not_a_real_module:registry",
        },
    )
    assert catalog_kinds() == ["models", "rewards", "unloaded"]
    assert catalog_names("models") == ["small"]
    assert catalog_names("rewards") == ["identity"]
    with pytest.raises(ConfigError, match="absent.*models.*rewards.*unloaded"):
        catalog_names("absent")


def test_catalog_can_be_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "CATALOG", {})
    assert catalog_kinds() == []
