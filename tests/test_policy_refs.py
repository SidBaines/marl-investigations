"""Policy names round-trip without backend imports, network access or manifest reads."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType

import pytest

import marli.policy.refs as refs_module
from marli.errors import ConfigError
from marli.policy.base import ChatPolicy, ChatReply, TokenPolicy
from marli.policy.refs import PolicyRef, parse_ref, resolve_scripted
from marli.policy.scripted import ScriptedChatPolicy, ScriptedPolicy, Turn, turns_by_agent
from marli.render.fake import FakeRenderer


def policy_factory() -> ScriptedPolicy:
    renderer = FakeRenderer()
    return ScriptedPolicy("factory-policy", renderer, turns_by_agent(renderer, {}, default=Turn()))


def failing_factory() -> ScriptedPolicy:
    raise RuntimeError("factory failed deliberately")


@pytest.mark.parametrize(
    ("text", "expected", "canonical"),
    [
        ("tinker:Qwen/Qwen3.5-4B", PolicyRef("tinker", "Qwen/Qwen3.5-4B"), None),
        (
            "tinker@http://host:8000|Qwen/Qwen3.5-4B",
            PolicyRef("tinker", "Qwen/Qwen3.5-4B", base_url="http://host:8000"),
            None,
        ),
        (
            "tinker@https://[::1]:8000/api/v1?key=value|Qwen/Qwen3.5-4B",
            PolicyRef("tinker", "Qwen/Qwen3.5-4B", base_url="https://[::1]:8000/api/v1?key=value"),
            None,
        ),
        (
            "tinker:Qwen/Qwen3.5-4B#sampler=tinker://run-id/sampler_weights/step10",
            PolicyRef(
                "tinker", "Qwen/Qwen3.5-4B", sampler="tinker://run-id/sampler_weights/step10"
            ),
            None,
        ),
        (
            "tinker@http://host:8000|Qwen/Qwen3.5-4B#sampler=tinker://run/sampler_weights/step10",
            PolicyRef(
                "tinker",
                "Qwen/Qwen3.5-4B",
                base_url="http://host:8000",
                sampler="tinker://run/sampler_weights/step10",
            ),
            None,
        ),
        ("ckpt:/abs/dir", PolicyRef("ckpt", "/abs/dir", step="final"), None),
        ("ckpt:rel/dir", PolicyRef("ckpt", "rel/dir", step="final"), None),
        ("ckpt:dir#step=12", PolicyRef("ckpt", "dir", step=12), None),
        ("ckpt:dir#step=0", PolicyRef("ckpt", "dir", step=0), None),
        ("ckpt:dir#step=final", PolicyRef("ckpt", "dir", step="final"), "ckpt:dir"),
        ("ckpt:dir#step=0012", PolicyRef("ckpt", "dir", step=12), "ckpt:dir#step=12"),
        (
            "vllm:http://127.0.0.1:8001#Qwen/Qwen3.5-4B",
            PolicyRef("vllm", "Qwen/Qwen3.5-4B", base_url="http://127.0.0.1:8001"),
            None,
        ),
        (
            "vllm:@path/to/server.json#learner@12",
            PolicyRef("vllm", "learner@12", server_json="path/to/server.json"),
            None,
        ),
        (
            "vllm:@/path/to/server.json#learner@12",
            PolicyRef("vllm", "learner@12", server_json="/path/to/server.json"),
            None,
        ),
        (
            "api:openrouter/anthropic/claude-sonnet-4.5",
            PolicyRef("api", "anthropic/claude-sonnet-4.5", provider="openrouter"),
            None,
        ),
        ("api:openai/gpt-5", PolicyRef("api", "gpt-5", provider="openai"), None),
        (
            "api:anthropic/claude-sonnet-4.5",
            PolicyRef("api", "claude-sonnet-4.5", provider="anthropic"),
            None,
        ),
        (
            "scripted:package.module:function",
            PolicyRef("scripted", "package.module:function"),
            None,
        ),
    ],
)
def test_ref_forms_round_trip(text: str, expected: PolicyRef, canonical: str | None) -> None:
    parsed = parse_ref(text)
    assert parsed == expected
    assert str(parsed) == (text if canonical is None else canonical)
    assert parse_ref(str(parsed)) == parsed
    assert parse_ref(str(expected)) == expected


def test_direct_checkpoint_defaults_to_final() -> None:
    ref = PolicyRef("ckpt", "checkpoint")
    assert ref.step == "final"
    assert str(ref) == "ckpt:checkpoint"
    assert parse_ref(str(ref)) == ref


@pytest.mark.parametrize(
    ("ref", "allowed"),
    [
        (PolicyRef("tinker", "model"), ("base_url", "sampler")),
        (PolicyRef("ckpt", "checkpoint"), ("step",)),
        (PolicyRef("vllm", "model", base_url="https://host"), ("base_url", "server_json")),
        (PolicyRef("api", "model", provider="openai"), ("provider",)),
        (PolicyRef("scripted", "module:factory"), ()),
    ],
)
def test_direct_refs_reject_forbidden_fields(ref: PolicyRef, allowed: tuple[str, ...]) -> None:
    for field, value in {
        "base_url": "https://host",
        "server_json": "server.json",
        "provider": "openai",
        "sampler": "sampler/path",
        "step": 1,
    }.items():
        if field not in allowed:
            with pytest.raises(ConfigError, match=field):
                replace(ref, **{field: value})


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "unknown", "target": "model"},
        {"kind": None, "target": "model"},
        {"kind": "tinker", "target": None},
        {"kind": "tinker", "target": ""},
        {"kind": "tinker", "target": "model name"},
        {"kind": "tinker", "target": "model#extra"},
        {"kind": "tinker", "target": "model", "base_url": "host"},
        {"kind": "tinker", "target": "model", "base_url": "https://host/a|b"},
        {"kind": "tinker", "target": "model", "sampler": ""},
        {"kind": "tinker", "target": "model", "sampler": "sampler#extra"},
        {"kind": "tinker", "target": "model", "sampler": "sampler "},
        {"kind": "ckpt", "target": " "},
        {"kind": "ckpt", "target": "checkpoint#extra"},
        {"kind": "ckpt", "target": "checkpoint "},
        {"kind": "vllm", "target": "model"},
        {"kind": "vllm", "target": "model", "base_url": "https://host", "server_json": "s.json"},
        {"kind": "vllm", "target": "model", "server_json": ""},
        {"kind": "vllm", "target": "model", "server_json": "server#extra.json"},
        {"kind": "api", "target": "model"},
        {"kind": "api", "target": "model", "provider": "unknown"},
        {"kind": "api", "target": "/model", "provider": "openai"},
        {"kind": "api", "target": "model/", "provider": "openai"},
        {"kind": "scripted", "target": "module"},
    ],
)
def test_invalid_direct_refs_raise_config_error(fields: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        PolicyRef(**fields)


@pytest.mark.parametrize("step", [-1, 1.0, True, False, "12", "0012", "latest"])
def test_direct_checkpoint_requires_canonical_step(step: object) -> None:
    with pytest.raises(ConfigError, match="step"):
        PolicyRef("ckpt", "checkpoint", step=step)


def test_ref_is_frozen() -> None:
    ref = parse_ref("ckpt:dir")
    with pytest.raises(FrozenInstanceError):
        ref.target = "other"  # type: ignore[misc]


def test_paths_are_resolved_only_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = "ckpt:some dir/../checkpoint#step=12"
    server = "vllm:@some dir/../server.json#learner@12"
    assert parse_ref(checkpoint).target == "some dir/../checkpoint"
    assert parse_ref(server).server_json == "some dir/../server.json"
    assert parse_ref(checkpoint, resolve_paths=True) == PolicyRef(
        "ckpt", str(tmp_path / "checkpoint"), step=12
    )
    assert parse_ref(server, resolve_paths=True) == PolicyRef(
        "vllm", "learner@12", server_json=str(tmp_path / "server.json")
    )
    for text in [f"ckpt:{tmp_path}/absolute", f"vllm:@{tmp_path}/server.json#model"]:
        assert parse_ref(text, resolve_paths=True) == parse_ref(text)
    for text in ["tinker:org/model", "vllm:https://host#model", "api:openai/model"]:
        assert parse_ref(text, resolve_paths=True) == parse_ref(text)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["ckpt", "vllm"])
def test_path_resolution_preserves_symlinks_and_absolute_spelling(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    for path, expected in (
        ("linked/file", str(tmp_path / "linked/file")),
        (f"{tmp_path}/linked/../file", f"{tmp_path}/linked/../file"),
        (f"{tmp_path}/linked/file", f"{tmp_path}/linked/file"),
    ):
        text = f"ckpt:{path}" if kind == "ckpt" else f"vllm:@{path}#model"
        ref = parse_ref(text, resolve_paths=True)
        assert (ref.target if kind == "ckpt" else ref.server_json) == expected


@pytest.mark.parametrize("value", [None, True, 7, b"ckpt:dir", Path("dir"), []])
def test_non_string_ref_is_config_error(value: object) -> None:
    with pytest.raises(ConfigError, match="reference must be a string"):
        parse_ref(value)


@pytest.mark.parametrize("kind", ["tinker", "vllm"])
@pytest.mark.parametrize("userinfo", ["user:pass", "user", ":pass", ""])
def test_urls_with_credentials_are_rejected(kind: str, userinfo: str) -> None:
    url = f"https://{userinfo}@host:8000/api"
    text = f"tinker@{url}|model" if kind == "tinker" else f"vllm:{url}#model"
    with pytest.raises(ConfigError, match="credentials"):
        parse_ref(text)
    with pytest.raises(ConfigError, match="credentials"):
        PolicyRef(kind, "model", base_url=url)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "non-empty"),
        (" tinker:model", "whitespace"),
        ("tinker", "separator"),
        ("unknown:model", "unknown policy kind"),
        ("tinker:", "model"),
        ("tinker: ", "whitespace"),
        ("tinker:model name", "model"),
        ("tinker@http://host:8000:org/model", "tinker@<url>|<base_model>"),
        ("tinker@http://host:8000|", "model"),
        ("tinker@|org/model", "URL"),
        ("tinker@http://host:bad|org/model", "Port"),
        ("tinker@http://host|org/model|extra", "pipe"),
        ("tinker:model#sampler=", "sampler"),
        ("tinker:model#step=12", "sampler"),
        ("tinker:model#sampler=path#extra", "one # suffix"),
        ("ckpt:", "path"),
        ("ckpt:dir#step=", "step"),
        ("ckpt:dir#step=-1", "step"),
        ("ckpt:dir#step=1.2", "step"),
        ("ckpt:dir#step=latest", "step"),
        ("ckpt:dir#steps=12", "suffix"),
        ("ckpt:dir#step=12#step=13", "one # suffix"),
        ("vllm:http://host:8000", "model"),
        ("vllm:http://host:8000#", "model"),
        ("vllm:@#model", "path"),
        ("vllm:#model", "URL"),
        ("vllm:host:8000#model", "URL"),
        ("vllm:http://[::1#model", "IPv6"),
        ("api:unknown/model", "provider"),
        ("api:openai", "model"),
        ("api:openai/", "model"),
        ("api:openai//", "model"),
        ("api:openai//model", "leading or trailing"),
        ("api:openrouter/org/model/", "leading or trailing"),
        ("api:openai/model#step=12", "suffix"),
        ("scripted:", "factory"),
        ("scripted:module", "factory"),
        ("scripted::factory", "factory"),
        ("scripted:module:", "factory"),
        ("scripted:module:factory:extra", "factory"),
        ("scripted:invalid-module:factory", "factory"),
        ("scripted:module:factory#extra", "suffix"),
    ],
)
def test_malformed_refs_explain_reason_and_valid_forms(text: str, reason: str) -> None:
    with pytest.raises(ConfigError) as caught:
        parse_ref(text)
    message = str(caught.value)
    assert repr(text) in message
    assert reason in message
    assert "Valid forms:" in message
    for kind in ("tinker:", "tinker@", "ckpt:", "vllm:", "api:", "scripted:"):
        assert kind in message


def test_scripted_factory_is_only_imported_and_called_during_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    real_import = refs_module.importlib.import_module

    def tracked_import(name: str) -> ModuleType:
        imports.append(name)
        return real_import(name)

    monkeypatch.setattr(refs_module.importlib, "import_module", tracked_import)
    ref = parse_ref(f"scripted:{__name__}:policy_factory")
    assert imports == []
    policy = resolve_scripted(ref)
    assert imports == [__name__]
    assert isinstance(policy, TokenPolicy)
    assert policy.policy_id == "factory-policy"
    assert resolve_scripted(ref) is not policy


@pytest.mark.parametrize(
    ("target", "cause"),
    [
        ("no_such_policy_module:factory", ModuleNotFoundError),
        (f"{__name__}:missing_factory", AttributeError),
        (f"{__name__}:__name__", TypeError),
        (f"{__name__}:failing_factory", RuntimeError),
        (f"{__name__}:parse_ref", TypeError),
    ],
)
def test_scripted_resolution_failures_are_config_errors(
    target: str, cause: type[Exception]
) -> None:
    with pytest.raises(ConfigError) as caught:
        resolve_scripted(parse_ref(f"scripted:{target}"))
    assert target in str(caught.value)
    assert isinstance(caught.value.__cause__, cause)
    assert str(caught.value.__cause__) in str(caught.value)


def test_resolve_scripted_rejects_other_kinds() -> None:
    with pytest.raises(ConfigError, match="scripted.*tinker"):
        resolve_scripted(parse_ref("tinker:org/model"))


def test_scripted_factory_accepts_chat_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    def reply(*args: object) -> ChatReply:
        pytest.fail("resolution must not make a chat call")

    policy = ScriptedChatPolicy("chat-factory", reply)
    module = ModuleType("chat_factory")
    module.factory = lambda: policy
    monkeypatch.setattr(refs_module.importlib, "import_module", lambda name: module)
    assert resolve_scripted(PolicyRef("scripted", "chat_factory:factory")) is policy
    assert isinstance(policy, ChatPolicy)


@pytest.mark.parametrize("result", [None, 42, "policy", object(), {"policy_id": "fake"}])
def test_scripted_factory_rejects_non_policies(
    result: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("invalid_factory")
    module.factory = lambda: result
    monkeypatch.setattr(refs_module.importlib, "import_module", lambda name: module)
    with pytest.raises(ConfigError, match="TokenPolicy or ChatPolicy"):
        resolve_scripted(PolicyRef("scripted", "invalid_factory:factory"))
