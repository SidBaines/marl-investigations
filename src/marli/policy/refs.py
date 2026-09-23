"""Policy refs keep config identities independent of live backend objects.

Custom Tinker endpoints use ``tinker@<url>|<base_model>`` exclusively: the
pipe separates the URL from the model, so ports, IPv6 and URL paths cannot
be mistaken for model delimiters. ``#`` introduces a single suffix (or the
vLLM model) and cannot occur literally inside a field.
Canonical checkpoint refs omit ``#step=final`` and normalize integer steps.
Parsing never imports a backend or reads checkpoint/server manifests.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from marli.errors import ConfigError
from marli.policy.base import ChatPolicy, TokenPolicy

_VALID_FORMS = (
    "tinker:<base_model>[#sampler=<path>]; "
    "tinker@<url>|<base_model>[#sampler=<path>]; "
    "ckpt:<path>[#step=<integer|final>]; "
    "vllm:<url>#<model>; vllm:@<server.json>#<model>; "
    "api:<openai|anthropic|openrouter>/<model>; scripted:<module>:<attr>"
)


@dataclass(frozen=True)
class PolicyRef:
    kind: str
    target: str
    base_url: str | None = None
    server_json: str | None = None
    provider: str | None = None
    sampler: str | None = None
    step: int | str | None = None

    def __post_init__(self) -> None:
        allowed = {
            "tinker": ("base_url", "sampler"),
            "ckpt": ("step",),
            "vllm": ("base_url", "server_json"),
            "api": ("provider",),
            "scripted": (),
        }
        try:
            if not isinstance(self.kind, str) or self.kind not in allowed:
                raise ValueError(f"unknown policy kind {self.kind!r}")
            for field in ("base_url", "server_json", "provider", "sampler", "step"):
                if field not in allowed[self.kind] and getattr(self, field) is not None:
                    raise ValueError(f"{self.kind} refs do not accept {field}")
            for field in ("target", "base_url", "server_json", "provider", "sampler"):
                value = getattr(self, field)
                if field != "target" and value is None:
                    continue
                if not isinstance(value, str) or not value.strip() or "#" in value:
                    raise ValueError(f"{field} must be a non-empty string without '#'")
            if self.base_url is not None:
                _validate_url(self.base_url)
            if self.kind in ("tinker", "vllm", "api"):
                _validate_model(self.target)
            if self.kind == "tinker":
                if self.base_url is not None and "|" in self.base_url:
                    raise ValueError("custom Tinker URLs cannot contain a literal pipe")
            elif self.kind == "ckpt":
                if self.step is None:
                    object.__setattr__(self, "step", "final")
                if self.step != "final" and not (type(self.step) is int and self.step >= 0):
                    raise ValueError("checkpoint step must be a non-negative integer or 'final'")
            elif self.kind == "vllm":
                if (self.base_url is None) == (self.server_json is None):
                    raise ValueError("vllm refs require exactly one of base_url or server_json")
            elif self.kind == "api":
                if self.provider not in ("openai", "anthropic", "openrouter"):
                    raise ValueError("API provider must be 'openai', 'anthropic', or 'openrouter'")
                if self.target.startswith("/") or self.target.endswith("/"):
                    raise ValueError("API model must not have a leading or trailing '/'")
            elif self.kind == "scripted":
                module, separator, attr = self.target.partition(":")
                if (
                    not separator
                    or not all(p.isidentifier() for p in module.split("."))
                    or not attr.isidentifier()
                ):
                    raise ValueError("scripted factory must be module:attr with valid Python names")
            if str(self) != str(self).strip():
                raise ValueError("reference must have no surrounding whitespace")
        except ValueError as exc:
            raise ConfigError(
                f"Invalid {self.kind!r} policy ref: {exc}. Valid forms: {_VALID_FORMS}"
            ) from exc

    def __str__(self) -> str:
        if self.kind == "tinker":
            head = (
                f"tinker@{self.base_url}|{self.target}"
                if self.base_url is not None
                else f"tinker:{self.target}"
            )
            return head if self.sampler is None else f"{head}#sampler={self.sampler}"
        if self.kind == "ckpt":
            suffix = "" if self.step in (None, "final") else f"#step={self.step}"
            return f"ckpt:{self.target}{suffix}"
        if self.kind == "vllm":
            server = f"@{self.server_json}" if self.server_json is not None else self.base_url
            return f"vllm:{server}#{self.target}"
        if self.kind == "api":
            return f"api:{self.provider}/{self.target}"
        if self.kind == "scripted":
            return f"scripted:{self.target}"
        raise ConfigError(f"Unknown policy kind {self.kind!r}. Valid forms: {_VALID_FORMS}")


def _validate_model(model: str) -> None:
    if not model or not model.strip("/") or any(c.isspace() for c in model) or "|" in model:
        raise ValueError("model must be non-empty and contain no whitespace or pipe")


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base URL must not contain credentials")
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or any(c.isspace() for c in url)
    ):
        raise ValueError("base URL must be an absolute http:// or https:// URL")
    # Accessing port also validates malformed/non-numeric ports without any I/O.
    _ = parsed.port


def parse_ref(s: str, *, resolve_paths: bool = False) -> PolicyRef:
    """Parse a ref without loading a backend; invalid syntax raises ConfigError.

    Relative checkpoint/server paths are preserved unless ``resolve_paths``
    is true, which makes them absolute against cwd without resolving symlinks.
    Absolute paths are always preserved unchanged.
    Checkpoint steps are non-negative integers or ``final`` (the default).
    Custom Tinker URLs require the pipe form documented above.
    """
    try:
        if not isinstance(s, str):
            raise ValueError("reference must be a string")
        if not s or s != s.strip():
            raise ValueError("reference must be non-empty with no surrounding whitespace")
        head, has_suffix, suffix = s.partition("#")
        if "#" in suffix:
            raise ValueError("only one # suffix is allowed")
        if head.startswith(("tinker:", "tinker@")):
            base_url = None
            if head.startswith("tinker@"):
                base_url, separator, target = head[len("tinker@") :].partition("|")
                if not separator:
                    raise ValueError("custom Tinker endpoints require tinker@<url>|<base_model>")
                _validate_url(base_url)
            else:
                target = head[len("tinker:") :]
            _validate_model(target)
            sampler = None
            if has_suffix:
                key, separator, sampler = suffix.partition("=")
                if key != "sampler" or not separator or not sampler.strip():
                    raise ValueError("Tinker suffix must be #sampler=<non-empty path>")
            return PolicyRef("tinker", target, base_url=base_url, sampler=sampler)

        kind, separator, target = head.partition(":")
        if not separator:
            raise ValueError("missing policy kind and ':' separator")
        if kind == "ckpt":
            if not target.strip():
                raise ValueError("checkpoint path must be non-empty")
            step: int | str = "final"
            if has_suffix:
                key, separator, value = suffix.partition("=")
                if key != "step" or not separator:
                    raise ValueError("checkpoint suffix must be #step=<integer|final>")
                if value != "final":
                    if not value.isascii() or not value.isdecimal():
                        raise ValueError(
                            "checkpoint step must be a non-negative integer or 'final'"
                        )
                    step = int(value)
            if resolve_paths and not os.path.isabs(target):
                target = os.path.abspath(target)
            return PolicyRef("ckpt", target, step=step)
        if kind == "vllm":
            _validate_model(suffix)
            if target.startswith("@"):
                path = target[1:]
                if not path.strip():
                    raise ValueError("server.json path must be non-empty")
                if resolve_paths and not os.path.isabs(path):
                    path = os.path.abspath(path)
                return PolicyRef("vllm", suffix, server_json=path)
            _validate_url(target)
            return PolicyRef("vllm", suffix, base_url=target)
        if has_suffix:
            raise ValueError(f"{kind!r} refs do not accept a # suffix")
        if kind == "api":
            provider, separator, model = target.partition("/")
            if provider not in ("openai", "anthropic", "openrouter"):
                raise ValueError("API provider must be 'openai', 'anthropic', or 'openrouter'")
            _validate_model(model)
            return PolicyRef("api", model, provider=provider)
        if kind == "scripted":
            if not target:
                raise ValueError("scripted factory must be module:attr with valid Python names")
            return PolicyRef("scripted", target)
        raise ValueError(f"unknown policy kind {kind!r}")
    except ValueError as exc:
        raise ConfigError(f"Invalid policy ref {s!r}: {exc}. Valid forms: {_VALID_FORMS}") from exc


def resolve_scripted(ref: PolicyRef) -> TokenPolicy | ChatPolicy:
    """Call a factory lazily; raise ConfigError unless it returns a token or chat policy."""
    if ref.kind != "scripted":
        raise ConfigError(f"Expected a scripted policy ref, got kind {ref.kind!r}")
    try:
        module_name, attr = ref.target.split(":")
        factory = getattr(importlib.import_module(module_name), attr)
        policy = factory()
    except Exception as exc:
        raise ConfigError(f"Cannot resolve scripted policy {ref.target!r}: {exc}") from exc
    if not isinstance(policy, (TokenPolicy, ChatPolicy)):
        raise ConfigError(
            f"Scripted factory {ref.target!r} must return a TokenPolicy or ChatPolicy, "
            f"got {type(policy).__name__}"
        )
    return policy
