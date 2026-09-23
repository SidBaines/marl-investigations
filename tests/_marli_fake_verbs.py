"""Importable lazy targets for CLI and runner tests under pytest's prepend mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from marli.config import input_field, runtime_field
from marli.handles import Handle, InputRef, register_handle
from marli.rundir import RunDir
from marli.verbs import VerbSpec


@dataclass
class EchoConfig:
    message: str = "hi"
    n: int = 3
    concurrency: int = runtime_field(4)
    source: str | None = input_field(None)


@register_handle
@dataclass(frozen=True, kw_only=True)
class DummyHandle(Handle):
    KIND: ClassVar[str] = "dummy"
    MANIFEST: ClassVar[str] = "dummy.json"
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ("rows",)
    rows: str = "rows.jsonl"
    n: int = 3

    def summary(self) -> dict[str, int]:
        return {"n": self.n}


invocations = 0
crash_after: int | None = None


async def echo(cfg: EchoConfig, run: RunDir) -> DummyHandle:
    global invocations
    invocations += 1
    print("junk from echo")
    done = run.done_keys("rows.jsonl", "i")
    for i in range(cfg.n):
        if i not in done:
            run.append_row("rows.jsonl", {"i": i, "message": cfg.message})
            done.add(i)
            run.write_progress({"done": len(done)})
            if len(done) == crash_after:
                raise RuntimeError("simulated crash")
    inputs = (InputRef.from_manifest(cfg.source),) if cfg.source is not None else ()
    return DummyHandle(root=run.out, n=cfg.n, inputs=inputs)


SPEC = VerbSpec(
    "debug echo",
    "_marli_fake_verbs:echo",
    "_marli_fake_verbs:EchoConfig",
    "dummy.json",
    "Write resumable rows for contract tests.",
)


@dataclass
class NestedConfig:
    child: EchoConfig = field(default_factory=EchoConfig)


@dataclass
class OptionalNestedConfig:
    child: EchoConfig | None = None


config_instance = EchoConfig()
