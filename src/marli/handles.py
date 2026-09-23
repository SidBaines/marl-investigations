"""Manifests make handles portable while retaining the exact provenance of inputs.

Own files are relative to the manifest; input references identify saved manifest
bytes. Atomic replacement keeps readers from observing partially written JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self
from uuid import uuid4

from marli import __version__
from marli.config import to_dict
from marli.errors import ConfigError

MANIFEST_VERSION = 1
_BOOKKEEPING = {"kind", "manifest_version", "marli_version", "created_at"}


def sha256_file(path: str | Path) -> str:
    """Hash the saved bytes without loading a potentially large file into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Durably replace a UTF-8 file using a temporary file on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid manifest JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: manifest must be a JSON object")
    return data


def _input_path(path: Path) -> str:
    path = path.resolve()
    data_dir = os.environ.get("MARLI_DATA")
    if data_dir:
        root = Path(data_dir).resolve()
        if path.is_relative_to(root):
            return f"$MARLI_DATA/{path.relative_to(root).as_posix()}"
    return str(path)


@dataclass(frozen=True)
class InputRef:
    """A consumed manifest's location and digest, independent of its handle type."""

    kind: str
    path: str
    sha256: str

    @classmethod
    def of(cls, handle: Handle) -> InputRef:
        """Reference an already saved handle, hashing its current manifest bytes."""
        path = handle.manifest_path
        return cls(kind=handle.KIND, path=_input_path(path), sha256=sha256_file(path))

    @classmethod
    def from_manifest(cls, path: str | Path) -> InputRef:
        """Read the kind and digest without requiring a registered handle class."""
        path = Path(path)
        content = path.read_bytes()
        return cls(
            kind=json.loads(content)["kind"],
            path=_input_path(path),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def resolve(self) -> Path:
        """Expand portable data-root references to an absolute manifest path."""
        if self.path == "$MARLI_DATA" or self.path.startswith("$MARLI_DATA/"):
            data_dir = os.environ.get("MARLI_DATA")
            if not data_dir:
                raise ConfigError(f"MARLI_DATA is unset; cannot resolve {self.path!r}")
            return (Path(data_dir) / self.path.removeprefix("$MARLI_DATA").lstrip("/")).resolve()
        return Path(self.path).resolve()


@dataclass(frozen=True, kw_only=True)
class Handle:
    """Base for small frozen dataclasses backed by a manifest beside their bytes.

    Subclasses set KIND, MANIFEST, and PATH_FIELDS and declare their data fields.
    JSON converts tuples to lists; load restores only inputs to a tuple. Other
    tuple-typed fields must be restored by the subclass, for example in __post_init__.
    """

    KIND: ClassVar[str]
    MANIFEST: ClassVar[str]
    PATH_FIELDS: ClassVar[tuple[str, ...]] = ()

    root: Path
    config_hash: str | None = None
    inputs: tuple[InputRef, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.root / self.MANIFEST

    def file(self, name: str) -> Path:
        """Resolve a declared path field to this handle's own bytes."""
        if name not in self.PATH_FIELDS:
            raise ConfigError(f"{type(self).__name__} has no path field {name!r}")
        value = getattr(self, name)
        if value is None:
            raise ConfigError(f"{self.manifest_path}: path field {name!r} is None")
        return self.root / value

    def summary(self) -> dict[str, Any]:
        """Extra fields for a verb's one-line CLI result."""
        return {}

    def save(self) -> Path:
        """Save relative file paths and preserve the manifest's original creation time."""
        data = to_dict(self)
        del data["root"]
        root = self.root.resolve()
        for name in self.PATH_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            path = (root / value).resolve()
            if not path.is_relative_to(root):
                raise ConfigError(
                    f"path field {name!r} resolves outside handle root {root}: {path}"
                )
            data[name] = path.relative_to(root).as_posix()
        path = self.manifest_path
        if path.exists():
            existing = _read_manifest(path)
            if "created_at" not in existing:
                raise ConfigError(f"{path}: manifest is missing created_at")
            created_at = existing["created_at"]
        else:
            created_at = datetime.now(UTC).isoformat(timespec="seconds")
        data.update(
            kind=self.KIND,
            manifest_version=MANIFEST_VERSION,
            marli_version=__version__,
            created_at=created_at,
        )
        atomic_write_text(
            path, json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        )
        return path

    def sha256(self) -> str:
        """Hash the saved manifest, rather than this instance's unsaved state."""
        return sha256_file(self.manifest_path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a manifest file or directory, rejecting incompatible schema changes."""
        path = Path(path)
        if path.is_dir():
            path /= cls.MANIFEST
        try:
            data = _read_manifest(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"manifest not found: {path}; use {cls.__name__}.at(path) for ad-hoc data"
            ) from exc
        if data.get("kind") != cls.KIND:
            raise ConfigError(f"{path}: expected kind {cls.KIND!r}, got {data.get('kind')!r}")
        version = data.get("manifest_version")
        if type(version) is not int:
            raise ConfigError(f"{path}: manifest_version must be an integer, got {version!r}")
        if version > MANIFEST_VERSION:
            raise ConfigError(
                f"{path}: manifest_version {data['manifest_version']} is newer than "
                f"supported version {MANIFEST_VERSION}"
            )
        names = {item.name for item in fields(cls)} - {"root"}
        unknown = data.keys() - names - _BOOKKEEPING
        if unknown:
            raise ConfigError(f"{path}: unknown manifest keys {sorted(unknown)}")
        values = {name: value for name, value in data.items() if name in names}
        try:
            values["inputs"] = tuple(InputRef(**ref) for ref in values.get("inputs", ()))
            return cls(root=path.resolve().parent, **values)
        except TypeError as exc:
            raise ConfigError(f"{path}: invalid manifest fields: {exc}") from exc

    @classmethod
    def at(cls, path: str | Path, **fields: Any) -> Self:
        """Explicitly wrap existing ad-hoc data without creating a manifest."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ad-hoc data not found: {path}")
        meta = {**fields.pop("meta", {}), "adhoc": True}
        return cls(root=path if path.is_dir() else path.parent, meta=meta, **fields)


HANDLE_TYPES: dict[str, type[Handle]] = {}


def register_handle[H: Handle](cls: type[H]) -> type[H]:
    """Register a unique manifest kind without changing the decorated class."""
    if cls.KIND in HANDLE_TYPES:
        raise ValueError(f"handle kind {cls.KIND!r} is already registered")
    HANDLE_TYPES[cls.KIND] = cls
    return cls


def load_any(path: str | Path) -> Handle:
    """Dispatch a manifest file to the registered class for its kind."""
    path = Path(path)
    kind = json.loads(path.read_text(encoding="utf-8"))["kind"]
    if kind not in HANDLE_TYPES:
        raise ConfigError(f"{path}: unknown handle kind {kind!r}")
    return HANDLE_TYPES[kind].load(path)


def inspect_chain(path: str | Path) -> dict[str, Any]:
    """Inspect each provenance edge, marking changed, missing, invalid, or cyclic inputs."""
    ancestors: set[Path] = set()

    def visit(path: Path) -> dict[str, Any]:
        path = path.resolve()
        if path in ancestors:
            return {"path": str(path), "cycle": True}
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"path": str(path), "sha256": digest, "invalid": True}
        children = []
        ancestors.add(path)
        try:
            for item in data.get("inputs", []):
                ref = InputRef(**item)
                child_path = ref.resolve()
                if not child_path.exists():
                    child = {
                        "kind": ref.kind,
                        "path": str(child_path),
                        "recorded_sha256": ref.sha256,
                        "missing": True,
                    }
                else:
                    child = visit(child_path)
                    if not child.get("cycle"):
                        child.update(
                            recorded_sha256=ref.sha256, sha256_ok=child["sha256"] == ref.sha256
                        )
                children.append(child)
        finally:
            ancestors.discard(path)
        return {
            "kind": data["kind"],
            "path": str(path),
            "sha256": digest,
            "config_hash": data.get("config_hash"),
            "meta": data.get("meta", {}),
            "inputs": children,
        }

    return visit(Path(path))
