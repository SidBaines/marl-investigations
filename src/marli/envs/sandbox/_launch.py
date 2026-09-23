"""Apply irreversible restrictions in a fresh, stdlib-only interpreter.

Loading sibling files by absolute path avoids editable installs of another
worktree and prevents model-written modules from running before privilege drop.
The cleanup helper drops identity before kill(-1), closing fork races without
making the trainer a subreaper of unrelated vLLM or training processes.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import resource
import signal
import sys
from contextlib import suppress
from pathlib import Path
from types import ModuleType


def _sibling(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sandbox restriction module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_new_privs() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def main() -> None:
    if sys.argv[1] == "--kill-uid":
        uid = int(sys.argv[2])
        if uid <= 0:
            raise ValueError("cleanup requires an unprivileged uid")
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        with suppress(ProcessLookupError):
            os.kill(-1, signal.SIGKILL)
        return
    if sys.argv[1] == "--probe-seccomp":
        _no_new_privs()
        _sibling("seccomp").install()
        sys.stdout.write(json.dumps({"launcher": str(Path(__file__).resolve())}) + "\n")
        return
    config = json.loads(sys.argv[1])
    if sys.argv[2] != "--" or not sys.argv[3:]:
        raise ValueError("expected launcher configuration followed by -- and command")
    landlock = _sibling("landlock")
    seccomp = _sibling("seccomp")
    limits = config["limits"]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name in ("as", "cpu", "fsize", "nofile", "nproc"):
        if name == "nproc" and config["uid"] is None:
            continue
        limit = limits[name]
        resource.setrlimit(getattr(resource, f"RLIMIT_{name.upper()}"), (limit, limit))
    Path("/proc/self/oom_score_adj").write_text("1000")
    _no_new_privs()
    if config["landlock_abi"]:
        landlock.restrict(Path(config["workdir"]), config["landlock_abi"])
    if config["uid"] is not None:
        os.setgroups([])
        os.setgid(config["uid"])
        os.setuid(config["uid"])
    if config["seccomp"]:
        seccomp.install()
    os.environ["TMPDIR"] = str(Path(config["workdir"]) / "tmp")
    os.execvp(sys.argv[3], sys.argv[3:])


if __name__ == "__main__":
    main()
