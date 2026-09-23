"""Set restrictions in a fresh interpreter, avoiding thread-unsafe preexec_fn."""

from __future__ import annotations

import ctypes
import json
import os
import resource
import sys
from pathlib import Path

from marli.envs.sandbox.landlock import restrict


def main() -> None:
    config = json.loads(sys.argv[1])
    if sys.argv[2] != "--" or not sys.argv[3:]:
        raise ValueError("expected launcher configuration followed by -- and command")
    limits = config["limits"]
    for name in ("as", "cpu", "fsize", "nofile", "nproc"):
        if name == "nproc" and config["uid"] is None:
            continue
        limit = limits[name]
        resource.setrlimit(getattr(resource, f"RLIMIT_{name.upper()}"), (limit, limit))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if config["landlock_abi"]:
        restrict(Path(config["workdir"]), config["landlock_abi"])
    if config["uid"] is not None:
        os.setgroups([])
        os.setgid(config["uid"])
        os.setuid(config["uid"])
    os.environ["TMPDIR"] = str(Path(config["workdir"]) / "tmp")
    os.execvp(sys.argv[3], sys.argv[3:])


if __name__ == "__main__":
    main()
