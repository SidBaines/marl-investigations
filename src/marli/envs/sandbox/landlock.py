"""Deny filesystem access and TCP by default before executing untrusted code.

Only the launcher installs a ruleset: restricting the training process would
be irreversible. ABI probing is safe in the parent and requires no privileges.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

_CREATE_RULESET = 444
_ADD_RULE = 445
_RESTRICT_SELF = 446
_READ = (1 << 0) | (1 << 2) | (1 << 3)
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _checked(result: int) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def abi_version() -> int:
    """Return zero when Landlock is unavailable on this Linux/x86_64 host."""
    if sys.platform != "linux" or platform.machine() != "x86_64":
        return 0
    return max(0, _LIBC.syscall(_CREATE_RULESET, 0, 0, 1))


def restrict(workdir: Path, abi: int) -> None:
    """Apply the supported filesystem rights and, since ABI 4, deny TCP."""
    fs_rights = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    rules = _Ruleset(fs_rights, 3 if abi >= 4 else 0)
    size = ctypes.sizeof(rules) if abi >= 4 else ctypes.sizeof(ctypes.c_uint64)
    ruleset_fd = _checked(_LIBC.syscall(_CREATE_RULESET, ctypes.byref(rules), size, 0))
    try:
        read_paths = [
            Path("/usr"),
            Path("/bin"),
            *Path("/").glob("lib*"),
            Path("/etc"),
            Path("/opt"),
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path("/dev/null"),
            Path("/dev/urandom"),
            Path("/proc/self"),
        ]
        for path in dict.fromkeys([*read_paths, workdir]):
            if not path.exists():
                continue
            access = fs_rights if path == workdir else _READ
            if not path.is_dir():
                access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathBeneath(access, path_fd)
                _checked(_LIBC.syscall(_ADD_RULE, ruleset_fd, 1, ctypes.byref(rule), 0))
            finally:
                os.close(path_fd)
        _checked(_LIBC.syscall(_RESTRICT_SELF, ruleset_fd, 0))
    finally:
        os.close(ruleset_fd)
