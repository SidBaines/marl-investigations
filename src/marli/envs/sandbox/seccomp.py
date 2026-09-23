"""Close socket and privileged-syscall escapes left open by Landlock ABI 4.

This module is stdlib-only: the isolated launcher loads it directly from the
same checkout, before executing any model-written program. Filters survive
exec and fork; an unknown syscall ABI must never bypass the x86_64 policy.
"""

from __future__ import annotations

import ctypes
import errno
import os

_AUDIT_ARCH_X86_64 = 0xC000003E
_KILL_PROCESS = 0x80000000
_ALLOW = 0x7FFF0000
_ERRNO = 0x00050000
_LD_W_ABS = 0x20
_AND_K = 0x54
_JEQ = 0x15
_JGE = 0x35
_RET = 0x06

# Linux x86_64 syscall numbers, independent of libc wrapper availability.
_DENIED = (
    425,  # io_uring_setup
    426,  # io_uring_enter
    427,  # io_uring_register
    101,  # ptrace
    165,  # mount
    166,  # umount2
    272,  # unshare
    308,  # setns
    321,  # bpf
    298,  # perf_event_open
    250,  # keyctl
    248,  # add_key
    249,  # request_key
    310,  # process_vm_readv
    311,  # process_vm_writev
    246,  # kexec_load
    175,  # init_module
    313,  # finit_module
)


class _Filter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _Program(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


def install() -> None:
    """Install after PR_SET_NO_NEW_PRIVS; raise OSError if enforcement fails."""
    instructions = [
        (_LD_W_ABS, 0, 0, 4),  # seccomp_data.arch
        (_JEQ, 1, 0, _AUDIT_ARCH_X86_64),
        (_RET, 0, 0, _KILL_PROCESS),
        (_LD_W_ABS, 0, 0, 0),  # seccomp_data.nr
        (_JGE, 0, 1, 0x40000000),  # x32 uses a separate syscall table
        (_RET, 0, 0, _KILL_PROCESS),
        (_JEQ, 0, 1, 41),  # socket: even AF_UNIX can reach host listeners
        (_RET, 0, 0, _ERRNO | errno.EACCES),
        (_JEQ, 0, 8, 53),  # socketpair: asyncio needs anonymous Unix stream pairs
        (_LD_W_ABS, 0, 0, 16),  # args[0] (the kernel interprets domain as int)
        (_JEQ, 1, 0, 1),  # AF_UNIX
        (_RET, 0, 0, _ERRNO | errno.EACCES),
        # Datagram pairs can disconnect and use connect/sendto to reach named
        # host peers. Stream pairs cannot reconnect; retain their creation flags.
        (_LD_W_ABS, 0, 0, 24),  # args[1]: socket type
        (_AND_K, 0, 0, 0xF),  # SOCK_TYPE_MASK (ignore NONBLOCK/CLOEXEC)
        (_JEQ, 1, 0, 1),  # SOCK_STREAM
        (_RET, 0, 0, _ERRNO | errno.EACCES),
        (_RET, 0, 0, _ALLOW),
    ]
    for number in _DENIED:
        instructions.extend(
            [
                (_JEQ, 0, 1, number),
                (_RET, 0, 0, _ERRNO | errno.EPERM),
            ]
        )
    instructions.append((_RET, 0, 0, _ALLOW))
    filters = (_Filter * len(instructions))(*(_Filter(*row) for row in instructions))
    program = _Program(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
