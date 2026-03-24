"""Shared fixtures and subprocess helpers for the zap-memwalk test suite."""

from __future__ import annotations

import select
import subprocess
import sys
from collections.abc import Generator

import pytest

try:
    import frida  # noqa: F401

    _FRIDA_AVAILABLE = True
except ImportError:
    _FRIDA_AVAILABLE = False

frida_mark = pytest.mark.skipif(not _FRIDA_AVAILABLE, reason="frida not installed")


def _allow_ptrace() -> None:
    """preexec_fn: allow any process to ptrace this child (Linux Yama workaround)."""
    import ctypes

    PR_SET_PTRACER = 0x59616D61
    PR_SET_PTRACER_ANY = -1
    try:
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    except Exception:
        pass


def _spawn(cmd: str) -> subprocess.Popen[bytes]:
    """Spawn sys.executable running *cmd*, ready to be signalled via stdin/stdout."""
    kwargs: dict[str, object] = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE}
    if sys.platform == "linux":
        kwargs["preexec_fn"] = _allow_ptrace
    return subprocess.Popen([sys.executable, "-c", cmd], **kwargs)  # type: ignore[call-overload]


def _teardown(proc: subprocess.Popen[bytes]) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _ready(proc: subprocess.Popen[bytes], timeout: float = 15.0) -> None:
    """Block until the child writes its readiness line to stdout."""
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        proc.kill()
        proc.wait()
        raise TimeoutError(f"child process did not signal ready within {timeout}s")
    proc.stdout.readline()


def _ready_with_line(proc: subprocess.Popen[bytes], timeout: float = 15.0) -> str:
    """Block until the child writes a line to stdout; return that line decoded."""
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        proc.kill()
        proc.wait()
        raise TimeoutError(f"child process did not signal ready within {timeout}s")
    return proc.stdout.readline().decode().strip()


# ── idle / allocation fixtures ────────────────────────────────────────────────


@pytest.fixture()
def idle_proc() -> Generator[subprocess.Popen[bytes], None, None]:
    """Spawn a Python interpreter blocking on stdin with no extra allocations."""
    proc = _spawn("import sys; print(flush=True); sys.stdin.read()")
    _ready(proc)
    yield proc
    _teardown(proc)


@pytest.fixture()
def alloc_proc() -> Generator[subprocess.Popen[bytes], None, None]:
    """Allocate 1 M ints, free every other one (~50% fill on 32-byte size class)."""
    proc = _spawn(
        "x = list(range(1_000_000)); del x[::2];"
        " import sys; print(flush=True); sys.stdin.read()"
    )
    _ready(proc)
    yield proc
    _teardown(proc)


# ── fixtures that export a known object address via stdout ────────────────────
# The child prints  id(obj)  on stdout before blocking so the test can target
# the exact address.  Use _ready_with_line() to read the address line.


@pytest.fixture()
def known_list_proc() -> Generator[tuple[int, int], None, None]:
    """Process holding a 1000-element list; yields (pid, address_of_list)."""
    proc = _spawn(
        "x = [1] * 1000; import sys; print(id(x), flush=True); sys.stdin.read()"
    )
    addr = int(_ready_with_line(proc))
    yield proc.pid, addr
    _teardown(proc)


@pytest.fixture()
def known_str_proc() -> Generator[tuple[int, int], None, None]:
    """Process holding a known string; yields (pid, address_of_str)."""
    proc = _spawn(
        "x = 'hello from zap-memwalk test';"
        " import sys; print(id(x), flush=True); sys.stdin.read()"
    )
    addr = int(_ready_with_line(proc))
    yield proc.pid, addr
    _teardown(proc)


@pytest.fixture()
def known_dict_proc() -> Generator[tuple[int, int], None, None]:
    """Process holding a small dict; yields (pid, address_of_dict)."""
    proc = _spawn(
        "x = {'key': 'value', 'num': 42};"
        " import sys; print(id(x), flush=True); sys.stdin.read()"
    )
    addr = int(_ready_with_line(proc))
    yield proc.pid, addr
    _teardown(proc)
