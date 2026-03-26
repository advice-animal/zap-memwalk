"""Shared fixtures and subprocess helpers for the zap-memwalk test suite."""

from __future__ import annotations

import json
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
    proc = _spawn("import sys; print('x', flush=True); sys.stdin.read()")
    _ready(proc)
    yield proc
    _teardown(proc)


@pytest.fixture()
def alloc_proc() -> Generator[subprocess.Popen[bytes], None, None]:
    """Allocate 1 M ints, free every other one (~50% fill on 32-byte size class)."""
    proc = _spawn(
        "x = list(range(1_000_000)); del x[::2];"
        " import sys; print('x', flush=True); sys.stdin.read()"
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


@pytest.fixture()
def known_float_proc() -> Generator[tuple[int, int], None, None]:
    """Process holding a float; yields (pid, address_of_float).

    Uses a non-interned value so it's heap-allocated in a pymalloc pool.
    """
    proc = _spawn("x = 1.5; import sys; print(id(x), flush=True); sys.stdin.read()")
    addr = int(_ready_with_line(proc))
    yield proc.pid, addr
    _teardown(proc)


# ── integration test fixtures ─────────────────────────────────────────────────

# _SEQUENTIAL_ARENAS_SCRIPT: fill arenas one at a time with gc.disable() so
# pymalloc must allocate them in a strict sequence.  Each batch saturates one
# arena (510 floats × 64 pools = 32640, but we use 32000 to leave a few pools
# free in each arena so the arenas stay in usable_arenas rather than being
# "full" and therefore invisible to the list).  We record the address of the
# first float in each batch so the test can map object → arenaindex.
_SEQUENTIAL_ARENAS_SCRIPT = """\
import sys, json, gc
gc.disable()
pool_size = 16384
arena_size = 1 << 20  # 1 MiB
batches = []
# Fill 3 arenas sequentially, leaving each slightly under-full so they remain
# in usable_arenas (pymalloc keeps arenas in the list while nfreepools > 0).
for _ in range(3):
    batch = [float(i) + 0.5 for i in range(30000)]
    batches.append(hex(id(batch[0])))
print(json.dumps({"ready": True, "first_addrs": batches}), flush=True)
input()
"""


@pytest.fixture()
def sequential_arenas_proc() -> Generator[tuple[int, list[int]], None, None]:
    """Three sequentially filled arenas; yields (pid, [first_float_addr_per_batch])."""
    proc = _spawn(_SEQUENTIAL_ARENAS_SCRIPT)
    line = _ready_with_line(proc, timeout=15)
    data = json.loads(line)
    addrs = [int(a, 16) for a in data["first_addrs"]]
    yield proc.pid, addrs
    _teardown(proc)


_PRIM_SCRIPT = """\
import sys, json, gc, ctypes
gc.disable()
obj_float = 1.5
obj_int   = 2**62
obj_str   = "zap" * 100
obj_bytes = b"zap" * 100
obj_list  = list(range(50))
obj_dict  = {str(i): i for i in range(10)}

# Determine whether each type has a GC header by checking Py_TPFLAGS_HAVE_GC (bit 14).
# For GC-tracked types, id(x) = block_start + 16; for others id(x) = block_start.
Py_TPFLAGS_HAVE_GC = 1 << 14
def has_gc(obj):
    tp = type(obj)
    flags = ctypes.c_ulong.from_address(id(tp) + 8*3).value  # tp_flags at offset 24
    return bool(flags & Py_TPFLAGS_HAVE_GC)

objs = [
    ("float", obj_float), ("int", obj_int), ("str", obj_str),
    ("bytes", obj_bytes), ("list", obj_list), ("dict", obj_dict),
]
addrs = {k: {"id": hex(id(v)), "gc": has_gc(v)} for k, v in objs}
print(json.dumps({"ready": True, "objects": addrs}), flush=True)
input()
"""


@pytest.fixture()
def prim_proc() -> Generator[tuple[int, dict[str, int], dict[str, bool]], None, None]:
    """Spawn a process with one of each primitive type.

    Yields (pid, {type: id_address}, {type: has_gc_header}).
    id_address is what id(obj) returns — the PyObject address.
    For GC-tracked types, block_start = id_address - 16.
    For non-GC types, block_start = id_address.
    """
    proc = _spawn(_PRIM_SCRIPT)
    line = _ready_with_line(proc, timeout=10)
    data = json.loads(line)
    addrs = {k: int(v["id"], 16) for k, v in data["objects"].items()}
    gc_map = {k: bool(v["gc"]) for k, v in data["objects"].items()}
    yield proc.pid, addrs, gc_map
    _teardown(proc)


_COALESCE_SCRIPT = """\
import sys, json, gc, mmap as _mmap, ctypes
gc.disable()
# ~one full arena of floats: 64 pools * 510 blocks at 32-byte size class
batch1 = [float(i) + 0.5 for i in range(32640)]
# Non-1MB-multiple anonymous mmap (3 pages = 12288 bytes)
m = _mmap.mmap(-1, 4096 * 3)
buf_addr = hex(ctypes.addressof(ctypes.c_char.from_buffer(m)))
batch2 = [float(i) + 1.5 for i in range(32640)]
print(json.dumps({"ready": True, "mmap_addr": buf_addr, "mmap_size": 4096*3}), flush=True)
input()
"""


# _PHANTOM_SCRIPT: allocate a small anonymous mmap at a pool-size-aligned address
# immediately before a pymalloc arena so that scanAllPools encounters a
# non-pool-header at a pool-aligned address within the combined VMA.
#
# We force alignment by:
#   1. mmap(NULL, pool_size) → get a pool-aligned address P (mmap returns
#      page-aligned; if page_size divides pool_size, P is pool-aligned iff
#      P % pool_size == 0, which happens for ~25% of allocations).
#   2. If P is not pool-aligned, retry until it is (or give up after 32 tries).
#   3. Fill enough floats to force a new pymalloc arena adjacent to P.
#
# Linux: the small mmap and the adjacent arena may coalesce into one VMA.
_PHANTOM_SCRIPT = """\
import sys, json, gc, mmap as _mmap, ctypes
gc.disable()
pool_size = 16384
# Try to get a pool-size-aligned small mmap
for _ in range(64):
    m = _mmap.mmap(-1, pool_size)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
    if addr % pool_size == 0:
        break
    m.close()
else:
    print(json.dumps({"ready": False, "reason": "could not get aligned mmap"}), flush=True)
    input()
    sys.exit(0)
small_addr = hex(addr)
# Now allocate enough floats to provoke a new arena adjacent to the mmap
batch = [float(i) + 0.5 for i in range(32640)]
print(json.dumps({"ready": True, "small_addr": small_addr, "pool_size": pool_size}), flush=True)
input()
"""


@pytest.fixture()
def phantom_proc() -> Generator[tuple[int, int] | None, None, None]:
    """Linux-only: small pool-aligned mmap adjacent to a pymalloc arena.

    Yields (pid, small_addr) if alignment succeeded, None otherwise.
    """
    proc = _spawn(_PHANTOM_SCRIPT)
    line = _ready_with_line(proc, timeout=15)
    data = json.loads(line)
    if not data.get("ready"):
        yield None
    else:
        yield proc.pid, int(data["small_addr"], 16)
    _teardown(proc)


@pytest.fixture()
def coalesced_proc() -> Generator[tuple[int, int, int], None, None]:
    """Linux-only: two arenas separated by a non-1MB-multiple mmap; yields (pid, mmap_addr, mmap_size)."""
    proc = _spawn(_COALESCE_SCRIPT)
    line = _ready_with_line(proc, timeout=15)
    data = json.loads(line)
    mmap_addr = int(data["mmap_addr"], 16)
    mmap_size = data["mmap_size"]
    yield proc.pid, mmap_addr, mmap_size
    _teardown(proc)


# _FREEPOOLS_SCRIPT: fill a pool with floats, free them all, then block.
# The freed pool stays in the arena on the freepools singly-linked list;
# its header bytes are partially overwritten but maxnextoffset survives.
_FREEPOOLS_SCRIPT = """\
import sys, json, gc, ctypes
gc.disable()
# Allocate exactly one pool's worth of floats (szidx=1, 510 blocks per pool).
# Keep the pool address so we can report it.
sentinel = float(0.123456789)           # canary: first block in its pool
pool_size = 16384
pool_addr = id(sentinel) & ~(pool_size - 1)
batch = [sentinel] + [float(i) + 0.5 for i in range(509)]  # 510 total
# Verify we haven't spilled into a second pool
assert all((id(f) & ~(pool_size - 1)) == pool_addr for f in batch), \
    "floats spilled across pool boundary"
# Record the pool address, then free all of them
batch_addrs = [hex(id(f)) for f in batch]
del batch, sentinel
# The pool is now on freepools (ref.count overwritten with freepools ptr,
# often 0 if this was the only free pool; maxnextoffset still valid).
print(json.dumps({"ready": True, "pool_addr": hex(pool_addr),
                  "block_addrs": batch_addrs}), flush=True)
input()
"""


@pytest.fixture()
def freepools_proc() -> Generator[tuple[int, int, list[int]], None, None]:
    """Process with one freed pool (all blocks returned); yields (pid, pool_addr, block_addrs)."""
    proc = _spawn(_FREEPOOLS_SCRIPT)
    line = _ready_with_line(proc, timeout=10)
    data = json.loads(line)
    pool_addr = int(data["pool_addr"], 16)
    block_addrs = [int(a, 16) for a in data["block_addrs"]]
    yield proc.pid, pool_addr, block_addrs
    _teardown(proc)


# _STALE_TYPE_SCRIPT: import a C-extension type, allocate and free one instance,
# then print the ob_type pointer so the test can check stale-name handling.
# Uses _decimal.Decimal which is always available in the stdlib.
_STALE_TYPE_SCRIPT = """\
import sys, json, gc, ctypes
gc.disable()
import _decimal
obj = _decimal.Decimal("3.14")
ob_type = ctypes.c_size_t.from_address(id(obj) + 8).value  # ob_type offset
obj_addr = hex(id(obj))
type_addr = hex(ob_type)
type_name = type(obj).__qualname__
module_name = type(obj).__module__
del obj
print(json.dumps({"ready": True, "obj_addr": obj_addr,
                  "ob_type": type_addr, "type_name": type_name,
                  "module": module_name}), flush=True)
input()
"""


@pytest.fixture()
def stale_type_proc() -> Generator[tuple[int, int, int, str, str], None, None]:
    """Process with a freed C-extension object; yields (pid, obj_addr, ob_type, type_name, module)."""
    proc = _spawn(_STALE_TYPE_SCRIPT)
    line = _ready_with_line(proc, timeout=10)
    data = json.loads(line)
    yield (
        proc.pid,
        int(data["obj_addr"], 16),
        int(data["ob_type"], 16),
        data["type_name"],
        data["module"],
    )
    _teardown(proc)
