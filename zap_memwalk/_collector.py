"""Frida-based pymalloc arena/pool collector."""

from __future__ import annotations

import sys
import time
from typing import Any

from zap_memwalk._agent import _AGENT_JS
from zap_memwalk._model import MemorySnapshot, PoolSnapshot, SizeClassSummary

_NUM_SIZE_CLASSES = 32


def _parse(
    raw_pools: list[dict[str, Any]],
    pid: int,
    ts: float,
    pool_size: int,
    arena_size: int,
    py_version: tuple[int, int],
) -> MemorySnapshot:
    """Convert raw pool dicts from the JS agent into a MemorySnapshot."""
    # Build a SizeClassSummary for each size class (0-31)
    buckets: list[list[PoolSnapshot]] = [[] for _ in range(_NUM_SIZE_CLASSES)]

    for p in raw_pools:
        szidx = p["szidx"]
        if szidx >= _NUM_SIZE_CLASSES:
            continue
        block_size = (szidx + 1) * 16
        addr = int(p["address"], 16)
        free_addresses = frozenset(int(a, 16) for a in p["freeAddrs"])
        snap = PoolSnapshot(
            address=addr,
            arena_index=p["arenaIndex"],
            szidx=szidx,
            block_size=block_size,
            ref_count=p["refCount"],
            nextoffset=p["nextoffset"],
            maxnextoffset=p["maxnextoffset"],
            free_addresses=free_addresses,
        )
        buckets[szidx].append(snap)

    size_classes = [
        SizeClassSummary(szidx=i, block_size=(i + 1) * 16, pools=buckets[i])
        for i in range(_NUM_SIZE_CLASSES)
    ]

    return MemorySnapshot(
        pid=pid,
        ts=ts,
        pool_size=pool_size,
        arena_size=arena_size,
        py_version=py_version,
        size_classes=size_classes,
    )


class MemWalkCollector:
    """Attach to a running Python process and sample its pymalloc state.

    Usage::

        with MemWalkCollector(pid=1234) as col:
            snap = col.collect()
            type_name, repr_str = col.repr_block(addr) or ("?", "")
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._session: Any = None
        self._script: Any = None
        self._pool_size: int = 16384
        self._arena_size: int = 1048576
        self._py_version: tuple[int, int] = (0, 0)

    def __enter__(self) -> "MemWalkCollector":
        import frida  # noqa: PLC0415

        self._session = frida.attach(self._pid)
        self._script = self._session.create_script(_AGENT_JS)
        self._script.on("message", self._on_message)
        self._script.load()
        result = self._script.exports_sync.setup()
        if not result.get("ok"):
            raise RuntimeError(f"agent setup failed: {result.get('error')}")
        self._pool_size = result.get("poolSize", 16384)
        self._arena_size = result.get("arenaSize", 1048576)
        self._py_version = (result.get("pyMajor", 0), result.get("pyMinor", 0))
        return self

    def _on_message(self, msg: dict[str, object], _data: object) -> None:
        if msg.get("type") == "error":
            print(f"[zap-memwalk] {msg.get('description', msg)}", file=sys.stderr)

    def collect(self) -> MemorySnapshot:
        """Take one full snapshot: all arenas → all pools → free lists."""
        result = self._script.exports_sync.collect()
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        return _parse(
            result["pools"],
            self._pid,
            time.monotonic(),
            self._pool_size,
            self._arena_size,
            self._py_version,
        )

    def read_pool(self, pool_addr: int) -> dict[str, Any]:
        """Return raw block data for a single pool (for the block view).

        Returns a dict with keys: szidx, refCount, nextoffset, maxnextoffset,
        freeAddrs (list of hex strings), raw (bytes).
        """
        result = self._script.exports_sync.read_pool(f"{pool_addr:x}")  # Frida maps → readPool
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        result["raw"] = bytes.fromhex(result.pop("rawHex", ""))
        return result

    def repr_block(self, block_addr: int) -> tuple[str, str] | None:
        """Safely repr a live block; returns (type_name, repr_str) or None.

        Bumps the refcount before calling PyObject_Repr to guard against GC.
        Returns None if the repr fails (block freed, repr raised, etc.).
        The type_name is always populated even on failure.
        """
        result = self._script.exports_sync.repr_block(f"{block_addr:x}")
        if result.get("ok"):
            return result["typeName"], result["reprStr"]
        # On failure still return typeName so the TUI can show it
        return None

    def resolve_type_names(self, type_ptrs: list[int]) -> dict[int, str]:
        """Bulk-resolve ob_type pointer values → tp_name strings (no GIL)."""
        if not type_ptrs:
            return {}
        unique = list({p for p in type_ptrs if p})
        hex_list = [f"{p:x}" for p in unique]
        result = self._script.exports_sync.resolve_type_names(hex_list)
        return {int(k, 16): v for k, v in result.items()}

    def get_type_name(self, block_addr: int) -> str:
        """Read ob_type.tp_name for a block without GIL (best-effort)."""
        result = self._script.exports_sync.repr_block(f"{block_addr:x}")
        return result.get("typeName", "?")

    def __exit__(self, *_: object) -> None:
        if self._script is not None:
            try:
                self._script.unload()
            except Exception:
                pass
        if self._session is not None:
            try:
                self._session.detach()
            except Exception:
                pass
