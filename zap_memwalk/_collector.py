"""Frida-based pymalloc arena/pool collector."""

from __future__ import annotations

import gc
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import keke

from zap_memwalk._agent import _AGENT_JS
from zap_memwalk._model import MemorySnapshot, PoolSnapshot, SizeClassSummary

_NUM_SIZE_CLASSES = 32


def _elf_build_id(path: str) -> str | None:
    """Return the GNU build-id hex string from an ELF file, or None."""
    try:
        with open(path, "rb") as f:
            ident = f.read(16)
        if ident[:4] != b"\x7fELF":
            return None
        ei_class = ident[4]  # 1=32-bit, 2=64-bit
        endian = "<" if ident[5] == 1 else ">"

        with open(path, "rb") as f:
            hdr_size = 64 if ei_class == 2 else 52
            hdr = f.read(hdr_size)
            if len(hdr) < hdr_size:
                return None

            if ei_class == 2:
                e_phoff = struct.unpack_from(endian + "Q", hdr, 32)[0]
                e_phentsize = struct.unpack_from(endian + "H", hdr, 54)[0]
                e_phnum = struct.unpack_from(endian + "H", hdr, 56)[0]
            else:
                e_phoff = struct.unpack_from(endian + "I", hdr, 28)[0]
                e_phentsize = struct.unpack_from(endian + "H", hdr, 42)[0]
                e_phnum = struct.unpack_from(endian + "H", hdr, 44)[0]

            for i in range(min(e_phnum, 64)):
                f.seek(e_phoff + i * e_phentsize)
                ph = f.read(e_phentsize)
                if len(ph) < e_phentsize:
                    break
                p_type = struct.unpack_from(endian + "I", ph, 0)[0]
                if p_type != 4:  # PT_NOTE
                    continue
                if ei_class == 2:
                    p_offset = struct.unpack_from(endian + "Q", ph, 8)[0]
                    p_filesz = struct.unpack_from(endian + "Q", ph, 32)[0]
                else:
                    p_offset = struct.unpack_from(endian + "I", ph, 4)[0]
                    p_filesz = struct.unpack_from(endian + "I", ph, 16)[0]

                f.seek(p_offset)
                note_data = f.read(p_filesz)
                pos = 0
                while pos + 12 <= len(note_data):
                    namesz, descsz, ntype = struct.unpack_from(
                        endian + "III", note_data, pos
                    )
                    pos += 12
                    name = note_data[pos : pos + namesz].rstrip(b"\x00")
                    pos += (namesz + 3) & ~3
                    desc = note_data[pos : pos + descsz]
                    pos += (descsz + 3) & ~3
                    if name == b"GNU" and ntype == 3:  # NT_GNU_BUILD_ID
                        return desc.hex()
    except Exception:
        pass
    return None


def _debuginfod_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(xdg) / "debuginfod_client"


def _run_eu_addr2line(
    offsets: list[int],
    debug_file: str,
    extra_env: dict[str, str] | None = None,
) -> dict[int, str]:
    """Run eu-addr2line -f for a batch of offsets; return {offset: symbol}."""
    eu = shutil.which("eu-addr2line")
    if not eu:
        return {}
    env = {**os.environ, **(extra_env or {})}
    args = [eu, "-f", "-e", debug_file] + [f"0x{off:x}" for off in offsets]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30, env=env)
        lines = r.stdout.splitlines()
        # Output is interleaved: function_name\nfile:line per address
        return {
            off: lines[i * 2].strip()
            for i, off in enumerate(offsets)
            if i * 2 < len(lines) and lines[i * 2].strip() not in ("", "??")
        }
    except Exception:
        return {}


@keke.ktrace("ts")
def _parse(
    raw_pools: list[dict[str, Any]],
    pid: int,
    ts: float,
    pool_size: int,
    arena_size: int,
    py_version: tuple[int, int],
) -> MemorySnapshot:
    """Convert raw pool dicts from the JS agent into a MemorySnapshot."""
    # Disable GC for the duration: allocating ~100k PoolSnapshot objects would
    # otherwise trigger ~140 gen0 sweeps mid-loop (threshold=700).  There are
    # no reference cycles in these objects, so refcount frees them when the old
    # snapshot is replaced; GC only wastes time traversing them.
    gc.disable()
    try:
        buckets: list[list[PoolSnapshot]] = [[] for _ in range(_NUM_SIZE_CLASSES)]

        for p in raw_pools:
            szidx = p["szidx"]
            if szidx >= _NUM_SIZE_CLASSES:
                continue
            block_size = (szidx + 1) * 16
            addr = int(p["address"], 16)
            _free = p.get("freeAddrs") or []
            free_addresses = (
                frozenset(int(a, 16) for a in _free) if _free else frozenset()
            )
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
    finally:
        gc.enable()


class MemWalkCollector:
    """Attach to a running Python process and sample its pymalloc state.

    Usage::

        with MemWalkCollector(pid=1234) as col:
            snap = col.collect()
            type_name, repr_str = col.repr_block(addr) or ("?", "")
    """

    def __init__(self, pid: int, debuginfod: str = "false") -> None:
        if debuginfod not in ("false", "cached", "true"):
            raise ValueError(
                f"debuginfod must be 'false', 'cached', or 'true', got {debuginfod!r}"
            )
        self._pid = pid
        self._debuginfod = debuginfod
        self._session: Any = None
        self._script: Any = None
        self._pool_size: int = 16384
        self._arena_size: int = 1048576
        self._py_version: tuple[int, int] = (0, 0)

    @keke.ktrace()
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

    @keke.ktrace()
    def collect(self) -> MemorySnapshot:
        """Take one full snapshot: all arenas → all pools → free lists."""
        with keke.kev("exports_sync.collect"):
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

    @keke.ktrace()
    def collect_streaming(self) -> MemorySnapshot:
        """Full pool scan via scanAllPools — same as collect() but explicit.

        scanAllPools scans at pool_size granularity so it correctly finds pools
        regardless of how the OS lays out arena VMAs (including Linux coalescing
        arenas with adjacent allocations into larger regions).  freeAddrs are
        deferred to readPool() so the response stays well under the D-Bus 128 MiB
        limit even for very large processes.
        """
        with keke.kev("exports_sync.collect"):
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

    @keke.ktrace("pool_addr")
    def read_pool(self, pool_addr: int) -> dict[str, Any]:
        """Return raw block data for a single pool (for the block view).

        Returns a dict with keys: szidx, refCount, nextoffset, maxnextoffset,
        freeAddrs (list of hex strings), raw (bytes).
        """
        result = self._script.exports_sync.read_pool(
            f"{pool_addr:x}"
        )  # Frida maps → readPool
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        result["raw"] = bytes.fromhex(result.pop("rawHex", ""))
        return dict(result)

    @keke.ktrace("pool_addr")
    def read_pool_snapshot(self, pool_addr: int) -> "PoolSnapshot | None":
        """Read pool header directly and return a PoolSnapshot, or None on failure.

        Used as a fallback when scanAllPools() missed the pool (e.g. rwx range).
        Applies the same maxnextoffset invariant as scanAllPools to reject non-pool memory.
        """
        try:
            raw = self.read_pool(pool_addr)
        except Exception:
            return None
        szidx = raw.get("szidx", 32)
        if szidx >= 32:
            return None
        block_size = (szidx + 1) * 16
        maxnextoffset = raw.get("maxnextoffset", 0)
        # Exact invariant from CPython obmalloc.c: maxnextoffset == POOL_SIZE - block_size
        if maxnextoffset != self._pool_size - block_size:
            return None
        nextoffset = raw.get("nextoffset", 0)
        ref_count = raw.get("refCount", 0)
        total_blocks = (self._pool_size - 48) // block_size
        if ref_count > total_blocks or nextoffset > self._pool_size:
            return None
        return PoolSnapshot(
            address=pool_addr,
            arena_index=0,
            szidx=szidx,
            block_size=block_size,
            ref_count=ref_count,
            nextoffset=nextoffset,
            maxnextoffset=maxnextoffset,
            free_addresses=frozenset(int(a, 16) for a in raw.get("freeAddrs", [])),
        )

    @keke.ktrace("block_addr")
    def repr_block(self, block_addr: int) -> tuple[str, str] | tuple[None, str] | None:
        """Safely repr a live block.

        Returns:
          (type_name, repr_str)  on success
          (None, error_msg)      on failure (error_msg from JS)
          None                   if the RPC call itself fails
        """
        try:
            result = self._script.exports_sync.repr_block(f"{block_addr:x}")
        except Exception as e:
            return None, str(e)
        if result.get("ok"):
            return result["typeName"], result["reprStr"]
        err = result.get("error") or "repr failed"
        return None, err

    def incref(self, obj_addr: int) -> str | None:
        """Increment the refcount of a live PyObject via Py_IncRef.

        Safe for immortal objects (CPython 3.12+): Py_IncRef is a no-op for
        ob_refcnt >= _Py_IMMORTAL_REFCNT.  Only call this on live non-cstr blocks.
        Returns None on success, or an error string if the block was rejected.
        """
        result = self._script.exports_sync.incref_block(f"{obj_addr:x}")
        if not result.get("ok"):
            return result.get("error") or "incref failed"
        return None

    def decref(self, obj_addr: int) -> None:
        """Decrement the refcount of a PyObject previously pinned by incref.

        Safe for immortal objects: Py_DecRef is a no-op for immortal refcounts.
        """
        self._script.exports_sync.decref_block(f"{obj_addr:x}")

    def ping_gil(self) -> bool:
        """Acquire and immediately release the GIL; return True on success.

        Used as a heartbeat: if the target's GIL is stuck this call blocks.
        Returns False if the RPC call fails for any reason.
        """
        try:
            result = self._script.exports_sync.ping_gil()
            return bool(result.get("ok"))
        except Exception:
            return False

    @keke.ktrace()
    def list_arenas(self) -> list[dict[str, Any]]:
        """Full scan → one entry per arena: {arenaIndex, base (hex str), poolCount}.

        The response is small (~1 KB for thousands of arenas) even for large
        processes.  Use list_pools_in_arena() for per-arena pool detail.
        """
        result = self._script.exports_sync.list_arenas()
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        return list(result["arenas"])

    @keke.ktrace("arena_base")
    def list_pools_in_arena(self, arena_base: int) -> list[dict[str, Any]]:
        """Scan one arena_size window; return pool-header dicts (no freeAddrs).

        arena_base must be the arena_size-aligned address from list_arenas().
        """
        result = self._script.exports_sync.list_pools_in_arena(f"{arena_base:x}")
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        return list(result["pools"])

    @keke.ktrace()
    def list_pool_subset(self, first_addr: int, half_open_last_addr: int) -> bytes:
        """Read raw bytes for [first_addr, half_open_last_addr) from target memory.

        Returns a bytes object; split by block_size to get per-block raw data.
        Range must be <= 65536 bytes.
        """
        result = self._script.exports_sync.list_pool_subset(
            f"{first_addr:x}", f"{half_open_last_addr:x}"
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error"))
        return bytes.fromhex(result["hex"])

    @keke.ktrace()
    def resolve_type_names(self, type_ptrs: list[int]) -> dict[int, str]:
        """Bulk-resolve ob_type pointer values → tp_name strings (no GIL)."""
        if not type_ptrs:
            return {}
        unique = list({p for p in type_ptrs if p})
        hex_list = [f"{p:x}" for p in unique]
        result = self._script.exports_sync.resolve_type_names(hex_list)
        return {int(k, 16): v for k, v in result.items()}

    @keke.ktrace()
    def symbolize_addresses(self, ptrs: list[int]) -> dict[int, dict[str, Any] | None]:
        """Bulk-resolve addresses → {module, path, offset, symbol} or None (no GIL)."""
        if not ptrs:
            return {}
        unique = list({p for p in ptrs if p})
        hex_list = [f"{p:x}" for p in unique]
        result = self._script.exports_sync.symbolize_addresses(hex_list)
        out: dict[int, dict[str, Any] | None] = {
            int(k, 16): (dict(v) if v else None) for k, v in result.items()
        }
        if self._debuginfod != "false":
            self._enhance_with_debuginfod(out)
        return out

    @keke.ktrace()
    def peek_cstrings(
        self, ptrs: list[int], max_len: int = 64, min_len: int = 4
    ) -> dict[int, str | None]:
        """Peek at addresses for null-terminated printable ASCII strings (no GIL)."""
        if not ptrs:
            return {}
        unique = list({p for p in ptrs if p})
        hex_list = [f"{p:x}" for p in unique]
        result = self._script.exports_sync.peek_cstrings(hex_list, max_len, min_len)
        return {int(k, 16): (v if v else None) for k, v in result.items()}

    @keke.ktrace()
    def _enhance_with_debuginfod(
        self, results: dict[int, dict[str, Any] | None]
    ) -> None:
        """Fill missing symbol names in-place using eu-addr2line + debuginfod."""
        # Group addresses-without-symbols by module path.
        by_path: dict[str, list[tuple[int, int]]] = {}  # path -> [(addr, offset)]
        for addr, info in results.items():
            if info is None or info.get("symbol") or not info.get("path"):
                continue
            by_path.setdefault(info["path"], []).append((addr, info["offset"]))

        for mod_path, addr_offsets in by_path.items():
            debug_file = self._resolve_debug_file(mod_path)
            if debug_file is None:
                continue
            offsets = [off for _, off in addr_offsets]
            # cached mode: pass empty DEBUGINFOD_URLS so eu-addr2line never fetches
            extra_env: dict[str, str] | None = (
                {"DEBUGINFOD_URLS": ""} if self._debuginfod == "cached" else None
            )
            sym_map = _run_eu_addr2line(offsets, debug_file, extra_env)
            for addr, off in addr_offsets:
                if off in sym_map:
                    existing = results[addr]
                    if existing is not None:
                        results[addr] = {**existing, "symbol": sym_map[off]}

    def _resolve_debug_file(self, mod_path: str) -> str | None:
        """Return the file path to pass to eu-addr2line, or None to skip."""
        if self._debuginfod == "cached":
            build_id = _elf_build_id(mod_path)
            if not build_id:
                return None
            cached = _debuginfod_cache_dir() / build_id / "debuginfo"
            return str(cached) if cached.exists() else None
        # "true": use the original binary; eu-addr2line fetches via DEBUGINFOD_URLS
        return mod_path if os.path.exists(mod_path) else None

    def get_range_protection(self, addr: int) -> str | None:
        """Return the memory protection string for the range containing addr, or None."""
        result = self._script.exports_sync.get_range_protection(f"{addr:x}")
        return result if isinstance(result, str) else None

    def get_type_name(self, block_addr: int) -> str:
        """Read ob_type.tp_name for a block without GIL (best-effort)."""
        result = self._script.exports_sync.repr_block(f"{block_addr:x}")
        return str(result.get("typeName", "?"))

    @keke.ktrace()
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
