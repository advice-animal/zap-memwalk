"""Interactive TUI for zap-memwalk using rich.Live.

Two hierarchy modes (Tab to toggle):
  size mode:  size-class → pools in that size class → blocks
  arena mode: arena       → pools in that arena       → blocks

Navigation:
  Top view  ──Enter──>  Pool view  ──Enter──>  Block view
      ^                     ^                      ^
      └──── Escape / b ─────┘                      │
                  └──────────── Escape / b ─────────┘

In the block view, press  r  on a selected (live) block to repr it,
or  R  to repr every visible typed block at once.
"""

from __future__ import annotations

import queue
import shutil
import struct
import sys
import time
from enum import Enum, auto
from typing import Any

import keke
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from zap_memwalk._collector import MemWalkCollector
from zap_memwalk._fallback_repr import (
    cstr_hint as _cstr_hint,
)
from zap_memwalk._fallback_repr import (
    extract_cstr as _extract_cstr,
)
from zap_memwalk._fallback_repr import (
    fallback_repr_from_raw as _fallback_repr_from_raw,
)
from zap_memwalk._fallback_repr import (
    hex_bytes as _hex_bytes,
)
from zap_memwalk._model import (
    BlockAgeTracker,
    BlockState,
    MemorySnapshot,
    PoolSnapshot,
)

_POOL_OVERHEAD = 48

# Types that are always GC-tracked regardless of Python version.
_ALWAYS_GC = frozenset(
    {
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "frame",
        "cell",
        "module",
        "memoryview",
        "enumerate",
        "function",
        "dict_keys",
        "dict_values",
        "dict_items",
        "generator",
        "coroutine",
        "async_generator",
        "classmethod",
        "staticmethod",
        "property",
        "weakref",
        "method_descriptor",
        "member_descriptor",
        "getset_descriptor",
        "wrapper_descriptor",
        "builtin_function_or_method",
    }
)
# Types that are never GC-tracked.
_NEVER_GC = frozenset(
    {
        "float",
        "int",
        "str",
        "bytes",
        "bool",
        "NoneType",
        "complex",
        "range",
        "bytearray",
        "cstr",  # internal sentinel for raw C-string blocks
    }
)


def _is_gc_type(eff_type: str, py_ver: tuple[int, int]) -> bool | None:
    """Return True/False for known types, None for unknown (caller uses pointer heuristic)."""
    if eff_type in _ALWAYS_GC:
        return True
    if eff_type in _NEVER_GC:
        return False
    if eff_type == "code":
        # CPython 3.13 removed Py_TPFLAGS_HAVE_GC from PyCode_Type.
        return py_ver < (3, 13)
    if eff_type == "method":
        return py_ver >= (3, 11)
    if eff_type == "slice":
        return py_ver >= (3, 13)
    return None


def _augmented_free_set(
    raw: bytes,
    pool_addr: int,
    block_size: int,
    nextoffset: int,
    freeaddrs_list: list[str],
) -> frozenset[int]:
    """Build the free-block set, augmenting the explicit freelist with a heuristic.

    The Frida collector walks pool->freeblock but can miss blocks due to timing
    races.  Any born block whose first 8 bytes look like a pool-internal freelist
    pointer is also treated as free.

    A value v at block+0 is considered a freelist pointer when all of:
      - v != 0
      - v >= pool_addr + _POOL_OVERHEAD  (points into the block area)
      - v < pool_addr + nextoffset       (within born range)
      - (v - pool_addr - _POOL_OVERHEAD) % block_size == 0  (block-aligned)
    """
    explicit = frozenset(int(a, 16) for a in freeaddrs_list)
    if len(raw) < 8 or block_size <= 0:
        return explicit
    extra: set[int] = set()
    base = pool_addr + _POOL_OVERHEAD
    for off in range(_POOL_OVERHEAD, nextoffset, block_size):
        if off + 8 > len(raw):
            break
        addr = pool_addr + off
        if addr in explicit:
            continue
        v = struct.unpack_from("<Q", raw, off)[0]
        if (
            v != 0
            and v >= base
            and v < pool_addr + nextoffset
            and (v - base) % block_size == 0
        ):
            extra.add(addr)
    return explicit | extra


def _ob_type_candidates(blk: bytes) -> tuple[int, int]:
    """Return (ptr_at_8, ptr_at_24) from raw block bytes.

    ptr_at_8  is ob_type for non-GC objects (no PyGC_Head prefix).
    ptr_at_24 is ob_type for GC-tracked objects (16-byte PyGC_Head at block+0).
    """
    ptr8 = struct.unpack_from("<Q", blk, 8)[0] if len(blk) >= 16 else 0
    ptr24 = struct.unpack_from("<Q", blk, 24)[0] if len(blk) >= 32 else 0
    return ptr8, ptr24


def _pick_ob_type(ptr8: int, ptr24: int, cache: dict[int, str]) -> tuple[int, str]:
    """Pick the ob_type ptr whose type name resolves (is not '' or '?').

    Returns (winning_ptr, type_name).  Falls back to (ptr8, name8) when
    neither resolves — caller gets an unresolved pointer to show as raw hex.
    """
    name8 = cache.get(ptr8, "?") if ptr8 else "?"
    if name8 not in ("", "?"):
        return ptr8, name8
    name24 = cache.get(ptr24, "?") if ptr24 else "?"
    if name24 not in ("", "?"):
        return ptr24, name24
    return ptr8, name8


class _View(Enum):
    SIZE_CLASS = auto()
    POOL = auto()
    BLOCK = auto()
    BLOCK_DETAIL = auto()


def _fill_bar(fill_pct: float, width: int = 20) -> Text:
    filled = round(fill_pct / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    if fill_pct >= 90:
        style = "bold red"
    elif fill_pct >= 60:
        style = "red"
    elif fill_pct >= 30:
        style = "yellow"
    else:
        style = "green"
    return Text(bar, style=style)


def _pct_text(pct: float) -> Text:
    s = f"{pct:5.1f}%"
    if pct >= 90:
        return Text(s, style="bold red")
    elif pct >= 60:
        return Text(s, style="red")
    elif pct >= 30:
        return Text(s, style="yellow")
    return Text(s, style="green")


class MemWalkTUI:
    """Three-level interactive TUI: size-class → pool → block."""

    def __init__(
        self,
        collector: MemWalkCollector,
        interval: float = 1.0,
    ) -> None:
        import threading

        self._col = collector
        self._interval = interval
        self._snap: MemorySnapshot | None = None
        self._refresh_parity: bool = False  # toggled after each completed refresh
        # Prevents concurrent Frida RPC from the background refresh thread and
        # an explicit 'R' keypress.  Background thread uses acquire(blocking=False)
        # to skip a cycle if a manual refresh is already in progress.
        self._collect_lock = threading.Lock()
        self._age = BlockAgeTracker()

        self._hier_mode: str = "size"  # "size" | "arena"
        self._view = _View.SIZE_CLASS
        self._cursor = 0  # row cursor in the current view
        self._vp_start = 0  # viewport first row

        # Selected size class / pool / arena for drill-down
        self._sel_szidx: int | None = None
        self._sel_arena_idx: int | None = None  # arena_index of selected ArenaSummary
        self._sel_pool: PoolSnapshot | None = None

        # Block view state
        self._pool_raw: dict[str, Any] | None = None  # from read_pool()
        self._repr_lines: list[tuple[int, str]] = []  # (block_addr, repr_text)
        self._type_names: dict[int, str] = {}  # block_addr -> type name
        self._name_hints: dict[
            int, str
        ] = {}  # block_addr -> identifying string (code/module)
        self._ob_type_syms: dict[
            int, str
        ] = {}  # ob_type_ptr -> formatted symbol string
        self._type_ptr_cache: dict[
            int, str
        ] = {}  # ob_type_ptr -> type name (RPC cache)
        self._rpc_lookups: int = 0  # cumulative Frida RPC lookup count
        self._sel_pool_protection: str | None = None  # memory prot of selected pool

        # Saved cursors so re-entering a previously visited level restores position.
        # Keys: pool address (for block view), szidx or arena_index (for pool view).
        self._saved_block_cursor: dict[int, int] = {}  # pool_addr → block cursor
        self._saved_pool_cursor: dict[
            tuple[str, int], int
        ] = {}  # (mode, key) → pool cursor

        # Auto-incref: address currently pinned with Py_IncRef (None if none).
        # Only set on live non-cstr blocks; managed by _pin/_unpin.
        self._pinned_addr: int | None = None

        # Block detail view: symbolize results for pointer classification.
        # Populated by _classify_detail_ptrs() on entry to BLOCK_DETAIL.
        self._detail_sym_results: dict[int, Any] = {}
        self._detail_cstr_results: dict[int, str | None] = {}

        # Referrer-search mode (activated by 'g' in block/block-detail view).
        # _referrer_results: None = not in mode; list = found block addresses.
        # _referrer_target: the PyObject* address we searched for.
        self._referrer_results: list[int] | None = None
        self._referrer_idx: int = 0
        self._referrer_target: int | None = None

        # Jump-to-address input state (activated by '/')
        self._jump_buf: str | None = (
            None  # None = not in jump mode; str = current input
        )
        self._jump_err: str = ""  # shown after a failed jump

        self._keys: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._flash_msg: str = ""
        self._flash_until: float = 0.0

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_size_class_view(self, height: int) -> Table:
        t = Table(
            box=box.SIMPLE_HEAD,
            show_edge=False,
            show_footer=False,
            highlight=False,
        )
        t.add_column("size cls", justify="right", style="cyan", no_wrap=True)
        t.add_column("pools", justify="right", no_wrap=True)
        t.add_column("used", justify="right", no_wrap=True)
        t.add_column("total", justify="right", no_wrap=True)
        t.add_column("fill%", justify="right", no_wrap=True)
        t.add_column("fill", no_wrap=True, min_width=16)

        if self._snap is None:
            return t

        classes = [sc for sc in self._snap.size_classes if sc.pool_count > 0]
        total_rows = len(classes)

        self._vp_start = _clamp_viewport(
            self._vp_start, self._cursor, height, total_rows
        )

        shown = 0
        for row_i, sc in enumerate(classes[self._vp_start : self._vp_start + height]):
            abs_i = row_i + self._vp_start
            cursor_mark = "▶" if abs_i == self._cursor else " "
            t.add_row(
                f"{cursor_mark} {sc.block_size}",
                str(sc.pool_count),
                f"{sc.used_blocks:,}",
                f"{sc.total_blocks:,}",
                _pct_text(sc.fill_pct),
                _fill_bar(sc.fill_pct, 16),
            )
            shown += 1
        for _ in range(shown, height):
            t.add_row("", "", "", "", "", "")
        return t

    def _render_pool_list(self, pools: list[PoolSnapshot], height: int) -> Table:
        """Shared pool-list renderer used by both size-class and arena pool views."""
        t = Table(box=box.SIMPLE_HEAD, show_edge=False)
        t.add_column("pool address", style="cyan", no_wrap=True)
        t.add_column("sz", justify="right", style="dim", no_wrap=True)
        t.add_column("used", justify="right", no_wrap=True)
        t.add_column("total", justify="right", no_wrap=True)
        t.add_column("fill%", justify="right", no_wrap=True)
        t.add_column("fill", no_wrap=True, min_width=16)

        total_rows = len(pools)
        self._vp_start = _clamp_viewport(
            self._vp_start, self._cursor, height, total_rows
        )

        shown = 0
        for row_i, pool in enumerate(pools[self._vp_start : self._vp_start + height]):
            abs_i = row_i + self._vp_start
            cursor_mark = "▶" if abs_i == self._cursor else " "
            t.add_row(
                f"{cursor_mark} 0x{pool.address:016x}",
                str(pool.block_size),
                str(pool.ref_count),
                str(pool.total_blocks),
                _pct_text(pool.fill_pct),
                _fill_bar(pool.fill_pct, 16),
            )
            shown += 1
        for _ in range(shown, height):
            t.add_row("", "", "", "", "", "")
        return t

    def _render_pool_view(self, height: int) -> Table:
        if self._snap is None or self._sel_szidx is None:
            return self._render_pool_list([], height)
        return self._render_pool_list(
            self._snap.size_classes[self._sel_szidx].pools, height
        )

    def _render_arena_view(self, height: int) -> Table:
        t = Table(
            box=box.SIMPLE_HEAD,
            show_edge=False,
            show_footer=False,
            highlight=False,
        )
        t.add_column("arena", justify="right", style="cyan", no_wrap=True)
        t.add_column("base address", style="dim", no_wrap=True)
        t.add_column("pools", justify="right", no_wrap=True)
        t.add_column("used", justify="right", no_wrap=True)
        t.add_column("total", justify="right", no_wrap=True)
        t.add_column("fill%", justify="right", no_wrap=True)
        t.add_column("fill", no_wrap=True, min_width=16)

        if self._snap is None:
            return t

        arenas = self._snap.arenas
        total_rows = len(arenas)
        self._vp_start = _clamp_viewport(
            self._vp_start, self._cursor, height, total_rows
        )

        shown = 0
        for row_i, arena in enumerate(arenas[self._vp_start : self._vp_start + height]):
            abs_i = row_i + self._vp_start
            cursor_mark = "▶" if abs_i == self._cursor else " "
            t.add_row(
                f"{cursor_mark} {arena.arena_index}",
                f"0x{arena.base_address:016x}",
                str(arena.pool_count),
                f"{arena.used_blocks:,}",
                f"{arena.total_blocks:,}",
                _pct_text(arena.fill_pct),
                _fill_bar(arena.fill_pct, 16),
            )
            shown += 1
        for _ in range(shown, height):
            t.add_row("", "", "", "", "", "", "")
        return t

    def _render_arena_pool_view(self, height: int) -> Table:
        if self._snap is None or self._sel_arena_idx is None:
            return self._render_pool_list([], height)
        arena = next(
            (a for a in self._snap.arenas if a.arena_index == self._sel_arena_idx),
            None,
        )
        return self._render_pool_list(arena.pools if arena else [], height)

    def _render_block_view(self, height: int) -> Table:
        import struct

        t = Table(box=box.SIMPLE_HEAD, show_edge=False)
        t.add_column("", width=2, no_wrap=True)
        t.add_column("offset", justify="right", style="dim", no_wrap=True)
        t.add_column("address", style="cyan", no_wrap=True)
        t.add_column("state", no_wrap=True, min_width=6)
        t.add_column("content", no_wrap=True, max_width=50)

        if self._sel_pool is None or self._pool_raw is None:
            return t

        pool = self._sel_pool
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", pool.szidx)
        nextoffset = self._pool_raw.get("nextoffset", pool.nextoffset)
        block_size = (szidx + 1) * 16

        free_set = _augmented_free_set(
            raw,
            pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )
        now = time.monotonic()
        py_ver = self._snap.py_version if self._snap else (3, 10)
        _IMMORTAL = 0xFFFF_FFFF

        repr_map = {a: txt for a, txt in self._repr_lines}

        # Build block list
        blocks = []
        for off in range(_POOL_OVERHEAD, len(raw), block_size):
            addr = pool.address + off
            if off >= nextoffset:
                state = BlockState.UNBORN
            elif addr in free_set:
                state = BlockState.FREE
            else:
                state = BlockState.LIVE
            blocks.append((off, addr, state))

        total_rows = len(blocks)
        self._vp_start = _clamp_viewport(
            self._vp_start, self._cursor, height, total_rows
        )

        _shown_blocks = 0
        for row_i, (off, addr, state) in enumerate(
            blocks[self._vp_start : self._vp_start + height]
        ):
            abs_i = row_i + self._vp_start
            cursor_mark = "▶" if abs_i == self._cursor else " "
            blk_bytes = raw[off : off + block_size]
            raw_type = self._type_names.get(addr, "")
            eff_type = (
                raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)
            )

            if state == BlockState.UNBORN:
                state_text = Text("unborn", style="dim")
                content = Text(_hex_bytes(blk_bytes[:8]), style="dim")

            elif eff_type == "cstr":
                # Raw byte content — no PyObject header structure.
                # Show the string itself, not refcount/type.
                style = "bright_black" if state == BlockState.FREE else ""
                state_text = Text(
                    "free" if state == BlockState.FREE else "live", style=style
                )
                cstr_val = _extract_cstr(blk_bytes)
                cstr_display = repr(cstr_val) if cstr_val is not None else "(empty)"
                content = Text(f"cstr  {cstr_display}", style=style, no_wrap=True)

            else:
                # PyObject layout: state shows refcount; content shows type + repr hint
                if state == BlockState.FREE:
                    state_text = Text("free", style="bright_black")
                    base_style = "bright_black"
                else:
                    color = self._age.color(addr, now)
                    base_style = color
                    # Refcount in the state column for live blocks.
                    gc_known = _is_gc_type(eff_type, py_ver)
                    if gc_known is None:
                        ptr8, ptr24 = _ob_type_candidates(blk_bytes)
                        gc_known = (
                            ptr24 in self._type_ptr_cache
                            and self._type_ptr_cache.get(ptr24, "?") not in ("", "?")
                            and (
                                self._type_ptr_cache.get(ptr8, "?") in ("", "?")
                                or not ptr8
                            )
                        )
                    rc_off = 16 if gc_known else 0
                    if len(blk_bytes) >= rc_off + 8:
                        rc_raw = struct.unpack_from("<Q", blk_bytes, rc_off)[0]
                        rc_label = (
                            "immortal" if rc_raw >= _IMMORTAL else f"refs={rc_raw}"
                        )
                    else:
                        rc_label = "live"
                    state_text = Text(rc_label, style=color)

                # Value hint: name_hint first, then fallback repr, always attempted
                value_hint = ""
                if addr in repr_map:
                    full = repr_map[addr]
                    bracket = full.find("]:")
                    value_hint = (
                        full[bracket + 2 :].strip()[:40] if bracket >= 0 else full[:40]
                    )
                elif addr in self._name_hints:
                    value_hint = self._name_hints[addr]
                else:
                    fallback = _fallback_repr_from_raw(blk_bytes, eff_type, py_ver)
                    if fallback:
                        value_hint = fallback[:40]

                parts = [p for p in (eff_type, value_hint) if p]
                content = Text("  ".join(parts), style=base_style, no_wrap=True)

            t.add_row(
                cursor_mark,
                f"+{off:04x}",
                f"0x{addr:x}",
                state_text,
                content,
            )
            _shown_blocks += 1
        for _ in range(_shown_blocks, height):
            t.add_row("", "", "", "", "")
        return t

    def _make_header(self) -> Text:
        snap = self._snap
        if snap is None:
            return Text("zap-memwalk — connecting…", style="bold")
        pid_info = f"PID {snap.pid}"
        ver_info = f"Python {snap.py_version[0]}.{snap.py_version[1]}"
        pool_info = f"{snap.pool_size // 1024}KiB pools"

        if self._view == _View.BLOCK_DETAIL:
            # Compute block address for the breadcrumb
            block_addr_str = ""
            if self._sel_pool is not None and self._pool_raw is not None:
                raw = bytes(self._pool_raw.get("raw", b"") or b"")
                szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
                blk = (szidx + 1) * 16
                blks = list(range(_POOL_OVERHEAD, len(raw), blk))
                if self._cursor < len(blks):
                    baddr = self._sel_pool.address + blks[self._cursor]
                    btype = self._type_names.get(baddr, "")
                    block_addr_str = f"0x{baddr:x}" + (
                        f"  {btype}" if btype and btype != "?" else ""
                    )
            view_crumb = (
                f"pool 0x{self._sel_pool.address:x}  →  {block_addr_str}"
                if self._sel_pool and block_addr_str
                else "block detail"
            )
        elif self._hier_mode == "arena":
            if self._view == _View.SIZE_CLASS:
                view_crumb = "arenas"
            elif self._view == _View.POOL:
                if self._sel_arena_idx is not None and snap is not None:
                    arena = next(
                        (
                            a
                            for a in snap.arenas
                            if a.arena_index == self._sel_arena_idx
                        ),
                        None,
                    )
                    view_crumb = (
                        f"arena {arena.arena_index}  (base 0x{arena.base_address:x})"
                        if arena
                        else f"arena {self._sel_arena_idx}"
                    )
                else:
                    view_crumb = "arena"
            else:  # BLOCK
                view_crumb = (
                    f"pool 0x{self._sel_pool.address:x}"
                    f"  arena {self._sel_pool.arena_index}"
                    f"  sz{self._sel_pool.szidx} ({self._sel_pool.block_size}B)"
                    + (
                        f"  [{self._sel_pool_protection}]"
                        if self._sel_pool_protection
                        else ""
                    )
                    if self._sel_pool
                    else ""
                )
        else:
            view_crumb = {
                _View.SIZE_CLASS: "size classes",
                _View.POOL: f"size-class {self._sel_szidx} ({((self._sel_szidx or 0) + 1) * 16}B pools)",
                _View.BLOCK: (
                    f"pool 0x{self._sel_pool.address:x}"
                    f"  sz{self._sel_pool.szidx} ({self._sel_pool.block_size}B)"
                    + (
                        f"  [{self._sel_pool_protection}]"
                        if self._sel_pool_protection
                        else ""
                    )
                    if self._sel_pool
                    else ""
                ),
            }[self._view]

        if self._jump_buf is not None:
            second_line = f"jump to address: {self._jump_buf}█  (0x… hex or decimal  Enter=go  Esc=cancel)"
        elif self._referrer_results is not None:
            if self._referrer_results:
                tgt = f"0x{self._referrer_target:x}" if self._referrer_target else "?"
                second_line = (
                    f"referrer mode ({self._referrer_idx + 1}/{len(self._referrer_results)}) "
                    f"for {tgt}  n/p=next/prev  b/Esc=exit referrer mode  g=re-search  q=quit"
                )
            else:
                tgt = f"0x{self._referrer_target:x}" if self._referrer_target else "?"
                second_line = (
                    f"no referrers found for {tgt}  b/Esc=exit  g=re-search  q=quit"
                )
        elif self._jump_err:
            second_line = f"↑↓=move  Enter=drill  b/Esc=back  Tab=mode  /=jump  r=repr  g=refs  q=quit  [{self._jump_err}]"
        elif self._view == _View.BLOCK_DETAIL:
            base = "↑↓=scroll  b/Esc=back  r=repr  g=refs  q=quit"
            second_line = (
                f"{base}  [lookups: {self._rpc_lookups}]"
                if self._rpc_lookups > 0
                else base
            )
        else:
            base = "↑↓=move  Enter=drill  b/Esc=back  Tab=mode  /=jump  r=repr  g=refs  q=quit"
            if self._rpc_lookups > 0:
                second_line = f"{base}  [lookups: {self._rpc_lookups}]"
            else:
                second_line = base

        # Flash message (j/k boundary) overrides status line briefly.
        if (
            self._jump_buf is None
            and self._referrer_results is None
            and self._flash_msg
            and time.monotonic() < self._flash_until
        ):
            second_line = self._flash_msg
            sl_style = "yellow"
        else:
            sl_style = "bold yellow" if self._jump_buf is not None else "dim"

        dot = ("■", "green") if self._refresh_parity else ("■", "dim")
        return Text.assemble(
            ("zap-memwalk", "bold"),
            " ",
            dot,
            "  ",
            (pid_info, "cyan"),
            "  ",
            (ver_info, "dim"),
            "  ",
            (pool_info, "dim"),
            "  │  ",
            (view_crumb, "bold cyan"),
            "\n",
            (second_line, sl_style),
        )

    def _make_detail_panel(self) -> Group:
        """Fixed 6-line panel at the bottom of the block view.

        Lines: rule / fallback+rc+gc / repr / type+symbol / ctypes / activities.
        """
        _empty = Group(
            Rule(style="dim"), Text(""), Text(""), Text(""), Text(""), Text("")
        )
        if self._sel_pool is None or self._pool_raw is None:
            return _empty

        import struct

        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        free_set = _augmented_free_set(
            raw,
            self._sel_pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )
        py_ver = self._snap.py_version if self._snap else (3, 10)
        _IMMORTAL = 0xFFFF_FFFF

        blocks = [
            (off, self._sel_pool.address + off)
            for off in range(_POOL_OVERHEAD, len(raw), block_size)
        ]
        if self._cursor >= len(blocks):
            return _empty

        off, addr = blocks[self._cursor]
        blk_bytes = raw[off : off + block_size]
        raw_type = self._type_names.get(addr, "")
        eff_type = raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)
        state_str = (
            "unborn" if off >= nextoffset else ("free" if addr in free_set else "live")
        )

        title_parts = [f"0x{addr:x}"]
        if eff_type:
            title_parts.append(f"[dim]{eff_type}[/dim]")
        title_parts.append(state_str)
        rule = Rule(title="  ".join(title_parts), style="dim")

        # ── line 1: fallback repr + rc + gc ───────────────────────────────────
        is_gc = False
        if state_str == "unborn":
            line1 = Text("(unborn — no object)", style="dim")
        elif eff_type == "cstr":
            cstr_val = _extract_cstr(blk_bytes)
            line1 = Text(
                f"cstr: {repr(cstr_val) if cstr_val is not None else '(empty)'}",
                style="dim",
                no_wrap=True,
            )
        else:
            gc_known = _is_gc_type(eff_type, py_ver)
            if gc_known is None:
                ptr8, ptr24 = _ob_type_candidates(blk_bytes)
                gc_known = (
                    ptr24 in self._type_ptr_cache
                    and self._type_ptr_cache.get(ptr24, "?") not in ("", "?")
                    and (self._type_ptr_cache.get(ptr8, "?") in ("", "?") or not ptr8)
                )
                if (
                    not gc_known
                    and ptr24 >= 0x1_0000
                    and bool(self._detail_sym_results.get(ptr24))
                ):
                    # ptr24 was symbolized by _classify_detail_ptrs (GC ob_type candidate)
                    # but ptr8 wasn't a known type and wasn't symbolized → block has GC header.
                    if not self._detail_sym_results.get(
                        ptr8
                    ) and self._type_ptr_cache.get(ptr8, "?") in ("", "?"):
                        gc_known = True
            is_gc = gc_known
            rc_off = 16 if is_gc else 0
            rc_str = ""
            if state_str == "live" and len(blk_bytes) >= rc_off + 8:
                rc_raw = struct.unpack_from("<Q", blk_bytes, rc_off)[0]
                rc_str = "immortal" if rc_raw >= _IMMORTAL else f"refs={rc_raw}"
            gc_str = "gc=yes" if is_gc else "gc=no"
            name_hint = self._name_hints.get(addr, "")
            fallback = (
                name_hint or _fallback_repr_from_raw(blk_bytes, raw_type, py_ver) or ""
            )
            parts = [p for p in (fallback, rc_str, gc_str) if p]
            line1 = Text(
                "  ".join(parts) if parts else "(no inline data)",
                style="dim",
                no_wrap=True,
            )

        # ── line 2: repr from RPC ──────────────────────────────────────────────
        repr_map = {a: t for a, t in self._repr_lines}
        if addr in repr_map:
            full = repr_map[addr]
            bracket = full.find("]:")
            repr_str = full[bracket + 2 :].strip() if bracket >= 0 else full
            line2 = Text(
                f"repr: {repr_str}", style="dim", no_wrap=True, overflow="crop"
            )
        else:
            line2 = Text("repr: (press r)", style="dim")

        # ── line 3: type guess + ob_type symbol ───────────────────────────────
        ptr8, ptr24 = _ob_type_candidates(blk_bytes)
        ob_type_ptr = (
            ptr24
            if (ptr24 in self._ob_type_syms and ptr8 not in self._ob_type_syms)
            else ptr8
        )
        if ob_type_ptr == 0:
            sym_str = "0x0"
        elif ob_type_ptr in self._ob_type_syms:
            sym_str = self._ob_type_syms[ob_type_ptr]
        else:
            sym_str = f"0x{ob_type_ptr:x}"
        type_str = f"{eff_type or '?'}  →  {sym_str}"
        line3 = Text(f"type: {type_str}", style="dim", no_wrap=True)

        # ── line 4: ctypes ────────────────────────────────────────────────────
        obj_addr = addr + 16 if is_gc else addr
        ctypes_expr = f"ctypes.cast(0x{obj_addr:x}, ctypes.py_object).value"
        if state_str != "live":
            ctypes_expr += f"  [{state_str}]"
        line4 = Text(ctypes_expr, style="dim", no_wrap=True)

        # ── line 5: activities ────────────────────────────────────────────────
        acts = "Enter=hexdump  r=repr  g=refs"
        if self._pinned_addr is not None:
            acts += f"  [pinned 0x{self._pinned_addr:x}]"
        line5 = Text(acts, style="dim", no_wrap=True)

        return Group(rule, line1, line2, line3, line4, line5)

    def _render_block_detail_view(self, max_rows: int = 9999) -> Group:
        """Structured field table + raw 8-byte hexdump for the selected block."""
        import struct as _struct

        if self._sel_pool is None or self._pool_raw is None:
            return Group(Text("(no block selected)", style="dim"))

        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        free_set = _augmented_free_set(
            raw,
            self._sel_pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )

        blks = list(range(_POOL_OVERHEAD, len(raw), block_size))
        if self._cursor >= len(blks):
            return Group(Text("(cursor out of range)", style="dim"))

        off = blks[self._cursor]
        addr = self._sel_pool.address + off
        blk_bytes = raw[off : off + block_size]
        state_str = (
            "unborn" if off >= nextoffset else ("free" if addr in free_set else "live")
        )
        raw_type = self._type_names.get(addr, "")
        eff_type = raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)

        py_ver = self._snap.py_version if self._snap else (3, 10)
        gc_known = _is_gc_type(eff_type, py_ver)
        if gc_known is None:
            ptr8, ptr24 = _ob_type_candidates(blk_bytes)
            gc_known = (
                ptr24 in self._type_ptr_cache
                and self._type_ptr_cache.get(ptr24, "?") not in ("", "?")
                and (self._type_ptr_cache.get(ptr8, "?") in ("", "?") or not ptr8)
            )
            if (
                not gc_known
                and ptr24 >= 0x1_0000
                and bool(self._detail_sym_results.get(ptr24))
            ):
                if not self._detail_sym_results.get(ptr8) and self._type_ptr_cache.get(
                    ptr8, "?"
                ) in ("", "?"):
                    gc_known = True
        is_gc = gc_known

        # ── pointer classification ─────────────────────────────────────────────
        def _cls(v: int) -> str:
            if v < 0x1_0000:  # below 64 KB — not a valid user-space pointer
                return ""
            if self._snap is not None:
                asiz = self._snap.arena_size
                for arena in self._snap.arenas:
                    if arena.base_address <= v < arena.base_address + asiz:
                        return "(p)"
            if v in self._detail_sym_results:
                return "(so)" if self._detail_sym_results.get(v) else "(m)"
            if v in self._ob_type_syms:
                sym = self._ob_type_syms[v]
                if not sym.startswith("<unmapped"):
                    return "(so)"
                # Unmapped by the symbolizer, but may still be a valid PyTypeObject.
                # Fall through to the type-ptr cache check below.
            if v in self._type_ptr_cache and self._type_ptr_cache[v] not in ("", "?"):
                return "(so)"
            return "(m)"

        def _full_sym(v: int) -> str:
            """Return the full symbol string (mod!sym or mod+0x...) for a pointer, or ''."""
            if v in self._ob_type_syms:
                return self._ob_type_syms[v]
            if v in self._detail_sym_results:
                info = self._detail_sym_results.get(v)
                if info:
                    sym = info.get("symbol", "")
                    mod = info.get("module", "")
                    offset = info.get("offset", 0)
                    sym_off = info.get("symbolOffset", 0)
                    if sym:
                        return (
                            f"{mod}!{sym}+0x{sym_off:x}" if sym_off else f"{mod}!{sym}"
                        )
                    return mod if offset == 0 else f"{mod}+0x{offset:x}"
            if v in self._detail_cstr_results:
                s = self._detail_cstr_results[v]
                if s:
                    return repr(s)
            return ""

        # ── struct field layout ───────────────────────────────────────────────
        fields = _block_field_layout(eff_type, is_gc, py_ver, len(blk_bytes))
        # For free blocks: pymalloc stores the freelist pointer at block+0, regardless
        # of GC status.  Rename whatever field lives at offset 0 to next_free_block.
        if state_str == "free":
            fields = [
                (
                    f_off,
                    "next_free_block" if f_off == 0 else name,
                    color,
                    "ptr" if f_off == 0 else interp,
                )
                for f_off, name, color, interp in fields
            ]
        # Map offset → (name, color, interp) for the 8-byte chunks we know about.
        # i32 fields occupy 4 bytes but are aligned on their own offset; track their
        # start offset so we can annotate the containing 8-byte group.
        field_map: dict[int, tuple[str, str, str]] = {}
        for f_off, name, color, interp in fields:
            chunk_base = (f_off // 8) * 8
            if chunk_base not in field_map:
                field_map[chunk_base] = (name, color, interp)

        # ── unified offset | hex | 0x... | name | region | symbol ─────────────
        ut = Table(
            box=box.SIMPLE_HEAD, show_edge=False, show_header=False, padding=(0, 1)
        )
        ut.add_column("off", style="dim", no_wrap=True, min_width=6)
        ut.add_column("hex", no_wrap=True, min_width=23)  # 8 bytes × "xx " − 1
        ut.add_column("ascii", style="dim", no_wrap=True, min_width=8)
        ut.add_column("value", no_wrap=True, min_width=18)
        ut.add_column("name", no_wrap=True, min_width=16)
        ut.add_column("region", no_wrap=True, min_width=4)
        ut.add_column("symbol", style="dim", no_wrap=True)

        total_chunks = len(blk_bytes) // 8
        shown = 0
        for chunk_off in range(0, len(blk_bytes), 8):
            if shown >= max_rows:
                remaining = total_chunks - shown
                ut.add_row(
                    Text("…", style="dim"),
                    Text(f"({remaining} more rows)", style="dim"),
                    Text(""),
                    Text(""),
                    Text(""),
                    Text(""),
                    Text(""),
                )
                break
            chunk = blk_bytes[chunk_off : chunk_off + 8]
            if len(chunk) < 8:
                break
            shown += 1
            v = _struct.unpack_from("<Q", chunk, 0)[0]
            hex_str = " ".join(f"{b:02x}" for b in chunk)

            fi = field_map.get(chunk_off)
            if fi:
                fname, fcolor, finterp = fi
                hex_lbl = Text(hex_str, style=fcolor, no_wrap=True)
                if finterp == "f64":
                    fv = _struct.unpack_from("<d", chunk, 0)[0]
                    val_lbl = Text(f"{fv!r}", style=fcolor, no_wrap=True)
                elif finterp == "i32":
                    vi = _struct.unpack_from("<I", chunk, 0)[0]
                    val_lbl = Text(f"0x{vi:08x}", style=fcolor, no_wrap=True)
                else:
                    val_lbl = Text(f"0x{v:016x}", style=fcolor, no_wrap=True)
                name_lbl = Text(fname, style=fcolor, no_wrap=True)
            else:
                hex_lbl = Text(hex_str, style="dim", no_wrap=True)
                val_lbl = Text(f"0x{v:016x}", style="dim", no_wrap=True)
                name_lbl = Text("", no_wrap=True)

            ascii_str = "".join(chr(b) if 0x20 <= b <= 0x7E else "." for b in chunk)
            # Only classify pointer-like values: annotated ptr fields, or unannotated chunks.
            treat_as_ptr = (fi is None) or (fi[2] == "ptr")
            cls_lbl = Text(_cls(v) if treat_as_ptr else "", style="dim", no_wrap=True)
            sym_lbl = Text(
                _full_sym(v) if treat_as_ptr else "", style="dim", no_wrap=True
            )
            ut.add_row(
                f"+{chunk_off:04x}",
                hex_lbl,
                Text(ascii_str, no_wrap=True),
                val_lbl,
                name_lbl,
                cls_lbl,
                sym_lbl,
            )

        return Group(ut)

    @keke.ktrace()
    def _render(self) -> Group | Layout:
        h = shutil.get_terminal_size().lines
        # _DETAIL_SIZE: rule + 5 info lines = 6; _HEADER_SIZE = 2.
        _HEADER_SIZE = 2
        _DETAIL_SIZE = 6
        if self._view == _View.BLOCK_DETAIL:
            detail_vp = max(1, h - _HEADER_SIZE - _DETAIL_SIZE)
            body = self._render_block_detail_view(max_rows=detail_vp)
            layout = Layout()
            layout.split_column(
                Layout(self._make_header(), size=_HEADER_SIZE, name="header"),
                Layout(body, ratio=1, name="body"),
                Layout(self._make_detail_panel(), size=_DETAIL_SIZE, name="panel"),
            )
            return layout
        elif self._view == _View.BLOCK:
            # SIMPLE_HEAD with show_header=True adds 2 chrome lines (header + separator)
            # that don't hold data rows, so subtract them from the viewport height passed
            # to _clamp_viewport inside _render_block_view.
            _TABLE_CHROME = 2
            vp_h = max(1, h - _HEADER_SIZE - _DETAIL_SIZE - _TABLE_CHROME)
            block_body = self._render_block_view(vp_h)
            layout = Layout()
            layout.split_column(
                Layout(self._make_header(), size=_HEADER_SIZE, name="header"),
                Layout(block_body, ratio=1, name="body"),
                Layout(self._make_detail_panel(), size=_DETAIL_SIZE, name="panel"),
            )
            return layout
        else:
            # 2 header + 2 table header/sep
            vp_h = max(1, h - 4)
            if self._view == _View.SIZE_CLASS:
                main_body = (
                    self._render_arena_view(vp_h)
                    if self._hier_mode == "arena"
                    else self._render_size_class_view(vp_h)
                )
            else:
                main_body = (
                    self._render_arena_pool_view(vp_h)
                    if self._hier_mode == "arena"
                    else self._render_pool_view(vp_h)
                )
            return Group(self._make_header(), main_body)

    def _resolve_block_types(self) -> None:
        """Extract ob_type ptrs from pool raw bytes and resolve to names in one RPC call."""
        if self._sel_pool is None or self._pool_raw is None:
            return
        import struct

        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        block_size = (szidx + 1) * 16

        # Collect candidate ob_type pointers for live AND free blocks.
        # GC-tracked objects (dict, list, etc.) have a 16-byte PyGC_Head before
        # the PyObject, so we read both block+8 and block+24 and let
        # resolveTypeNames determine which is the real ob_type pointer.
        block_cands: list[tuple[int, int, int]] = []  # (addr, ptr8, ptr24)
        all_candidates: set[int] = set()
        for off in range(_POOL_OVERHEAD, len(raw), block_size):
            addr = self._sel_pool.address + off
            if off >= nextoffset:
                continue  # unborn — never had a type
            ptr8, ptr24 = _ob_type_candidates(raw[off : off + block_size])
            all_candidates.update(p for p in (ptr8, ptr24) if p)
            block_cands.append((addr, ptr8, ptr24))

        if not block_cands:
            return

        # Only call RPC for type ptrs not already in our per-pointer cache.
        new_type_ptrs = [p for p in all_candidates if p not in self._type_ptr_cache]
        if new_type_ptrs:
            self._rpc_lookups += len(new_type_ptrs)
            try:
                resolved = self._col.resolve_type_names(new_type_ptrs)
                self._type_ptr_cache.update(resolved)
            except Exception:
                pass

        # Populate per-block-address names, picking the candidate that resolves.
        winning_ptrs: list[tuple[int, int]] = []  # (addr, ob_type_ptr)
        for addr, ptr8, ptr24 in block_cands:
            ob_type_ptr, name = _pick_ob_type(ptr8, ptr24, self._type_ptr_cache)
            if name not in ("", "?"):
                self._type_names[addr] = name
            elif addr not in self._type_names:
                self._type_names[addr] = "?"
            if ob_type_ptr:
                winning_ptrs.append((addr, ob_type_ptr))

        # Symbolize winning ob_type pointers not yet in cache.
        # Skip pointers below 64 KB — they are not valid user-space addresses and
        # only arise from fallback when neither ptr8 nor ptr24 resolved (e.g. GC objects
        # whose ob_type is at +24 but the field layout guessed non-GC).
        unique_sym_ptrs = [
            p for _, p in winning_ptrs if p >= 0x1_0000 and p not in self._ob_type_syms
        ]
        if unique_sym_ptrs:
            self._rpc_lookups += len(unique_sym_ptrs)
            try:
                sym_results = self._col.symbolize_addresses(unique_sym_ptrs)
            except Exception:
                sym_results = {}
            for p, info in sym_results.items():
                if info is None:
                    self._ob_type_syms[p] = f"<unmapped 0x{p:x}>"
                else:
                    mod = info["module"]
                    offset = info["offset"]
                    symbol = info.get("symbol")
                    sym_off = info.get("symbolOffset", 0)
                    if symbol:
                        self._ob_type_syms[p] = (
                            f"{mod}!{symbol}+0x{sym_off:x}"
                            if sym_off
                            else f"{mod}!{symbol}"
                        )
                    elif offset == 0:
                        self._ob_type_syms[p] = mod
                    else:
                        self._ob_type_syms[p] = f"{mod}+0x{offset:x}"

        # For code objects, eagerly resolve co_name + co_filename via repr_block.
        # Python 3.11+ moved co_filename from +96 to +40 when the struct was reorganised.
        py_ver = self._snap.py_version if self._snap else (3, 10)
        if py_ver >= (3, 11):
            co_name_off, co_filename_off = 48, 40
        else:
            co_name_off, co_filename_off = 104, 96

        for addr, _ in winning_ptrs:
            type_n = self._type_names.get(addr)
            if addr in self._name_hints:
                continue
            off = addr - self._sel_pool.address

            if type_n == "code":
                parts = []
                for field_off in (co_name_off, co_filename_off):
                    if off + field_off + 8 > len(raw):
                        continue
                    ptr = struct.unpack_from("<Q", raw, off + field_off)[0]
                    if not ptr:
                        continue
                    try:
                        result = self._col.repr_block(ptr)
                        if result and result[0] is not None:
                            parts.append(result[1])
                    except Exception:
                        pass
                if parts:
                    self._name_hints[addr] = " in ".join(parts)

            elif type_n == "module":
                # md_name is a Python str at +48 (stable across 3.10–3.13)
                if off + 56 <= len(raw):
                    ptr = struct.unpack_from("<Q", raw, off + 48)[0]
                    if ptr:
                        try:
                            result = self._col.repr_block(ptr)
                            if result and result[0] is not None:
                                self._name_hints[addr] = result[1]
                        except Exception:
                            pass

            elif type_n == "frame":
                # frame is GC-tracked: PyGC_Head(16) + ob_refcnt(8) + ob_type(8) = 32 bytes,
                # then f_back(8), then f_code/f_frame at block+40.
                #   3.10:  f_code (PyCodeObject*) at block+40
                #   3.11+: f_frame (_PyInterpreterFrame*) at block+40;
                #          for heap frames f_frame points into the same pool block;
                #          f_code is at _PyInterpreterFrame+32
                if off + 48 > len(raw):
                    continue
                ptr_at_40 = struct.unpack_from("<Q", raw, off + 40)[0]
                if not ptr_at_40:
                    continue
                if py_ver >= (3, 11):
                    # Follow f_frame → f_code only if f_frame lands inside this pool's raw bytes
                    # (heap frames). Stack frames can't be followed safely.
                    f_frame_in_raw = ptr_at_40 - self._sel_pool.address
                    if not (0 <= f_frame_in_raw + 40 <= len(raw)):
                        continue
                    f_code_ptr = struct.unpack_from("<Q", raw, f_frame_in_raw + 32)[0]
                    if not f_code_ptr:
                        continue
                    ptr_at_40 = f_code_ptr
                # ptr_at_40 is now the f_code PyObject* — call repr_block to get its name.
                try:
                    result = self._col.repr_block(ptr_at_40)
                    if result and result[0] is not None:
                        import re

                        m = re.search(r"<code object ([^\s>]+)", result[1])
                        if m:
                            self._name_hints[addr] = m.group(1)
                except Exception:
                    pass

    def _pin(self, addr: int) -> None:
        """Incref addr if it differs from the current pin; decref the old one first."""
        if addr == self._pinned_addr:
            return
        self._unpin()
        try:
            err = self._col.incref(addr)
            if err is None:
                self._pinned_addr = addr
            # else: rejected by safety guard — leave _pinned_addr as None
        except Exception:
            pass

    def _unpin(self) -> None:
        """Decref the currently pinned address, if any."""
        if self._pinned_addr is None:
            return
        addr = self._pinned_addr
        self._pinned_addr = None
        try:
            self._col.decref(addr)
        except Exception:
            pass

    def _on_cursor_change(self) -> None:
        """Update the auto-incref pin after cursor movement in block/detail view."""
        if (
            self._view not in (_View.BLOCK, _View.BLOCK_DETAIL)
            or self._sel_pool is None
            or self._pool_raw is None
        ):
            self._unpin()
            return
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        block_size = (szidx + 1) * 16
        free_set = _augmented_free_set(
            raw,
            self._sel_pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )

        blocks = [
            (self._sel_pool.address + off, off)
            for off in range(_POOL_OVERHEAD, len(raw), block_size)
        ]
        if self._cursor >= len(blocks):
            self._unpin()
            return

        addr, off = blocks[self._cursor]
        if off >= nextoffset or addr in free_set:
            # Unborn or free — no live PyObject to pin
            self._unpin()
            return

        blk_bytes = raw[off : off + block_size]
        raw_type = self._type_names.get(addr, "")
        eff_type = raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)
        if eff_type == "cstr":
            # No PyObject refcount structure
            self._unpin()
            return

        # Determine the PyObject address to incref:
        # GC-tracked objects have PyGC_Head at block start; ob_refcnt is at block+16.
        py_ver = self._snap.py_version if self._snap else (3, 10)
        gc_known = _is_gc_type(eff_type, py_ver)
        if gc_known is None:
            ptr8, ptr24 = _ob_type_candidates(blk_bytes)
            gc_known = (
                ptr24 in self._type_ptr_cache
                and self._type_ptr_cache.get(ptr24, "?") not in ("", "?")
                and (self._type_ptr_cache.get(ptr8, "?") in ("", "?") or not ptr8)
            )
        obj_addr = addr + 16 if gc_known else addr
        # Safety: skip pinning when:
        #  - ob_refcnt == 0: block is dead/freed (dealloc zeroes the refcount,
        #    or the free-list heuristic missed it).  Calling Py_IncRef on a
        #    refcount-0 object would resurrect it and corrupt the heap.
        #  - ob_refcnt > 1000: looks like a raw pointer rather than a count
        #    (mis-identified GC vs non-GC layout).
        # Always call _classify_detail_ptrs regardless — skipped blocks still
        # need pointer symbols in the block-detail view.
        import struct as _s_pin

        _rc_blk_off = 16 if gc_known else 0
        _should_pin = True
        if len(blk_bytes) >= _rc_blk_off + 8:
            _rc_raw = _s_pin.unpack_from("<Q", blk_bytes, _rc_blk_off)[0]
            if _rc_raw == 0 or _rc_raw > 1000:
                _should_pin = False
        if _should_pin:
            self._pin(obj_addr)
        else:
            self._unpin()
        if self._view == _View.BLOCK_DETAIL:
            self._classify_detail_ptrs()

    def _classify_detail_ptrs(self) -> None:
        """Batch-symbolize 8-byte values in the current block for pointer classification.

        Results go into _detail_sym_results: {value -> {module,...} | None}.
        Called on entry to BLOCK_DETAIL and on cursor change within it.
        Skips values already resolved in _ob_type_syms / _type_ptr_cache.
        """
        import struct as _struct

        self._detail_sym_results = {}
        self._detail_cstr_results = {}
        if self._sel_pool is None or self._pool_raw is None:
            return
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        blks = list(range(_POOL_OVERHEAD, len(raw), block_size))
        if self._cursor >= len(blks):
            return
        off = blks[self._cursor]
        blk_bytes = raw[off : off + block_size]
        candidates = []
        for i in range(0, len(blk_bytes) - 7, 8):
            v = _struct.unpack_from("<Q", blk_bytes, i)[0]
            if (
                v >= 0x1000
                and v not in self._type_ptr_cache
                and v not in self._ob_type_syms
            ):
                candidates.append(v)
        if candidates:
            try:
                self._detail_sym_results = self._col.symbolize_addresses(candidates)
            except Exception:
                pass
        cstr_candidates = [v for v in candidates if not self._detail_sym_results.get(v)]
        if cstr_candidates:
            try:
                self._detail_cstr_results = self._col.peek_cstrings(cstr_candidates)
            except Exception:
                pass

    def _do_referrer_search(self) -> None:
        """Scan all pools in the snapshot for blocks that contain a pointer to the current block.

        For GC-tracked objects the pointer in the wild is the PyObject* (block+16),
        not the pool-block base address.  We search for both to be safe.
        """
        import struct

        if self._snap is None or self._sel_pool is None or self._pool_raw is None:
            return
        if self._view not in (_View.BLOCK, _View.BLOCK_DETAIL):
            return

        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        blks = list(range(_POOL_OVERHEAD, len(raw), block_size))
        if self._cursor >= len(blks):
            return

        off = blks[self._cursor]
        addr = self._sel_pool.address + off
        blk_bytes = raw[off : off + block_size]

        raw_type = self._type_names.get(addr, "")
        eff_type = raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)
        py_ver = self._snap.py_version if self._snap else (3, 10)
        gc_known = _is_gc_type(eff_type, py_ver)
        if gc_known is None:
            ptr8, ptr24 = _ob_type_candidates(blk_bytes)
            gc_known = (
                ptr24 in self._type_ptr_cache
                and self._type_ptr_cache.get(ptr24, "?") not in ("", "?")
                and (self._type_ptr_cache.get(ptr8, "?") in ("", "?") or not ptr8)
            )
        # PyObject* address: block+16 for GC-tracked objects (PyGC_Head is the 16-byte prefix),
        # block+0 for non-GC.  References in other objects will hold this address.
        obj_addr = addr + 16 if gc_known else addr
        self._referrer_target = obj_addr

        results: list[int] = []
        all_pools = [p for sc in self._snap.size_classes for p in sc.pools]

        for pool in all_pools:
            try:
                if pool.address == self._sel_pool.address:
                    pr_raw = raw
                    pr_szidx = szidx
                    pr_nextoffset = self._pool_raw.get("nextoffset", pool.nextoffset)
                else:
                    pr = self._col.read_pool(pool.address)
                    pr_raw = bytes(pr.get("raw", b"") or b"")
                    pr_szidx = pr.get("szidx", pool.szidx)
                    pr_nextoffset = pr.get("nextoffset", pool.nextoffset)
            except Exception:
                continue

            pr_bs = (pr_szidx + 1) * 16
            for b_off in range(_POOL_OVERHEAD, len(pr_raw), pr_bs):
                if b_off >= pr_nextoffset:
                    break
                b_addr = pool.address + b_off
                blk = pr_raw[b_off : b_off + pr_bs]
                for i in range(0, len(blk) - 7, 8):
                    if struct.unpack_from("<Q", blk, i)[0] == obj_addr:
                        results.append(b_addr)
                        break

        self._referrer_results = results
        self._referrer_idx = 0

    def _referrer_nav(self, delta: int) -> None:
        """Jump to the next/prev referrer in the result list."""
        if not self._referrer_results:
            return
        n = len(self._referrer_results)
        self._referrer_idx = (self._referrer_idx + delta) % n
        target_addr = self._referrer_results[self._referrer_idx]
        # Preserve referrer state across the navigation jump.
        saved_results = self._referrer_results
        saved_idx = self._referrer_idx
        saved_target = self._referrer_target
        try:
            self._jump_to(target_addr)
        except Exception:
            pass
        self._referrer_results = saved_results
        self._referrer_idx = saved_idx
        self._referrer_target = saved_target

    def _fetch_pool_protection(self) -> None:
        """Fetch and cache the memory protection string for the selected pool."""
        if self._sel_pool is None:
            self._sel_pool_protection = None
            return
        try:
            self._sel_pool_protection = self._col.get_range_protection(
                self._sel_pool.address
            )
        except Exception:
            self._sel_pool_protection = None

    # ── snapshot refresh ──────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if (
            self._view in (_View.BLOCK, _View.BLOCK_DETAIL)
            and self._sel_pool is not None
        ):
            self._refresh_block_view()
        else:
            self._refresh_full()

    @keke.ktrace()
    def _refresh_full(self) -> None:
        """Full arena scan — used for size-class and pool views."""
        if not self._col.ping_gil():
            return
        self._refresh_parity = not self._refresh_parity
        self._snap = self._col.collect_streaming()

    @keke.ktrace()
    def _refresh_block_view(self) -> None:
        """Fast path: re-read only the current pool (no full arena scan)."""
        assert self._sel_pool is not None
        if not self._col.ping_gil():
            return
        self._refresh_parity = not self._refresh_parity
        now = time.monotonic()
        try:
            self._pool_raw = self._col.read_pool(self._sel_pool.address)
        except Exception:
            return
        self._resolve_block_types()

        # Update age tracker from this pool's block states
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        free_set = _augmented_free_set(
            raw,
            self._sel_pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )
        live: set[int] = set()
        for off in range(_POOL_OVERHEAD, len(raw), block_size):
            addr = self._sel_pool.address + off
            if off < nextoffset and addr not in free_set:
                live.add(addr)
        self._age.update(live, now)

    # ── keyboard handling ─────────────────────────────────────────────────────

    def _current_list_len(self) -> int:
        if self._snap is None:
            return 0
        if self._view == _View.SIZE_CLASS:
            if self._hier_mode == "arena":
                return len(self._snap.arenas)
            return sum(1 for sc in self._snap.size_classes if sc.pool_count > 0)
        if self._view == _View.POOL:
            if self._hier_mode == "arena":
                if self._sel_arena_idx is not None:
                    arena = next(
                        (
                            a
                            for a in self._snap.arenas
                            if a.arena_index == self._sel_arena_idx
                        ),
                        None,
                    )
                    return len(arena.pools) if arena else 0
                return 0
            if self._sel_szidx is not None:
                return len(self._snap.size_classes[self._sel_szidx].pools)
            return 0
        if (
            self._view in (_View.BLOCK, _View.BLOCK_DETAIL)
            and self._sel_pool is not None
        ):
            szidx = (
                self._pool_raw.get("szidx", self._sel_pool.szidx)
                if self._pool_raw
                else self._sel_pool.szidx
            )
            blk = (szidx + 1) * 16
            raw = (
                bytes(self._pool_raw.get("raw", b"") or b"") if self._pool_raw else b""
            )
            return max(0, (len(raw) - _POOL_OVERHEAD) // blk) if blk else 0
        return 0

    def _jump_to(self, addr: int) -> None:
        """Navigate to the pool containing *addr* and position cursor on that block."""
        # Always do a fresh full collect so _snap is up-to-date.
        self._refresh_full()
        if self._snap is None:
            raise RuntimeError("no snapshot")
        pool_size = self._snap.pool_size
        pool_addr = addr & ~(pool_size - 1)  # mask to pool-aligned base

        # Find the size class and pool index
        found_pool = None
        found_szidx = None
        pool_cursor = 0
        for sc in self._snap.size_classes:
            for pool_idx, pool in enumerate(sc.pools):
                if pool.address == pool_addr:
                    found_pool = pool
                    found_szidx = sc.szidx
                    pool_cursor = pool_idx
                    break
            if found_pool:
                break

        if found_pool is None:
            found_pool = self._col.read_pool_snapshot(pool_addr)
            found_szidx = found_pool.szidx if found_pool else None
            pool_cursor = 0

        if found_pool is not None:
            assert found_szidx is not None
            # Navigate: switch to pool view for this size class / arena
            self._sel_szidx = found_szidx
            if self._hier_mode == "arena" and self._snap is not None:
                for arena in self._snap.arenas:
                    if arena.arena_index == found_pool.arena_index:
                        self._sel_arena_idx = arena.arena_index
                        break
            self._view = _View.POOL
            self._cursor = pool_cursor
            self._vp_start = 0
            # Immediately drill into block view
            self._sel_pool = found_pool
            try:
                self._pool_raw = self._col.read_pool(found_pool.address)
            except Exception:
                self._pool_raw = None
            self._repr_lines = []
            self._type_names = {}
            self._name_hints = {}
            self._ob_type_syms = {}
            self._type_ptr_cache = {}
            self._fetch_pool_protection()
            self._resolve_block_types()
            self._view = _View.BLOCK
            self._vp_start = 0
            # Position cursor on the target block (floor-div; no strict alignment check)
            block_size = (found_szidx + 1) * 16
            off = addr - pool_addr
            self._cursor = (
                (off - _POOL_OVERHEAD) // block_size if off >= _POOL_OVERHEAD else 0
            )
            return

        sym_label = f"0x{addr:x}"
        try:
            info = self._col.symbolize_addresses([addr]).get(addr)
            if info is not None:
                mod, offset, symbol = info["module"], info["offset"], info.get("symbol")
                sym_label = (
                    f"{mod}!{symbol}"
                    if symbol
                    else (mod if offset == 0 else f"{mod}+0x{offset:x}")
                )
        except Exception:
            pass
        raise RuntimeError(f"{sym_label} not in any pymalloc pool")

    def _enter(self) -> None:
        if self._snap is None:
            return
        if self._view == _View.SIZE_CLASS:
            if self._hier_mode == "arena":
                arenas = self._snap.arenas
                if self._cursor < len(arenas):
                    arena = arenas[self._cursor]
                    self._sel_arena_idx = arena.arena_index
                    saved = self._saved_pool_cursor.get(
                        ("arena", self._sel_arena_idx), None
                    )
                    if saved is None and self._sel_szidx is not None:
                        # Align cursor to the first pool with the same size class
                        # so that toggling back to size mode lands on the right row.
                        for j, p in enumerate(arena.pools):
                            if p.szidx == self._sel_szidx:
                                saved = j
                                break
                    if saved is None:
                        saved = 0
                    self._view = _View.POOL
                    self._cursor = saved
                    self._vp_start = 0
            else:
                active = [sc for sc in self._snap.size_classes if sc.pool_count > 0]
                if self._cursor < len(active):
                    self._sel_szidx = active[self._cursor].szidx
                    saved = self._saved_pool_cursor.get(("size", self._sel_szidx), 0)
                    self._view = _View.POOL
                    self._cursor = saved
                    self._vp_start = 0
        elif self._view == _View.BLOCK:
            self._classify_detail_ptrs()
            self._view = _View.BLOCK_DETAIL
        elif self._view == _View.POOL:
            if self._hier_mode == "arena":
                pools: list[PoolSnapshot] = []
                if self._sel_arena_idx is not None:
                    _arena = next(
                        (
                            a
                            for a in self._snap.arenas
                            if a.arena_index == self._sel_arena_idx
                        ),
                        None,
                    )
                    pools = _arena.pools if _arena else []
            else:
                pools = (
                    self._snap.size_classes[self._sel_szidx].pools
                    if self._sel_szidx is not None
                    else []
                )
            if self._cursor < len(pools):
                self._unpin()
                self._sel_pool = pools[self._cursor]
                try:
                    self._pool_raw = self._col.read_pool(self._sel_pool.address)
                except Exception:
                    self._pool_raw = None
                self._repr_lines = []
                self._type_names = {}
                self._name_hints = {}
                self._ob_type_syms = {}
                self._type_ptr_cache = {}
                self._fetch_pool_protection()
                self._resolve_block_types()
                self._view = _View.BLOCK
                self._cursor = self._saved_block_cursor.get(self._sel_pool.address, 0)
                self._vp_start = 0
                self._on_cursor_change()

    def _back(self) -> None:
        if self._view == _View.BLOCK_DETAIL:
            self._view = _View.BLOCK
            return
        if self._view == _View.BLOCK:
            self._unpin()
            # Save block cursor so re-entering this pool restores position.
            if self._sel_pool is not None:
                self._saved_block_cursor[self._sel_pool.address] = self._cursor
            # Return to pool view; restore cursor to the pool we came from.
            cursor = 0
            if self._snap is not None and self._sel_pool is not None:
                if self._hier_mode == "arena" and self._sel_arena_idx is not None:
                    arena = next(
                        (
                            a
                            for a in self._snap.arenas
                            if a.arena_index == self._sel_arena_idx
                        ),
                        None,
                    )
                    if arena:
                        for i, p in enumerate(arena.pools):
                            if p.address == self._sel_pool.address:
                                cursor = i
                                break
                elif self._sel_szidx is not None:
                    pools = self._snap.size_classes[self._sel_szidx].pools
                    for i, p in enumerate(pools):
                        if p.address == self._sel_pool.address:
                            cursor = i
                            break
            self._view = _View.POOL
            self._cursor = cursor
            self._vp_start = 0
            self._repr_lines = []
            self._type_names = {}
            self._name_hints = {}
            self._ob_type_syms = {}
            self._type_ptr_cache = {}
            self._sel_pool_protection = None
        elif self._view == _View.POOL:
            # Save pool cursor so re-entering this size class / arena restores position.
            pool_key: tuple[str, int] | None = None
            if self._hier_mode == "arena" and self._sel_arena_idx is not None:
                pool_key = ("arena", self._sel_arena_idx)
            elif self._sel_szidx is not None:
                pool_key = ("size", self._sel_szidx)
            if pool_key is not None:
                self._saved_pool_cursor[pool_key] = self._cursor
            # Return to top-level view; restore cursor to the item we came from.
            cursor = 0
            if self._snap is not None:
                if self._hier_mode == "arena" and self._sel_arena_idx is not None:
                    for i, arena in enumerate(self._snap.arenas):
                        if arena.arena_index == self._sel_arena_idx:
                            cursor = i
                            break
                elif self._sel_szidx is not None:
                    active = [sc for sc in self._snap.size_classes if sc.pool_count > 0]
                    for i, sc in enumerate(active):
                        if sc.szidx == self._sel_szidx:
                            cursor = i
                            break
            self._view = _View.SIZE_CLASS
            self._cursor = cursor
            self._vp_start = 0

    def _toggle_hier_mode(self) -> None:
        """Toggle between 'size' and 'arena' hierarchy modes, preserving selection."""
        snap = self._snap
        target = "arena" if self._hier_mode == "size" else "size"

        if self._view == _View.SIZE_CLASS:
            if target == "arena" and snap is not None:
                # Find the arena containing the first pool of the highlighted size class.
                active = [sc for sc in snap.size_classes if sc.pool_count > 0]
                arenas = snap.arenas
                arena_cursor = 0
                if self._cursor < len(active) and active[self._cursor].pools:
                    first_pool = active[self._cursor].pools[0]
                    for i, a in enumerate(arenas):
                        if a.arena_index == first_pool.arena_index:
                            arena_cursor = i
                            break
                self._cursor = arena_cursor
            elif target == "size" and snap is not None:
                # Find the size class of the first pool in the highlighted arena.
                arenas = snap.arenas
                active = [sc for sc in snap.size_classes if sc.pool_count > 0]
                sc_cursor = 0
                if self._cursor < len(arenas) and arenas[self._cursor].pools:
                    first_szidx = arenas[self._cursor].pools[0].szidx
                    for i, sc in enumerate(active):
                        if sc.szidx == first_szidx:
                            sc_cursor = i
                            break
                self._cursor = sc_cursor
            self._vp_start = 0

        elif self._view == _View.POOL:
            if target == "arena" and snap is not None and self._sel_szidx is not None:
                # Find the highlighted pool and locate it in arena mode.
                pools = snap.size_classes[self._sel_szidx].pools
                pool = pools[self._cursor] if self._cursor < len(pools) else None
                if pool is not None:
                    for a in snap.arenas:
                        if a.arena_index == pool.arena_index:
                            self._sel_arena_idx = a.arena_index
                            for j, ap in enumerate(a.pools):
                                if ap.address == pool.address:
                                    self._cursor = j
                                    break
                            break
            elif (
                target == "size"
                and snap is not None
                and self._sel_arena_idx is not None
            ):
                # Find the highlighted pool and locate it in size mode.
                arena = next(
                    (a for a in snap.arenas if a.arena_index == self._sel_arena_idx),
                    None,
                )
                pool = None
                if arena and arena.pools:
                    pool_idx = min(self._cursor, len(arena.pools) - 1)
                    pool = arena.pools[pool_idx]
                if pool is not None:
                    self._sel_szidx = pool.szidx
                    size_pools = snap.size_classes[pool.szidx].pools
                    for j, sp in enumerate(size_pools):
                        if sp.address == pool.address:
                            self._cursor = j
                            break
            self._vp_start = 0

        else:  # BLOCK / BLOCK_DETAIL — just flip the flag, keeping _sel_pool and cursor unchanged.
            if target == "arena" and snap is not None and self._sel_pool is not None:
                for a in snap.arenas:
                    if a.arena_index == self._sel_pool.arena_index:
                        self._sel_arena_idx = a.arena_index
                        break
            elif target == "size" and self._sel_pool is not None:
                self._sel_szidx = self._sel_pool.szidx

        self._hier_mode = target

    def _repr_one_block(
        self,
        addr: int,
        off: int,
        blk_bytes: bytes,
        raw_type: str,
        free_set: frozenset[int],
        nextoffset: int,
    ) -> str:
        """Compute a repr line for one block and return it (does not update _repr_lines)."""
        eff_type = raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)

        if eff_type == "cstr":
            cstr_val = _extract_cstr(blk_bytes)
            return (
                f"0x{addr:x} [cstr]: {repr(cstr_val)}"
                if cstr_val is not None
                else f"0x{addr:x} [cstr]: (empty)"
            )
        elif off >= nextoffset or addr in free_set:
            state_tag = "unborn" if off >= nextoffset else "free"
            if addr in self._name_hints:
                return f"0x{addr:x} [{state_tag} {eff_type}]: {self._name_hints[addr]}"
            py_ver = self._snap.py_version if self._snap else (3, 10)
            fallback = _fallback_repr_from_raw(blk_bytes, raw_type, py_ver)
            return (
                f"0x{addr:x} [{state_tag} {eff_type}]: {fallback}"
                if fallback
                else f"0x{addr:x}: ({state_tag} — no inline data)"
            )
        else:
            py_ver = self._snap.py_version if self._snap else (3, 10)
            gc_known = _is_gc_type(eff_type, py_ver)
            if gc_known is None:
                ptr8, ptr24 = _ob_type_candidates(blk_bytes)
                gc_known = (
                    ptr24 in self._type_ptr_cache
                    and self._type_ptr_cache.get(ptr24, "?") not in ("", "?")
                    and (self._type_ptr_cache.get(ptr8, "?") in ("", "?") or not ptr8)
                )
            obj_addr = addr + 16 if gc_known else addr
            result = self._col.repr_block(obj_addr)
            if result is None:
                return f"0x{addr:x}: repr failed (rpc error)"
            type_name, repr_str = result
            if type_name is not None:
                return f"0x{addr:x} [{type_name}]: {repr_str}"
            return f"0x{addr:x}: repr failed — {repr_str}"

    def _do_repr(self) -> None:
        """Repr selected block (block or block-detail view)."""
        if (
            self._view not in (_View.BLOCK, _View.BLOCK_DETAIL)
            or self._sel_pool is None
            or self._pool_raw is None
        ):
            return
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        free_set = _augmented_free_set(
            raw,
            self._sel_pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )

        blocks = [
            self._sel_pool.address + off
            for off in range(_POOL_OVERHEAD, len(raw), block_size)
        ]
        if self._cursor >= len(blocks):
            return
        addr = blocks[self._cursor]
        off = addr - self._sel_pool.address
        blk_bytes = raw[off : off + block_size]
        raw_type = self._type_names.get(addr, "")
        line = self._repr_one_block(
            addr, off, blk_bytes, raw_type, free_set, nextoffset
        )
        # Replace existing repr for this address or append
        self._repr_lines = [(a, t) for a, t in self._repr_lines if a != addr]
        self._repr_lines.append((addr, line))

    def _do_repr_all_visible(self) -> None:
        """Repr every visible block in the block view that has a resolved type."""
        if self._sel_pool is None or self._pool_raw is None:
            return
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        free_set = _augmented_free_set(
            raw,
            self._sel_pool.address,
            block_size,
            nextoffset,
            self._pool_raw.get("freeAddrs", []),
        )

        # Match the viewport height calculation used in _render().
        h = shutil.get_terminal_size().lines
        vp_h = max(1, h - 10)  # _HEADER_SIZE=2 + _DETAIL_SIZE=6 + _TABLE_CHROME=2

        block_offsets = list(range(_POOL_OVERHEAD, len(raw), block_size))
        visible = block_offsets[self._vp_start : self._vp_start + vp_h]

        new_reprs: list[tuple[int, str]] = []
        for off in visible:
            addr = self._sel_pool.address + off
            blk_bytes = raw[off : off + block_size]
            raw_type = self._type_names.get(addr, "")
            eff_type = (
                raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)
            )
            if not eff_type:
                continue
            # Skip unborn blocks — they hold no object data.
            if off >= nextoffset:
                continue
            line = self._repr_one_block(
                addr, off, blk_bytes, raw_type, free_set, nextoffset
            )
            new_reprs.append((addr, line))

        # Merge: replace existing entries for these addresses, keep the rest.
        updated_addrs = {a for a, _ in new_reprs}
        self._repr_lines = [
            (a, t) for a, t in self._repr_lines if a not in updated_addrs
        ] + new_reprs

    def _handle_key(self, ch: str) -> bool:
        """Process one keypress. Returns True if we should quit."""
        # ── jump-to-address input mode ─────────────────────────────────────
        if self._jump_buf is not None:
            if ch in ("\r", "\n"):  # confirm
                self._jump_err = ""
                try:
                    addr = int(self._jump_buf, 0)  # 0x… = hex, else decimal
                    self._jump_to(addr)
                except ValueError:
                    self._jump_err = (
                        f"invalid address: {self._jump_buf!r} (use 0x… or decimal)"
                    )
                except Exception as e:
                    self._jump_err = str(e)
                self._jump_buf = None
            elif ch in ("\x1b", "\x7f", "\x08"):  # Esc or backspace
                if ch in ("\x7f", "\x08") and self._jump_buf:
                    self._jump_buf = self._jump_buf[:-1]
                else:
                    self._jump_buf = None
            elif ch.isprintable() and ch not in ("\r", "\n", " "):
                self._jump_buf += ch
            return False

        # ── normal mode ───────────────────────────────────────────────────
        if ch in ("q", "Q"):
            return True
        n = self._current_list_len()
        h = shutil.get_terminal_size().lines
        page = max(1, h - 10)
        _prev_cursor = self._cursor
        if ch == "\x1b[A" or ch == "k":  # up
            self._cursor = max(0, self._cursor - 1)
            if self._cursor == _prev_cursor and self._cursor == 0:
                self._flash_msg = "beginning of list"
                self._flash_until = time.monotonic() + 1.5
        elif ch == "\x1b[B" or ch == "j":  # down
            self._cursor = min(max(0, n - 1), self._cursor + 1)
            if self._cursor == _prev_cursor and n > 0:
                self._flash_msg = "end of list"
                self._flash_until = time.monotonic() + 1.5
        elif ch == "n" or ch == "\x1b[6~":  # page down / next referrer
            if self._referrer_results is not None:
                self._referrer_nav(+1)
            else:
                self._cursor = min(max(0, n - 1), self._cursor + page)
        elif ch == "p" or ch == "\x1b[5~":  # page up / prev referrer
            if self._referrer_results is not None:
                self._referrer_nav(-1)
            else:
                self._cursor = max(0, self._cursor - page)
        elif ch in ("\r", "\n", "o"):  # Enter / o
            self._enter()
        elif ch in ("b", "\x1b"):  # back / Escape
            if self._referrer_results is not None:
                # First press exits referrer mode; subsequent press exits block view.
                self._referrer_results = None
                self._referrer_target = None
            else:
                self._back()
        if (
            self._view in (_View.BLOCK, _View.BLOCK_DETAIL)
            and self._cursor != _prev_cursor
        ):
            self._on_cursor_change()
        elif ch == "/":
            self._jump_buf = ""
            self._jump_err = ""
        elif ch == "\t":
            self._toggle_hier_mode()
        elif ch == "r":
            self._do_repr()
        elif ch == "R":
            if self._view == _View.BLOCK:
                self._do_repr_all_visible()
            else:
                with self._collect_lock:
                    self._refresh()
        elif ch == "g":
            if self._view in (_View.BLOCK, _View.BLOCK_DETAIL):
                self._do_referrer_search()
        return False

    # ── main loop ─────────────────────────────────────────────────────────────

    def _key_reader(self) -> None:
        """Background thread: read stdin one byte at a time in raw mode.

        Uses os.read(fd, 1) + select([fd], ...) on the raw file descriptor to
        avoid Python's TextIOWrapper buffering ahead, which would consume the
        full escape sequence (\x1b[B) but return only \x1b to read(1), causing
        select to time out on an empty fd and emit a spurious bare \x1b.
        """
        import atexit
        import os
        import select as _sel
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)

        def _restore() -> None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        atexit.register(_restore)
        try:
            tty.setcbreak(fd)
            buf = b""
            while True:
                data = os.read(fd, 1)
                if not data:
                    break
                buf += data
                if buf.startswith(b"\x1b"):
                    timeout = 0.05 if buf == b"\x1b" else 0.01
                    ready, _, _ = _sel.select([fd], [], [], timeout)
                    if ready:
                        continue
                self._keys.put(buf.decode("utf-8", errors="replace"))
                buf = b""
        finally:
            _restore()

    def run(self) -> None:
        import threading

        from rich.live import Live

        # Initial snapshot
        try:
            self._refresh()
        except Exception as exc:
            print(f"Error during initial collect: {exc}", file=sys.stderr)
            return

        def _refresh_loop() -> None:
            while not _stop.is_set():
                _stop.wait(self._interval)
                if _stop.is_set():
                    break
                # Skip this cycle if a manual refresh (R key) is in progress.
                if self._collect_lock.acquire(blocking=False):
                    try:
                        self._refresh()
                    except Exception:
                        pass
                    finally:
                        self._collect_lock.release()

        _stop = threading.Event()
        t_key = threading.Thread(target=self._key_reader, daemon=True)
        t_ref = threading.Thread(target=_refresh_loop, daemon=True)
        t_key.start()
        t_ref.start()

        console = Console()

        with Live(
            self._render(), console=console, screen=True, auto_refresh=False
        ) as live:
            try:
                while True:
                    # Drain the key queue — never blocks on data collection.
                    quit_requested = False
                    while not self._keys.empty():
                        try:
                            ch = self._keys.get_nowait()
                        except queue.Empty:
                            break
                        if self._handle_key(ch):
                            quit_requested = True
                            break
                    if quit_requested:
                        break

                    with keke.kev("live.update"):
                        live.update(self._render())
                    with keke.kev("live.refresh"):
                        live.refresh()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass
            finally:
                _stop.set()
                # Join the background thread before unloading the Frida script.
                # We intentionally do NOT call _unpin() here.  _unpin() sends a
                # decref_block RPC which calls PyGILState_Ensure in the JS event
                # loop.  PyGILState_Ensure can *block* waiting for the target's
                # Python GIL; while it's blocked the event loop is stalled and
                # any concurrent Frida call from the background thread will never
                # get a response, hanging that thread.  Worse, if script.unload()
                # fires while PyGILState_Ensure is blocking, Frida may kill the JS
                # thread mid-syscall, leaving the target GIL permanently stuck.
                #
                # The cost: one Python object in the target retains an extra
                # reference until it exits.  For any live object with more than
                # one real reference this is invisible; and it's far better than
                # freezing the target process.
                t_ref.join(timeout=self._interval + 2.0)


# ── helpers ───────────────────────────────────────────────────────────────────


def _block_field_layout(
    eff_type: str,
    is_gc: bool,
    py_ver: tuple[int, int],
    blk_size: int,
) -> list[tuple[int, str, str, str]]:
    """Return [(offset, name, color, interp), ...] for the known fields of eff_type.

    interp: "ptr" | "f64" | "i64" | "i32"
    Colors: bright_green=refcount, bright_blue=ob_type, dim cyan=gc fields,
            yellow=size/count, cyan=pointer-to-other-objects, white=scalar data.
    """
    gc_hdr: list[tuple[int, str, str, str]] = [
        (0, "_gc_next", "dim cyan", "ptr"),
        (8, "_gc_prev", "dim cyan", "ptr"),
        (16, "ob_refcnt", "bright_green", "i64"),
        (24, "ob_type", "bright_blue", "ptr"),
    ]
    nongc_hdr: list[tuple[int, str, str, str]] = [
        (0, "ob_refcnt", "bright_green", "i64"),
        (8, "ob_type", "bright_blue", "ptr"),
    ]

    if eff_type == "float":
        return nongc_hdr + [(16, "ob_fval", "white", "f64")]

    if eff_type == "bytes":
        return nongc_hdr + [
            (16, "ob_size", "yellow", "i64"),
            (24, "ob_hash", "dim", "i64"),
            (32, "ob_sval[0:8]", "white", "i64"),
        ]

    if eff_type == "str":
        return nongc_hdr + [
            (16, "ob_length", "yellow", "i64"),
            (24, "ob_hash", "dim", "i64"),
            (32, "ob_state", "dim", "i32"),
        ]

    if eff_type == "int":
        tag_name = "lv_tag" if py_ver >= (3, 12) else "ob_size"
        fields = nongc_hdr + [(16, tag_name, "yellow", "i64")]
        # Two i32 digits fit per 8-byte chunk; label each pair together.
        for i in range(max(0, blk_size - 24) // 8):
            fields.append((24 + i * 8, f"digit[{2 * i},{2 * i + 1}]", "white", "i64"))
        return fields

    if eff_type in ("list", "tuple"):
        fields = gc_hdr + [(32, "ob_size", "yellow", "i64")]
        if eff_type == "list":
            fields += [
                (40, "*ob_item", "cyan", "ptr"),
                (48, "allocated", "dim", "i64"),
            ]
        else:
            n_items = (blk_size - 40) // 8
            for i in range(n_items):
                fields.append((40 + i * 8, f"ob_item[{i}]", "cyan", "ptr"))
        return fields

    if eff_type == "dict":
        return gc_hdr + [
            (32, "ma_used", "yellow", "i64"),
            (40, "ma_version_tag", "dim", "i64"),
            (48, "*ma_keys", "cyan", "ptr"),
            (56, "*ma_values", "cyan", "ptr"),
        ]

    if eff_type in ("set", "frozenset"):
        return gc_hdr + [
            (32, "fill", "dim", "i64"),
            (40, "used", "yellow", "i64"),
            (48, "mask", "dim", "i64"),
            (56, "*table", "cyan", "ptr"),
            (64, "hash", "dim", "i64"),
            (72, "finger", "dim", "i64"),
        ]

    if eff_type == "code":
        if py_ver >= (3, 13):
            # 3.13: Py_TPFLAGS_HAVE_GC removed; no PyGC_Head prefix.
            # All offsets shift 16 bytes earlier relative to block start.
            return nongc_hdr + [
                (16, "?", "dim", "i64"),
                (24, "*co_filename", "cyan", "ptr"),
                (32, "*co_name", "cyan", "ptr"),
            ]
        if py_ver >= (3, 11):
            # 3.11-3.12: reorganised PyCodeObject; key pointers near the start
            return gc_hdr + [
                (40, "*co_filename", "cyan", "ptr"),
                (48, "*co_name", "cyan", "ptr"),
            ]
        else:
            # 3.10-: i32 fields first, then pointers
            return gc_hdr + [
                (32, "co_argcount", "yellow", "i32"),
                (36, "co_posonlyargcount", "dim", "i32"),
                (40, "co_kwonlyargcount", "dim", "i32"),
                (44, "co_nlocals", "dim", "i32"),
                (48, "co_stacksize", "dim", "i32"),
                (52, "co_flags", "dim", "i32"),
                (56, "co_firstlineno", "dim", "i32"),
                (64, "*co_code", "cyan", "ptr"),
                (72, "*co_consts", "cyan", "ptr"),
                (80, "*co_names", "dim", "ptr"),
                (88, "*co_varnames", "dim", "ptr"),
                (96, "*co_filename", "cyan", "ptr"),
                (104, "*co_name", "cyan", "ptr"),
            ]

    if eff_type == "frame":
        return gc_hdr + [
            (32, "*f_back", "cyan", "ptr"),
            (40, "*f_frame", "cyan", "ptr"),
        ]

    if eff_type == "complex":
        # PyComplexObject — non-GC, stable across all 3.x
        return nongc_hdr + [
            (16, "cval.real", "white", "f64"),
            (24, "cval.imag", "white", "f64"),
        ]

    if eff_type == "range":
        # PyRangeObject — non-GC, stores ints directly (not pointers)
        return nongc_hdr + [
            (16, "start", "white", "i64"),
            (24, "stop", "white", "i64"),
            (32, "step", "white", "i64"),
            (40, "length", "dim", "i64"),
        ]

    if eff_type == "cell":
        # PyCellObject — GC-tracked; single payload field
        return gc_hdr + [
            (32, "*cell_contents", "cyan", "ptr"),
        ]

    if eff_type == "module":
        # PyModuleObject — GC-tracked; md_name added in 3.3
        fields = gc_hdr + [
            (32, "*md_dict", "cyan", "ptr"),
            (40, "*md_state", "dim", "ptr"),
            (48, "*md_weaklist", "dim", "ptr"),
        ]
        if py_ver >= (3, 3):
            fields += [(56, "*md_name", "cyan", "ptr")]
        return fields

    if eff_type == "bytearray":
        # PyByteArrayObject — non-GC; ob_exports (i32) added in 3.9
        fields = nongc_hdr + [
            (16, "ob_size", "yellow", "i64"),
            (24, "ob_alloc", "dim", "i64"),
            (32, "*ob_bytes", "cyan", "ptr"),
        ]
        if py_ver >= (3, 9):
            fields += [(40, "ob_exports", "dim", "i32")]
        return fields

    if eff_type == "slice":
        # PySliceObject — non-GC through 3.12, GC-tracked in 3.13+
        hdr = gc_hdr if py_ver >= (3, 13) else nongc_hdr
        base = 32 if py_ver >= (3, 13) else 16
        return hdr + [
            (base, "*start", "cyan", "ptr"),
            (base + 8, "*stop", "cyan", "ptr"),
            (base + 16, "*step", "cyan", "ptr"),
        ]

    if eff_type == "weakref":
        # PyWeakReference — GC-tracked (Py_TPFLAGS_HAVE_GC set in all 3.x)
        return gc_hdr + [
            (32, "*wr_object", "cyan", "ptr"),
            (40, "*wr_callback", "cyan", "ptr"),
            (48, "hash", "dim", "i64"),
            (56, "*wr_prev", "dim", "ptr"),
            (64, "*wr_next", "dim", "ptr"),
        ]

    if eff_type == "enumerate":
        # PyEnumObject — GC-tracked
        return gc_hdr + [
            (32, "*en_iter", "cyan", "ptr"),
            (40, "*en_func", "dim", "ptr"),
            (48, "en_index", "yellow", "i64"),
        ]

    if eff_type == "memoryview":
        # PyMemoryViewObject — GC-tracked
        return gc_hdr + [
            (32, "*obj", "cyan", "ptr"),
            (40, "*buf", "dim", "ptr"),
            (48, "hash", "dim", "i64"),
            (56, "flags", "dim", "i32"),
        ]

    if eff_type == "method":
        # PyMethodObject — non-GC in 3.10, GC-tracked in 3.11+
        if py_ver >= (3, 11):
            return gc_hdr + [
                (32, "*im_func", "cyan", "ptr"),
                (40, "*im_self", "cyan", "ptr"),
                (48, "*im_weakreflist", "dim", "ptr"),
            ]
        else:
            return nongc_hdr + [
                (16, "*im_func", "cyan", "ptr"),
                (24, "*im_self", "cyan", "ptr"),
            ]

    if eff_type == "function":
        # PyFunctionObject — GC-tracked; layout varies by version
        # +32 and +40 are stable; beyond that shifts occur between versions
        fields = gc_hdr + [
            (32, "*func_code", "cyan", "ptr"),
            (40, "*func_globals", "cyan", "ptr"),
        ]
        if py_ver >= (3, 11):
            fields += [
                (48, "*func_closure", "cyan", "ptr"),
                (56, "func_doc", "dim", "ptr"),
                (64, "*func_dict", "dim", "ptr"),
                (72, "*func_weakref", "dim", "ptr"),
                (80, "func_name", "cyan", "ptr"),
                (88, "func_qualname", "dim", "ptr"),
            ]
        else:
            fields += [
                (48, "*func_builtins", "dim", "ptr"),
                (56, "*func_closure", "cyan", "ptr"),
                (64, "func_doc", "dim", "ptr"),
                (72, "*func_dict", "dim", "ptr"),
                (80, "*func_weakref", "dim", "ptr"),
                (88, "func_name", "cyan", "ptr"),
                (96, "func_qualname", "dim", "ptr"),
            ]
        return fields

    if eff_type in ("dict_keys", "dict_values", "dict_items"):
        # dictviewobject — GC-tracked; single payload field (parent dict)
        return gc_hdr + [
            (32, "*dv_dict", "cyan", "ptr"),
        ]

    if eff_type in ("method_descriptor", "member_descriptor", "getset_descriptor"):
        # PyDescrObject subtypes — GC-tracked; stable layout 3.10-3.13.
        # PyDescr_COMMON: d_type(+32), d_name(+40), d_qualname(+48).
        payload_name = {
            "method_descriptor": "*d_method",
            "member_descriptor": "*d_member",
            "getset_descriptor": "*d_getset",
        }[eff_type]
        fields = gc_hdr + [
            (32, "*d_type", "cyan", "ptr"),
            (40, "*d_name", "cyan", "ptr"),
            (48, "*d_qualname", "dim", "ptr"),
            (56, payload_name, "dim", "ptr"),
        ]
        if eff_type == "method_descriptor":
            fields.append((64, "vectorcall", "dim", "ptr"))
        return fields

    if eff_type == "wrapper_descriptor":
        # PyWrapperDescrObject — GC-tracked; wraps a C slot function
        return gc_hdr + [
            (32, "*d_type", "cyan", "ptr"),
            (40, "*d_name", "cyan", "ptr"),
            (48, "*d_qualname", "dim", "ptr"),
            (56, "*d_wrapped", "dim", "ptr"),
        ]

    if eff_type == "builtin_function_or_method":
        # PyCFunctionObject — GC-tracked
        return gc_hdr + [
            (32, "*m_ml", "dim", "ptr"),
            (40, "*m_self", "cyan", "ptr"),
            (48, "*m_module", "cyan", "ptr"),
            (56, "*m_weakreflist", "dim", "ptr"),
            (64, "vectorcall", "dim", "ptr"),
        ]

    # Generic fallback: just annotate the header
    return gc_hdr if is_gc else nongc_hdr


def _clamp_viewport(vp_start: int, cursor: int, height: int, total: int) -> int:
    """Return a viewport start that keeps cursor visible."""
    height = max(1, height)
    vp_start = min(vp_start, max(0, total - height))
    if cursor < vp_start:
        vp_start = cursor
    elif cursor >= vp_start + height:
        vp_start = cursor - height + 1
    return max(0, vp_start)
