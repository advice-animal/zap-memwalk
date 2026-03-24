"""Interactive TUI for zap-memwalk using rich.Live.

Two hierarchy modes (Tab to toggle):
  size mode:  size-class → pools in that size class → blocks
  arena mode: arena       → pools in that arena       → blocks

Navigation:
  Top view  ──Enter──>  Pool view  ──Enter──>  Block view
      ^                     ^                      ^
      └──── Escape / b ─────┘                      │
                  └──────────── Escape / b ─────────┘

In the block view, press  r  on a selected (live) block to repr it.
"""

from __future__ import annotations

import queue
import shutil
import sys
import time
from enum import Enum, auto
from typing import Any

from rich import box
from rich.console import Console, Group
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


def _ob_type_candidates(blk: bytes) -> tuple[int, int]:
    """Return (ptr_at_8, ptr_at_24) from raw block bytes.

    ptr_at_8  is ob_type for non-GC objects (no PyGC_Head prefix).
    ptr_at_24 is ob_type for GC-tracked objects (16-byte PyGC_Head at block+0).
    """
    import struct

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
        self._col = collector
        self._interval = interval
        self._snap: MemorySnapshot | None = None
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

        # Jump-to-address input state (activated by '/')
        self._jump_buf: str | None = (
            None  # None = not in jump mode; str = current input
        )
        self._jump_err: str = ""  # shown after a failed jump

        self._keys: queue.SimpleQueue[str] = queue.SimpleQueue()

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
        t = Table(box=box.SIMPLE_HEAD, show_edge=False)
        t.add_column("", width=2, no_wrap=True)
        t.add_column("offset", justify="right", style="dim", no_wrap=True)
        t.add_column("address", style="cyan", no_wrap=True)
        t.add_column("state", no_wrap=True, min_width=6)
        t.add_column("type", no_wrap=True, max_width=10)
        t.add_column("hex / repr", no_wrap=False)

        if self._sel_pool is None or self._pool_raw is None:
            return t

        pool = self._sel_pool
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", pool.szidx)
        nextoffset = self._pool_raw.get("nextoffset", pool.nextoffset)
        block_size = (szidx + 1) * 16

        free_set = frozenset(int(a, 16) for a in self._pool_raw.get("freeAddrs", []))
        now = time.monotonic()

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

        for row_i, (off, addr, state) in enumerate(
            blocks[self._vp_start : self._vp_start + height]
        ):
            abs_i = row_i + self._vp_start
            cursor_mark = "▶" if abs_i == self._cursor else " "
            blk_bytes = raw[off : off + block_size]

            if state == BlockState.UNBORN:
                state_text = Text("unborn", style="dim")
                type_text = Text(_cstr_hint(blk_bytes), style="italic dim")
                content = Text(_hex_bytes(blk_bytes[:16]), style="dim")
            elif state == BlockState.FREE:
                state_text = Text("free", style="bright_black")
                raw_type = self._type_names.get(addr, "")
                eff_type = (
                    raw_type
                    if (raw_type and raw_type != "?")
                    else _cstr_hint(blk_bytes)
                )
                type_text = Text(eff_type, style="bright_black")
                content = Text(_hex_bytes(blk_bytes[:16]), style="bright_black")
            else:
                color = self._age.color(addr, now)
                state_text = Text("live", style=color)
                raw_type = self._type_names.get(addr, "")
                eff_type = (
                    raw_type
                    if (raw_type and raw_type != "?")
                    else _cstr_hint(blk_bytes)
                )
                type_text = Text(eff_type, style="dim")
                content = Text(_hex_bytes(blk_bytes[:16]), style=color, no_wrap=True)

            t.add_row(
                cursor_mark,
                f"+{off:04x}",
                f"0x{addr:x}",
                state_text,
                type_text,
                content,
            )
        return t

    def _make_header(self) -> Text:
        snap = self._snap
        if snap is None:
            return Text("zap-memwalk — connecting…", style="bold")
        pid_info = f"PID {snap.pid}"
        ver_info = f"Python {snap.py_version[0]}.{snap.py_version[1]}"
        pool_info = f"{snap.pool_size // 1024}KiB pools"

        if self._hier_mode == "arena":
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
        elif self._jump_err:
            second_line = f"↑↓=move  Enter=drill  b/Esc=back  Tab=mode  /=jump  r=repr  q=quit  [{self._jump_err}]"
        else:
            base = "↑↓=move  Enter=drill  b/Esc=back  Tab=mode  /=jump  r=repr  q=quit"
            if self._rpc_lookups > 0:
                second_line = f"{base}  [lookups: {self._rpc_lookups}]"
            else:
                second_line = base
        return Text.assemble(
            ("zap-memwalk", "bold"),
            "  ",
            (pid_info, "cyan"),
            "  ",
            (ver_info, "dim"),
            "  ",
            (pool_info, "dim"),
            "  │  ",
            (view_crumb, "bold cyan"),
            "\n",
            (second_line, "bold yellow" if self._jump_buf is not None else "dim"),
        )

    def _make_detail_panel(self) -> Group:
        """Fixed 6-line panel at the bottom of the block view."""
        if self._sel_pool is None or self._pool_raw is None:
            return Group(Rule(style="dim"), Text(""))

        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        free_set = frozenset(int(a, 16) for a in self._pool_raw.get("freeAddrs", []))

        blocks = [
            (off, self._sel_pool.address + off)
            for off in range(_POOL_OVERHEAD, len(raw), block_size)
        ]
        if self._cursor >= len(blocks):
            return Group(Rule(style="dim"), Text(""))

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

        _CONTENT_LINES = 3
        repr_map = {a: t for a, t in self._repr_lines}
        if addr in repr_map:
            lines = repr_map[addr].splitlines()[:_CONTENT_LINES]
        elif eff_type == "cstr":
            cstr_val = _extract_cstr(blk_bytes)
            lines = ([repr(cstr_val)] if cstr_val is not None else [])[:_CONTENT_LINES]
        elif state_str != "live" and addr in self._name_hints:
            lines = [f"(freed {eff_type}) {self._name_hints[addr]}"]
        else:
            # Auto fallback repr for known types with inline data
            py_ver = self._snap.py_version if self._snap else (3, 10)
            fallback = _fallback_repr_from_raw(blk_bytes, raw_type, py_ver)
            if fallback and state_str != "live":
                lines = fallback.splitlines()[:_CONTENT_LINES]
            else:
                lines = _hex_bytes(blk_bytes[:48]).splitlines()[:_CONTENT_LINES]
        # Pad to exactly _CONTENT_LINES so the status line is always last.
        lines += [""] * (_CONTENT_LINES - len(lines))
        content = Text("\n".join(lines), style="dim", overflow="crop", no_wrap=True)

        if state_str == "live":
            status = Text(
                f"ctypes.cast(0x{addr:x}, ctypes.py_object).value",
                style="dim",
                no_wrap=True,
            )
        else:
            status = Text(
                f"ctypes.cast(0x{addr:x}, ctypes.py_object).value  [{state_str}]",
                style="dim",
                no_wrap=True,
            )

        ptr8, ptr24 = _ob_type_candidates(blk_bytes)
        # Prefer whichever candidate was resolved by _resolve_block_types.
        ob_type_ptr = (
            ptr24
            if (ptr24 in self._ob_type_syms and ptr8 not in self._ob_type_syms)
            else ptr8
        )
        if ob_type_ptr == 0:
            ob_type_line = Text("ob_type → 0x0", style="dim", no_wrap=True)
        elif ob_type_ptr in self._ob_type_syms:
            ob_type_line = Text(
                f"ob_type → {self._ob_type_syms[ob_type_ptr]}",
                style="dim",
                no_wrap=True,
            )
        else:
            ob_type_line = Text(
                f"ob_type → 0x{ob_type_ptr:x}", style="dim", no_wrap=True
            )

        return Group(rule, content, status, ob_type_line)

    def _render(self) -> Group:
        h = shutil.get_terminal_size().lines
        if self._view == _View.BLOCK:
            # 2 header + 2 table header/sep + 1 rule + 5 detail = 10
            vp_h = max(1, h - 10)
            body = self._render_block_view(vp_h)
            return Group(self._make_header(), body, self._make_detail_panel())
        else:
            # 2 header + 2 table header/sep
            vp_h = max(1, h - 4)
            if self._view == _View.SIZE_CLASS:
                body = (
                    self._render_arena_view(vp_h)
                    if self._hier_mode == "arena"
                    else self._render_size_class_view(vp_h)
                )
            else:
                body = (
                    self._render_arena_pool_view(vp_h)
                    if self._hier_mode == "arena"
                    else self._render_pool_view(vp_h)
                )
            return Group(self._make_header(), body)

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
        unique_sym_ptrs = [p for _, p in winning_ptrs if p not in self._ob_type_syms]
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
                    if symbol:
                        self._ob_type_syms[p] = f"{mod}!{symbol}"
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
                        if result:
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
                            if result:
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
                    if result:
                        import re

                        m = re.search(r"<code object ([^\s>]+)", result[1])
                        if m:
                            self._name_hints[addr] = m.group(1)
                except Exception:
                    pass

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
        if self._view == _View.BLOCK and self._sel_pool is not None:
            self._refresh_block_view()
        else:
            self._refresh_full()

    def _refresh_full(self) -> None:
        """Full arena scan — used for size-class and pool views."""
        snap = self._col.collect()
        now = snap.ts
        live: set[int] = set()
        for sc in snap.size_classes:
            for pool in sc.pools:
                for addr in pool.iter_block_addresses():
                    if pool.block_state(addr) == BlockState.LIVE:
                        live.add(addr)
        self._age.update(live, now)
        self._snap = snap

    def _refresh_block_view(self) -> None:
        """Fast path: re-read only the current pool (no full arena scan)."""
        assert self._sel_pool is not None
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
        free_set = frozenset(int(a, 16) for a in self._pool_raw.get("freeAddrs", []))
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
        if self._view == _View.BLOCK and self._sel_pool is not None:
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

    def _back(self) -> None:
        if self._view == _View.BLOCK:
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

        else:  # BLOCK — just flip the flag, keeping _sel_pool and cursor unchanged.
            if target == "arena" and snap is not None and self._sel_pool is not None:
                for a in snap.arenas:
                    if a.arena_index == self._sel_pool.arena_index:
                        self._sel_arena_idx = a.arena_index
                        break
            elif target == "size" and self._sel_pool is not None:
                self._sel_szidx = self._sel_pool.szidx

        self._hier_mode = target

    def _do_repr(self) -> None:
        """Repr selected block (block view only)."""
        if (
            self._view != _View.BLOCK
            or self._sel_pool is None
            or self._pool_raw is None
        ):
            return
        szidx = self._pool_raw.get("szidx", self._sel_pool.szidx)
        block_size = (szidx + 1) * 16
        nextoffset = self._pool_raw.get("nextoffset", self._sel_pool.nextoffset)
        raw = bytes(self._pool_raw.get("raw", b"") or b"")
        free_set = frozenset(int(a, 16) for a in self._pool_raw.get("freeAddrs", []))

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
        eff_type = raw_type if (raw_type and raw_type != "?") else _cstr_hint(blk_bytes)

        if eff_type == "cstr":
            # Display null-terminated string; never call repr_block on cstr data.
            cstr_val = _extract_cstr(blk_bytes)
            line = (
                f"0x{addr:x} [cstr]: {repr(cstr_val)}"
                if cstr_val is not None
                else f"0x{addr:x} [cstr]: (empty)"
            )
        elif off >= nextoffset or addr in free_set:
            if addr in self._name_hints:
                line = f"0x{addr:x} [freed {eff_type}]: {self._name_hints[addr]}"
            else:
                py_ver = self._snap.py_version if self._snap else (3, 10)
                fallback = _fallback_repr_from_raw(blk_bytes, raw_type, py_ver)
                line = (
                    f"0x{addr:x}: {fallback}"
                    if fallback
                    else f"0x{addr:x}: (free / unborn — cannot repr)"
                )
        else:
            result = self._col.repr_block(addr)
            if result is not None:
                type_name, repr_str = result
                line = f"0x{addr:x} [{type_name}]: {repr_str}"
            else:
                line = f"0x{addr:x}: repr failed"
        # Replace existing repr for this address or append
        self._repr_lines = [(a, t) for a, t in self._repr_lines if a != addr]
        self._repr_lines.append((addr, line))

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
        if ch == "\x1b[A" or ch == "k":  # up
            self._cursor = max(0, self._cursor - 1)
        elif ch == "\x1b[B" or ch == "j":  # down
            self._cursor = min(max(0, n - 1), self._cursor + 1)
        elif ch == "n" or ch == "\x1b[6~":  # page down
            self._cursor = min(max(0, n - 1), self._cursor + page)
        elif ch == "p" or ch == "\x1b[5~":  # page up
            self._cursor = max(0, self._cursor - page)
        elif ch in ("\r", "\n", "o"):  # Enter / o
            self._enter()
        elif ch in ("b", "\x1b"):  # back / Escape
            self._back()
        elif ch == "/":
            self._jump_buf = ""
            self._jump_err = ""
        elif ch == "\t":
            self._toggle_hier_mode()
        elif ch == "r":
            self._do_repr()
        elif ch == "R":
            self._refresh()
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

        t = threading.Thread(target=self._key_reader, daemon=True)
        t.start()

        console = Console()
        last_refresh = time.monotonic()

        with Live(
            self._render(), console=console, screen=True, auto_refresh=False
        ) as live:
            try:
                while True:
                    # Drain the key queue
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

                    # Periodic refresh
                    now = time.monotonic()
                    if now - last_refresh >= self._interval:
                        try:
                            self._refresh()
                        except Exception:
                            pass
                        last_refresh = now

                    live.update(self._render())
                    live.refresh()
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass


# ── helpers ───────────────────────────────────────────────────────────────────


def _clamp_viewport(vp_start: int, cursor: int, height: int, total: int) -> int:
    """Return a viewport start that keeps cursor visible."""
    height = max(1, height)
    vp_start = min(vp_start, max(0, total - height))
    if cursor < vp_start:
        vp_start = cursor
    elif cursor >= vp_start + height:
        vp_start = cursor - height + 1
    return max(0, vp_start)
