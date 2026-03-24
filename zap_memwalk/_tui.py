"""Interactive TUI for zap-memwalk using rich.Live.

Navigation:
  Size-class view  ──Enter──>  Pool view  ──Enter──>  Block view
       ^                           ^                       ^
       └──── Escape / b ───────────┘                       │
                   └──────────────── Escape / b ───────────┘

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

        self._view = _View.SIZE_CLASS
        self._cursor = 0  # row cursor in the current view
        self._vp_start = 0  # viewport first row

        # Selected size class / pool for drill-down
        self._sel_szidx: int | None = None
        self._sel_pool: PoolSnapshot | None = None

        # Block view state
        self._pool_raw: dict[str, Any] | None = None  # from read_pool()
        self._repr_lines: list[tuple[int, str]] = []  # (block_addr, repr_text)
        self._type_names: dict[int, str] = {}  # block_addr -> type name
        self._name_hints: dict[
            int, str
        ] = {}  # block_addr -> identifying string (code/module)

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

    def _render_pool_view(self, height: int) -> Table:
        t = Table(box=box.SIMPLE_HEAD, show_edge=False)
        t.add_column("pool address", style="cyan", no_wrap=True)
        t.add_column("used", justify="right", no_wrap=True)
        t.add_column("total", justify="right", no_wrap=True)
        t.add_column("fill%", justify="right", no_wrap=True)
        t.add_column("fill", no_wrap=True, min_width=16)

        if self._snap is None or self._sel_szidx is None:
            return t

        sc = self._snap.size_classes[self._sel_szidx]
        pools = sc.pools
        total_rows = len(pools)

        self._vp_start = _clamp_viewport(
            self._vp_start, self._cursor, height, total_rows
        )

        for row_i, pool in enumerate(pools[self._vp_start : self._vp_start + height]):
            abs_i = row_i + self._vp_start
            cursor_mark = "▶" if abs_i == self._cursor else " "
            t.add_row(
                f"{cursor_mark} 0x{pool.address:016x}",
                str(pool.ref_count),
                str(pool.total_blocks),
                _pct_text(pool.fill_pct),
                _fill_bar(pool.fill_pct, 16),
            )
        return t

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

        view_crumb = {
            _View.SIZE_CLASS: "size classes",
            _View.POOL: f"size-class {self._sel_szidx} ({((self._sel_szidx or 0) + 1) * 16}B pools)",
            _View.BLOCK: (
                f"pool 0x{self._sel_pool.address:x}"
                f"  sz{self._sel_pool.szidx} ({self._sel_pool.block_size}B)"
                if self._sel_pool
                else ""
            ),
        }[self._view]

        if self._jump_buf is not None:
            second_line = f"jump to address: {self._jump_buf}█  (0x… hex or decimal  Enter=go  Esc=cancel)"
        elif self._jump_err:
            second_line = f"↑↓=move  Enter=drill  b/Esc=back  /=jump  r=repr  q=quit  [{self._jump_err}]"
        else:
            second_line = "↑↓=move  Enter=drill  b/Esc=back  /=jump  r=repr  q=quit"
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
        """Fixed 5-line panel at the bottom of the block view."""
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

        return Group(rule, content, status)

    def _render(self) -> Group:
        h = shutil.get_terminal_size().lines
        if self._view == _View.BLOCK:
            # 2 header + 2 table header/sep + 1 rule + 4 detail = 9
            vp_h = max(1, h - 9)
            body = self._render_block_view(vp_h)
            return Group(self._make_header(), body, self._make_detail_panel())
        else:
            # 2 header + 2 table header/sep
            vp_h = max(1, h - 4)
            if self._view == _View.SIZE_CLASS:
                body = self._render_size_class_view(vp_h)
            else:
                body = self._render_pool_view(vp_h)
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

        # Collect (block_addr, ob_type_ptr) for live AND free blocks.
        # When a block is freed, CPython overwrites only offset 0-7 (ob_refcnt)
        # with the free-list link; offset 8-15 (ob_type) is left as stale data.
        pairs: list[tuple[int, int]] = []
        for off in range(_POOL_OVERHEAD, len(raw), block_size):
            addr = self._sel_pool.address + off
            if off >= nextoffset:
                continue  # unborn — never had a type
            if off + 16 > len(raw):
                continue
            ob_type_ptr = struct.unpack_from("<Q", raw, off + 8)[0]
            if ob_type_ptr:
                pairs.append((addr, ob_type_ptr))

        if not pairs:
            return

        unique_ptrs = list({p for _, p in pairs})
        try:
            resolved = self._col.resolve_type_names(unique_ptrs)
        except Exception:
            return

        for addr, ob_type_ptr in pairs:
            name = resolved.get(ob_type_ptr) or None
            if name:
                self._type_names[addr] = name
            elif addr not in self._type_names:
                self._type_names[addr] = "?"

        # For code objects, eagerly resolve co_name + co_filename via repr_block.
        # Python 3.11+ moved co_filename from +96 to +40 when the struct was reorganised.
        py_ver = self._snap.py_version if self._snap else (3, 10)
        if py_ver >= (3, 11):
            co_name_off, co_filename_off = 48, 40
        else:
            co_name_off, co_filename_off = 104, 96

        for addr, _ in pairs:
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
            return sum(1 for sc in self._snap.size_classes if sc.pool_count > 0)
        if self._view == _View.POOL and self._sel_szidx is not None:
            return len(self._snap.size_classes[self._sel_szidx].pools)
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
        for sc in self._snap.size_classes:
            for pool_idx, pool in enumerate(sc.pools):
                if pool.address == pool_addr:
                    # Navigate: switch to pool view for this size class
                    self._sel_szidx = sc.szidx
                    self._view = _View.POOL
                    self._cursor = pool_idx
                    self._vp_start = 0
                    # Immediately drill into block view
                    self._sel_pool = pool
                    try:
                        self._pool_raw = self._col.read_pool(pool.address)
                    except Exception:
                        self._pool_raw = None
                    self._repr_lines = []
                    self._type_names = {}
                    self._name_hints = {}
                    self._resolve_block_types()
                    self._view = _View.BLOCK
                    self._vp_start = 0
                    # Position cursor on the target block
                    block_size = (sc.szidx + 1) * 16
                    off = addr - pool_addr
                    if (
                        off >= _POOL_OVERHEAD
                        and (off - _POOL_OVERHEAD) % block_size == 0
                    ):
                        self._cursor = (off - _POOL_OVERHEAD) // block_size
                    else:
                        self._cursor = 0
                    return

        raise RuntimeError(
            f"address 0x{addr:x} → pool 0x{pool_addr:x} not found in snapshot"
        )

    def _enter(self) -> None:
        if self._snap is None:
            return
        if self._view == _View.SIZE_CLASS:
            active = [sc for sc in self._snap.size_classes if sc.pool_count > 0]
            if self._cursor < len(active):
                self._sel_szidx = active[self._cursor].szidx
                self._view = _View.POOL
                self._cursor = 0
                self._vp_start = 0
        elif self._view == _View.POOL and self._sel_szidx is not None:
            pools = self._snap.size_classes[self._sel_szidx].pools
            if self._cursor < len(pools):
                self._sel_pool = pools[self._cursor]
                try:
                    self._pool_raw = self._col.read_pool(self._sel_pool.address)
                except Exception:
                    self._pool_raw = None
                self._repr_lines = []
                self._type_names = {}
                self._name_hints = {}
                self._resolve_block_types()
                self._view = _View.BLOCK
                self._cursor = 0
                self._vp_start = 0

    def _back(self) -> None:
        if self._view == _View.BLOCK:
            # Return to pool view; restore cursor to the pool we were viewing.
            cursor = 0
            if (
                self._snap is not None
                and self._sel_szidx is not None
                and self._sel_pool is not None
            ):
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
        elif self._view == _View.POOL:
            # Return to size-class view; restore cursor to the size class we were in.
            cursor = 0
            if self._snap is not None and self._sel_szidx is not None:
                active = [sc for sc in self._snap.size_classes if sc.pool_count > 0]
                for i, sc in enumerate(active):
                    if sc.szidx == self._sel_szidx:
                        cursor = i
                        break
            self._view = _View.SIZE_CLASS
            self._cursor = cursor
            self._vp_start = 0

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
        page = max(1, h - 9)
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
        import os
        import select as _sel
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
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
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

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
