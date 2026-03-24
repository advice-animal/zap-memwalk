"""Unit tests for _model.py — no Frida required."""

from __future__ import annotations

import time

from zap_memwalk._model import (
    BlockAgeTracker,
    BlockState,
    MemorySnapshot,
    PoolSnapshot,
    SizeClassSummary,
)
from zap_memwalk._tui import _fill_bar, _clamp_viewport


def _make_pool(
    address: int = 0x1000_0000,
    szidx: int = 1,
    ref_count: int = 100,
    nextoffset: int = 3248,   # some bytes into pool
    maxnextoffset: int = 16320,
    free_addresses: frozenset[int] = frozenset(),
) -> PoolSnapshot:
    block_size = (szidx + 1) * 16
    return PoolSnapshot(
        address=address,
        arena_index=0,
        szidx=szidx,
        block_size=block_size,
        ref_count=ref_count,
        nextoffset=nextoffset,
        maxnextoffset=maxnextoffset,
        free_addresses=free_addresses,
    )


class TestPoolSnapshot:
    def test_pool_size_derived_from_maxnextoffset(self):
        pool = _make_pool(szidx=1, maxnextoffset=16320)
        # pool_size = maxnextoffset + POOL_OVERHEAD + block_size = 16320 + 48 + 32 = 16400
        # but canonical is 16384; maxnextoffset = 16384 - 48 - 32 = 16304
        pool2 = _make_pool(szidx=1, maxnextoffset=16304)
        assert pool2.pool_size == 16384

    def test_total_blocks(self):
        pool = _make_pool(szidx=1, maxnextoffset=16304)
        # (16384 - 48) // 32 = 510
        assert pool.total_blocks == 510

    def test_fill_pct(self):
        pool = _make_pool(szidx=1, ref_count=255, maxnextoffset=16304)
        assert abs(pool.fill_pct - 255 / 510 * 100) < 0.01

    def test_fill_pct_zero_when_no_blocks(self):
        pool = _make_pool(szidx=1, ref_count=0, maxnextoffset=0)
        assert pool.fill_pct == 0.0

    def test_block_state_live(self):
        addr = 0x1000_0030   # offset 48 from pool at 0x1000_0000
        free = frozenset({0x1000_0050})
        pool = _make_pool(address=0x1000_0000, szidx=1,
                          nextoffset=512, free_addresses=free)
        assert pool.block_state(addr) == BlockState.LIVE

    def test_block_state_free(self):
        addr = 0x1000_0050
        free = frozenset({addr})
        pool = _make_pool(address=0x1000_0000, szidx=1,
                          nextoffset=512, free_addresses=free)
        assert pool.block_state(addr) == BlockState.FREE

    def test_block_state_unborn(self):
        pool = _make_pool(address=0x1000_0000, szidx=1, nextoffset=96)
        # offset 96 is >= nextoffset=96
        addr = 0x1000_0000 + 96
        assert pool.block_state(addr) == BlockState.UNBORN

    def test_iter_block_addresses(self):
        pool = _make_pool(address=0x1000_0000, szidx=0, maxnextoffset=16320)
        # block_size = 16; pool_size = 16320 + 48 + 16 = 16384
        addrs = pool.iter_block_addresses()
        assert addrs[0] == 0x1000_0000 + 48
        assert (addrs[1] - addrs[0]) == 16


class TestSizeClassSummary:
    def test_aggregation(self):
        p1 = _make_pool(ref_count=100, maxnextoffset=16304)
        p2 = _make_pool(ref_count=200, maxnextoffset=16304)
        sc = SizeClassSummary(szidx=1, block_size=32, pools=[p1, p2])
        assert sc.pool_count == 2
        assert sc.used_blocks == 300
        assert sc.total_blocks == 1020
        assert abs(sc.fill_pct - 300 / 1020 * 100) < 0.01

    def test_empty_class(self):
        sc = SizeClassSummary(szidx=5, block_size=96, pools=[])
        assert sc.pool_count == 0
        assert sc.fill_pct == 0.0


class TestMemorySnapshot:
    def _make_snap(self) -> MemorySnapshot:
        p = _make_pool(ref_count=50, maxnextoffset=16304)
        sc = SizeClassSummary(szidx=1, block_size=32, pools=[p])
        empties = [SizeClassSummary(szidx=i, block_size=(i + 1) * 16) for i in range(32)]
        empties[1] = sc
        return MemorySnapshot(
            pid=999,
            ts=1.0,
            pool_size=16384,
            arena_size=1048576,
            py_version=(3, 13),
            size_classes=empties,
        )

    def test_to_dict_structure(self):
        snap = self._make_snap()
        d = snap.to_dict()
        assert d["pid"] == 999
        assert d["py_version"] == [3, 13]
        assert len(d["size_classes"]) == 1   # only non-empty classes
        assert d["size_classes"][0]["szidx"] == 1

    def test_total_pools(self):
        snap = self._make_snap()
        assert snap.total_pools == 1

    def test_total_used_blocks(self):
        snap = self._make_snap()
        assert snap.total_used_blocks == 50


class TestBlockAgeTracker:
    def test_new_block_is_yellow(self):
        tracker = BlockAgeTracker()
        now = time.monotonic()
        tracker.update({0xABC}, now)
        assert tracker.color(0xABC, now) == "yellow"

    def test_old_block_turns_red(self):
        tracker = BlockAgeTracker()
        past = time.monotonic() - 70
        tracker._first_seen[0xABC] = past
        assert tracker.color(0xABC, time.monotonic()) == "bold red"

    def test_medium_age(self):
        tracker = BlockAgeTracker()
        past = time.monotonic() - 5
        tracker._first_seen[0xABC] = past
        assert tracker.color(0xABC, time.monotonic()) == "orange1"

    def test_pruned_when_gone(self):
        tracker = BlockAgeTracker()
        now = time.monotonic()
        tracker.update({0xABC, 0xDEF}, now)
        tracker.update({0xABC}, now + 1)
        assert 0xDEF not in tracker._first_seen

    def test_first_seen_not_overwritten(self):
        tracker = BlockAgeTracker()
        t0 = time.monotonic()
        tracker.update({0xABC}, t0)
        tracker.update({0xABC}, t0 + 10)
        assert tracker._first_seen[0xABC] == t0


class TestTuiHelpers:
    def test_fill_bar_full(self):
        bar = _fill_bar(100.0, width=10)
        assert "█" * 10 in bar.plain

    def test_fill_bar_empty(self):
        bar = _fill_bar(0.0, width=10)
        assert "░" * 10 in bar.plain

    def test_fill_bar_half(self):
        bar = _fill_bar(50.0, width=10)
        assert "█" * 5 in bar.plain

    def test_clamp_viewport_cursor_at_start(self):
        assert _clamp_viewport(0, 0, 10, 20) == 0

    def test_clamp_viewport_cursor_below(self):
        assert _clamp_viewport(5, 3, 10, 20) == 3

    def test_clamp_viewport_cursor_above(self):
        assert _clamp_viewport(0, 12, 10, 20) == 3
