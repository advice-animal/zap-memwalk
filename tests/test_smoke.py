"""Frida integration tests — attach to a subprocess and verify the collected data."""

from __future__ import annotations

import pytest
from tests.conftest import frida_mark
from zap_memwalk._collector import MemWalkCollector


class TestCollect:
    @frida_mark
    def test_idle_proc_returns_snapshot(self, idle_proc):
        with MemWalkCollector(idle_proc.pid) as col:
            snap = col.collect()
        assert snap.pid == idle_proc.pid
        assert snap.pool_size in (4096, 16384)
        assert snap.arena_size in (262144, 1048576)
        assert len(snap.size_classes) == 32
        assert snap.total_pools >= 1

    @frida_mark
    def test_szidx_in_range(self, idle_proc):
        with MemWalkCollector(idle_proc.pid) as col:
            snap = col.collect()
        for sc in snap.size_classes:
            assert 0 <= sc.szidx < 32
            for pool in sc.pools:
                assert pool.szidx == sc.szidx
                assert pool.block_size == (sc.szidx + 1) * 16

    @pytest.mark.xfail
    @frida_mark
    def test_alloc_fills_32byte_pools(self, alloc_proc):
        """After allocating 1M ints (~28 bytes each → size-class 1, 32-byte blocks),
        the 32-byte size class should have substantial utilisation."""
        with MemWalkCollector(alloc_proc.pid) as col:
            snap = col.collect()
        sc32 = snap.size_classes[1]  # szidx=1 → 32-byte blocks
        assert sc32.pool_count >= 1, "expected at least one 32-byte pool"
        assert sc32.used_blocks >= 400_000, (
            f"expected ≥400k used blocks in 32-byte class, got {sc32.used_blocks}"
        )

    @frida_mark
    def test_free_addresses_detected(self, alloc_proc):
        """After freeing every other int, some blocks should be in free lists.

        Free lists are fetched lazily via read_pool() rather than collect(),
        so verify via read_pool_snapshot() on the collected pools.
        """
        with MemWalkCollector(alloc_proc.pid) as col:
            snap = col.collect()
            sc32 = snap.size_classes[1]
            assert sc32.pool_count >= 1, "no 32-byte pools found"
            total_free = 0
            for pool in sc32.pools:
                ps = col.read_pool_snapshot(pool.address)
                if ps is not None:
                    total_free += len(ps.free_addresses)
                if total_free > 0:
                    break
        assert total_free > 0, (
            "expected at least some free-listed blocks in 32-byte pools"
        )

    @frida_mark
    def test_multiple_collects(self, idle_proc):
        """Repeated collect() calls should work without errors."""
        with MemWalkCollector(idle_proc.pid) as col:
            s1 = col.collect()
            s2 = col.collect()
        assert s1.pid == s2.pid


class TestReprBlock:
    @frida_mark
    def test_repr_list(self, known_list_proc):
        """repr_block on a live list should return type 'list' and a repr string."""
        pid, addr = known_list_proc
        with MemWalkCollector(pid) as col:
            result = col.repr_block(addr)
        assert result is not None, "repr_block returned None unexpectedly"
        type_name, repr_str = result
        assert type_name == "list", f"expected 'list', got {type_name!r}"
        assert repr_str.startswith("[1, 1, 1"), (
            f"unexpected list repr: {repr_str[:60]!r}"
        )

    @frida_mark
    def test_repr_str(self, known_str_proc):
        """repr_block on a live str should return type 'str' and the expected value."""
        pid, addr = known_str_proc
        with MemWalkCollector(pid) as col:
            result = col.repr_block(addr)
        assert result is not None
        type_name, repr_str = result
        assert type_name == "str", f"expected 'str', got {type_name!r}"
        assert "zap-memwalk test" in repr_str

    @frida_mark
    def test_repr_dict(self, known_dict_proc):
        """repr_block on a live dict should return type 'dict'."""
        pid, addr = known_dict_proc
        with MemWalkCollector(pid) as col:
            result = col.repr_block(addr)
        assert result is not None
        type_name, repr_str = result
        assert type_name == "dict", f"expected 'dict', got {type_name!r}"
        assert "key" in repr_str

    @frida_mark
    def test_repr_implausible_addr_returns_none(self, idle_proc):
        """repr_block on address 0x1 should not crash; returns (None, error_msg)."""
        with MemWalkCollector(idle_proc.pid) as col:
            result = col.repr_block(0x1)
        assert result is not None
        type_name, err_msg = result
        assert type_name is None
        assert err_msg


class TestReadPool:
    @frida_mark
    def test_read_pool_returns_raw_bytes(self, alloc_proc):
        """read_pool on a valid 32-byte pool should return 16384 bytes of raw data."""
        with MemWalkCollector(alloc_proc.pid) as col:
            snap = col.collect()
            sc32 = snap.size_classes[1]
            assert sc32.pools, "no 32-byte pools found"
            pool = sc32.pools[0]
            data = col.read_pool(pool.address)
        assert data["szidx"] == 1
        raw = bytes(data["raw"])
        assert len(raw) == snap.pool_size
