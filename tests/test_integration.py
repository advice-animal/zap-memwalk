"""Integration tests: real subprocess allocation → MemWalkCollector verification."""

from __future__ import annotations

import pathlib
import sys

import pytest
from tests.conftest import frida_mark
from zap_memwalk._collector import MemWalkCollector
from zap_memwalk._model import POOL_OVERHEAD, BlockState

pytestmark = frida_mark


class TestPrimitiveAddresses:
    @pytest.mark.parametrize(
        "type_name,min_bs,max_bs",
        [
            ("float", 32, 32),
            ("int", 32, 48),
            ("str", 32, 512),
            ("bytes", 32, 512),
            ("list", 64, 512),
            ("dict", 256, 512),
        ],
    )
    def test_object_in_expected_size_class(
        self,
        prim_proc: tuple[int, dict[str, int], dict[str, bool]],
        type_name: str,
        min_bs: int,
        max_bs: int,
    ) -> None:
        pid, addrs, gc_map = prim_proc
        addr = addrs[type_name]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_addr = addr & ~(snap.pool_size - 1)
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, (
            f"{type_name} pool 0x{pool_addr:x} not found in snapshot"
        )
        assert min_bs <= found.block_size <= max_bs, (
            f"{type_name}: block_size {found.block_size} not in [{min_bs}, {max_bs}]"
        )

    def test_float_block_is_live(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        pid, addrs, gc_map = prim_proc
        addr = addrs["float"]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_addr = addr & ~(snap.pool_size - 1)
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, f"float pool 0x{pool_addr:x} not found"
        assert found.block_state(addr) == BlockState.LIVE

    def test_list_pools_in_arena_finds_float(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        pid, addrs, gc_map = prim_proc
        addr = addrs["float"]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
            pool_addr = addr & ~(snap.pool_size - 1)
            arena_addr = addr & ~(snap.arena_size - 1)
            pools = col.list_pools_in_arena(arena_addr)
        assert pool_addr in {int(p["address"], 16) for p in pools}, (
            f"pool 0x{pool_addr:x} not found in list_pools_in_arena(0x{arena_addr:x})"
        )

    def test_addr_json_block_contains_addr(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """The floor-div block arithmetic places addr inside the correct block."""
        pid, addrs, gc_map = prim_proc
        addr = addrs["float"]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
            pool_addr = addr & ~(snap.pool_size - 1)
            found = next(
                (
                    p
                    for sc in snap.size_classes
                    for p in sc.pools
                    if p.address == pool_addr
                ),
                None,
            )
            assert found is not None, f"float pool 0x{pool_addr:x} not found"
            pool_raw = col.read_pool(pool_addr)
        block_size = (pool_raw.get("szidx", found.szidx) + 1) * 16
        off = addr - pool_addr
        assert off >= POOL_OVERHEAD, (
            f"addr offset {off} < POOL_OVERHEAD {POOL_OVERHEAD}"
        )
        block_addr = (
            pool_addr
            + POOL_OVERHEAD
            + ((off - POOL_OVERHEAD) // block_size) * block_size
        )
        assert block_addr <= addr < block_addr + block_size, (
            f"addr 0x{addr:x} not inside block [0x{block_addr:x}, 0x{block_addr + block_size:x})"
        )


GC_HEADER_SIZE = 16  # sizeof(PyGC_Head) = 2 × 8-byte pointers


class TestGCHeader:
    """Validate the relationship between id(x), PyGC_Head, and pymalloc block addresses.

    For non-GC types (float, int, str, bytes):
        block_start == id(x)       ob_refcnt at id(x)+0, ob_type at id(x)+8

    For GC-tracked types (list, dict, tuple, set, user classes):
        block_start == id(x) - 16  PyGC_Head at block_start+0..15
                                   ob_refcnt  at id(x)+0
                                   ob_type    at id(x)+8

    The bug in block_state(id(x)) for GC objects: free_addresses contains
    block_start = id(x)-16, so a freed GC object is misclassified as LIVE.
    """

    @pytest.mark.parametrize("type_name", ["float", "int", "str", "bytes"])
    def test_non_gc_id_is_block_start(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]], type_name: str
    ) -> None:
        """For non-GC types, id(x) == block_start: it lies on a block boundary."""
        pid, addrs, gc_map = prim_proc
        assert not gc_map[type_name], f"{type_name} unexpectedly GC-tracked"
        addr = addrs[type_name]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_addr = addr & ~(snap.pool_size - 1)
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, f"{type_name} pool not found"
        off = addr - pool_addr
        assert (off - POOL_OVERHEAD) % found.block_size == 0, (
            f"{type_name}: id(x)=0x{addr:x} is not on a block boundary "
            f"(off={off}, POOL_OVERHEAD={POOL_OVERHEAD}, block_size={found.block_size})"
        )

    @pytest.mark.parametrize("type_name", ["list", "dict"])
    def test_gc_id_is_block_start_plus_16(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]], type_name: str
    ) -> None:
        """For GC-tracked types, id(x) is 16 bytes past block_start (PyGC_Head)."""
        pid, addrs, gc_map = prim_proc
        assert gc_map[type_name], f"{type_name} unexpectedly non-GC"
        addr = addrs[type_name]
        block_start = addr - GC_HEADER_SIZE
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_addr = addr & ~(snap.pool_size - 1)
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, f"{type_name} pool not found"
        off = block_start - pool_addr
        assert off >= POOL_OVERHEAD, (
            f"{type_name}: block_start=0x{block_start:x} is below POOL_OVERHEAD"
        )
        assert (off - POOL_OVERHEAD) % found.block_size == 0, (
            f"{type_name}: block_start=id(x)-16=0x{block_start:x} is not on a block boundary "
            f"(off={off}, POOL_OVERHEAD={POOL_OVERHEAD}, block_size={found.block_size})"
        )

    @pytest.mark.parametrize("type_name", ["list", "dict"])
    def test_block_state_needs_block_start_not_id_for_gc(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]], type_name: str
    ) -> None:
        """block_state(id(x)) is LIVE; block_state(id(x)-16) is the authoritative answer.

        This test documents the current API: callers must subtract GC_HEADER_SIZE
        for GC-tracked objects before calling block_state.  A freed GC object
        will appear LIVE if the caller passes id(x) instead of id(x)-16, because
        free_addresses contains block_start addresses from the pool freeblock list.
        """
        pid, addrs, gc_map = prim_proc
        assert gc_map[type_name]
        addr = addrs[type_name]  # id(x) — PyObject address
        blk_addr = addr - GC_HEADER_SIZE  # block_start
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_addr = addr & ~(snap.pool_size - 1)
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, f"{type_name} pool not found"
        # Object is live — both addresses should agree it's not FREE
        assert found.block_state(blk_addr) == BlockState.LIVE, (
            f"{type_name}: block_start=0x{blk_addr:x} should be LIVE"
        )
        # Using id(x) directly also returns LIVE for a live object (watermark
        # check passes by coincidence: 16 < block_size always for GC objects)
        assert found.block_state(addr) == BlockState.LIVE, (
            f"{type_name}: id(x)=0x{addr:x} should also appear LIVE while live"
        )

    @pytest.mark.parametrize("type_name", ["list", "dict"])
    def test_pool_mask_correct_for_gc_objects(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]], type_name: str
    ) -> None:
        """id(x) & ~(pool_size-1) gives the correct pool even for GC-tracked objects.

        The 16-byte GC header offset is smaller than pool_size (16 KiB), so the
        pool boundary mask is unaffected.
        """
        pid, addrs, gc_map = prim_proc
        assert gc_map[type_name]
        addr = addrs[type_name]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_from_id = addr & ~(snap.pool_size - 1)
        pool_from_blk_start = (addr - GC_HEADER_SIZE) & ~(snap.pool_size - 1)
        assert pool_from_id == pool_from_blk_start, (
            f"{type_name}: pool mask differs: id gives 0x{pool_from_id:x}, "
            f"block_start gives 0x{pool_from_blk_start:x}"
        )
        found = next(
            (
                p
                for sc in snap.size_classes
                for p in sc.pools
                if p.address == pool_from_id
            ),
            None,
        )
        assert found is not None, f"{type_name} pool 0x{pool_from_id:x} not in snapshot"


class TestArenaDetection:
    def test_collect_finds_arenas(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        assert len(snap.arenas) > 0, "collect() returned no arenas"

    def test_list_arenas_nonempty(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            arenas = col.list_arenas()
        assert len(arenas) > 0, "list_arenas() returned empty list"

    def test_collect_arenas_subset_of_list_arenas(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """Every arena base from collect() must appear in list_arenas()."""
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
            arenas = col.list_arenas()
        list_bases = {int(a["base"], 16) for a in arenas}
        for arena in snap.arenas:
            assert arena.base_address in list_bases, (
                f"arena base 0x{arena.base_address:x} from collect() missing from list_arenas()"
            )

    def test_arena_pool_count_is_63_or_64(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """Each arena has either 63 or 64 pools.

        CPython requests 1 MiB via mmap(NULL, ...) which returns a 4 KiB-aligned
        address. Pool size is 16 KiB (4 pages), so ~75% of arenas land at a
        non-pool-aligned base and waste the first few KiB, yielding 63 pools.
        The remaining ~25% land pool-aligned and yield 64 pools. Both are correct.
        """
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        for arena in snap.arenas:
            assert arena.pool_count in (63, 64), (
                f"arena 0x{arena.base_address:x}: unexpected pool_count {arena.pool_count}"
            )

    def test_arena_alignment_explains_pool_count(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """Pool count follows from base address alignment.

        If arena_base is pool_size-aligned → 64 pools (no wasted prefix).
        Otherwise → 63 pools (first pool starts at ROUNDUP(base, pool_size)).
        """
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_size = snap.pool_size
        for arena in snap.arenas:
            aligned = (arena.base_address % pool_size) == 0
            expected = 64 if aligned else 63
            assert arena.pool_count == expected, (
                f"arena 0x{arena.base_address:x}: base {'IS' if aligned else 'is NOT'} "
                f"pool-aligned, expected {expected} pools, got {arena.pool_count}"
            )

    def test_every_pool_arenaindex_is_consistent(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """Every pool's arenaindex must refer to an arena that contains that pool.

        A phantom pool (e.g. a glibc malloc chunk at a pool-aligned address
        coalesced into a pymalloc VMA) can pass the maxnextoffset invariant
        check but will have an arenaindex that points to an arena whose address
        range does not include the phantom pool.  This test catches that case.
        """
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        arena_size = snap.arena_size
        # Build a map from arena_index → (base, top) for quick lookup
        arena_ranges: dict[int, tuple[int, int]] = {}
        for arena in snap.arenas:
            base = arena.base_address
            arena_ranges[arena.arena_index] = (base, base + arena_size)
        for sc in snap.size_classes:
            for pool in sc.pools:
                if pool.arena_index not in arena_ranges:
                    continue  # arena_index unknown — separate assertion could check this
                base, top = arena_ranges[pool.arena_index]
                assert base <= pool.address < top, (
                    f"pool 0x{pool.address:x} has arenaindex={pool.arena_index} "
                    f"but arena range is [0x{base:x}, 0x{top:x})"
                )

    def test_all_pools_in_arena_agree_on_arenaindex(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """All pools within an ArenaSummary report the same arenaindex.

        ArenaSummary groups pools by arenaindex from the pool header (+32).
        If any pool's arenaindex disagrees with the group key, the grouping
        is wrong — likely caused by reading the wrong offset or a phantom.
        """
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        for arena in snap.arenas:
            for pool in arena.pools:
                assert pool.arena_index == arena.arena_index, (
                    f"pool 0x{pool.address:x} has arenaindex={pool.arena_index} "
                    f"but is grouped under arena {arena.arena_index}"
                )


class TestCoalescedVMA:
    @pytest.mark.skipif(
        sys.platform != "linux", reason="VMA coalescing is Linux-specific"
    )
    def test_scanallpools_finds_pools_across_gap(
        self, coalesced_proc: tuple[int, int, int]
    ) -> None:
        """
        Two pymalloc arenas separated by a non-1MB-multiple anonymous mmap.
        On Linux, consecutive anonymous rw- mmaps coalesce into one VMA entry;
        floor-div listArenas would miss the second arena, but scanAllPools must not.

        Coalescing is kernel behavior (not glibc), stable since 2.6, does not
        occur on macOS. In a controlled subprocess with gc.disable() and no
        background threads, consecutive mmap(NULL,...) calls land adjacently,
        so coalescing is expected and asserted rather than skipped.
        """
        pid, mmap_addr, mmap_size = coalesced_proc

        # Assert that coalescing actually occurred: the VMA containing mmap_addr
        # should be larger than mmap_size alone (it absorbed the adjacent arenas).
        maps = pathlib.Path(f"/proc/{pid}/maps").read_text()
        coalesced = False
        for line in maps.splitlines():
            parts = line.split()
            if not parts:
                continue
            start, end = (int(x, 16) for x in parts[0].split("-"))
            if start <= mmap_addr < end and (end - start) > mmap_size:
                coalesced = True
                break
        assert coalesced, (
            f"expected mmap at 0x{mmap_addr:x} to be coalesced into a larger VMA, "
            f"but /proc/maps shows it as isolated"
        )

        with MemWalkCollector(pid) as col:
            snap = col.collect()

        # Two full arenas of floats: 63–64 pools each, szidx=1 (32-byte blocks).
        # Min is 126 because ~75% of arenas land at non-pool-aligned bases (63 pools).
        float_sc = snap.size_classes[1]
        assert float_sc.pool_count >= 126, (
            f"expected ≥126 float pools (two arenas × 63–64), got {float_sc.pool_count}"
        )


class TestArenaOrdering:
    """Validate that arenaindex (from pool headers) gives pymalloc allocation order,
    and that we can approximate the usable_arenas usage-preference order from
    pool fill levels without reading the (unexported) usable_arenas linked list.

    CPython's usable_arenas is sorted by nfreepools ascending: most-full arenas
    are served first so emptier arenas can drain to zero and be returned to the OS.
    We approximate nfreepools as:
        free_pools  = pools with ref_count == 0   (on the freepools singly-linked list)
        unborn      = ntotalpools - scanned_pool_count
        nfreepools  ≈ free_pools + unborn
    where ntotalpools = 63 or 64 depending on arena_base alignment.
    """

    @staticmethod
    def _ntotalpools(arena_base: int, pool_size: int) -> int:
        """63 if arena_base is not pool-aligned, else 64."""
        return 64 if (arena_base % pool_size) == 0 else 63

    @staticmethod
    def _nfreepools(arena: object, pool_size: int) -> int:  # arena: ArenaSummary
        free = sum(1 for p in arena.pools if p.ref_count == 0)  # type: ignore[union-attr]
        unborn = (
            TestArenaOrdering._ntotalpools(arena.base_address, pool_size)
            - arena.pool_count
        )  # type: ignore[union-attr]
        return free + unborn

    def test_arenaindex_order_differs_from_address_order(
        self, sequential_arenas_proc: tuple[int, list[int]]
    ) -> None:
        """Confirm arenaindex order and address order can diverge.

        In practice mmap(NULL, 1MiB) does not guarantee monotonically increasing
        addresses, so the display should use arenaindex, not address.  This test
        documents the relationship by collecting both orderings; it does not fail
        if they happen to agree (that is legal), but it confirms arenaindex is
        derivable independently of address.
        """
        pid, _ = sequential_arenas_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        arenas = snap.arenas  # already grouped by real arenaindex from pool headers
        # Both orderings must cover the same arenas
        by_arenaindex = sorted(arenas, key=lambda a: a.arena_index)
        by_address = sorted(arenas, key=lambda a: a.base_address)
        assert {a.arena_index for a in by_arenaindex} == {
            a.arena_index for a in by_address
        }
        # Record whether they agree, for documentation purposes
        agree = all(
            a.arena_index == b.arena_index for a, b in zip(by_arenaindex, by_address)
        )
        # Not an assertion — just validate the lists are the same length
        assert len(by_arenaindex) == len(by_address)
        _ = agree  # may be True or False; both are valid

    def test_sequential_fills_produce_increasing_arenaindex(
        self, sequential_arenas_proc: tuple[int, list[int]]
    ) -> None:
        """Floats allocated in batches should land in arenas with increasing arenaindex.

        With gc.disable() and sequential allocation, pymalloc hands out pools
        from the current arena before requesting a new one.  The first float in
        each batch therefore occupies a different arena, and those arenas should
        have strictly increasing arenaindex values (each batch required a fresh
        arena from allarenas[]).
        """
        pid, first_addrs = sequential_arenas_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_size = snap.pool_size
        arena_size = snap.arena_size
        # Map each first_addr to its arena_index
        arena_by_addr: dict[int, int] = {
            a.base_address: a.arena_index for a in snap.arenas
        }
        batch_arenaindices = []
        for addr in first_addrs:
            arena_base = addr & ~(arena_size - 1)
            # arena_base might be off by a pool alignment wasted prefix; try nearby
            idx = arena_by_addr.get(arena_base)
            if idx is None:
                # Try the pool-aligned base within the arena
                pool_addr = addr & ~(pool_size - 1)
                pool = next(
                    (
                        p
                        for sc in snap.size_classes
                        for p in sc.pools
                        if p.address == pool_addr
                    ),
                    None,
                )
                if pool is not None:
                    idx = pool.arena_index
            assert idx is not None, f"could not find arena for first_addr 0x{addr:x}"
            batch_arenaindices.append(idx)
        # Each batch must land in a distinct arena
        assert len(set(batch_arenaindices)) == len(batch_arenaindices), (
            f"batches share arenas: {batch_arenaindices}"
        )
        # Arenaindices must be strictly increasing (each batch needed a new arena)
        for i in range(1, len(batch_arenaindices)):
            assert batch_arenaindices[i] > batch_arenaindices[i - 1], (
                f"arenaindex did not increase between batch {i - 1} and {i}: "
                f"{batch_arenaindices}"
            )

    def test_usable_arenas_recoverable_via_disassembly(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """usable_arenas is locatable by disassembling _PyObject_Malloc.

        The collector exposes list_arenas() which currently uses VMA scanning.
        This test verifies that the arena bases returned by list_arenas() are a
        superset of the arenas reachable by following usable_arenas->nextarena,
        i.e. the disassembly-based approach doesn't surface arenas unknown to
        the VMA scanner.  It is skipped if the collector doesn't yet implement
        disassembly-based discovery (no `list_usable_arenas` method).
        """
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            if not hasattr(col, "list_usable_arenas"):
                pytest.skip("list_usable_arenas not yet implemented")
            usable = col.list_usable_arenas()  # type: ignore[attr-defined]
            all_arenas = col.list_arenas()
        all_bases = {int(a["base"], 16) for a in all_arenas}
        for entry in usable:
            base = int(entry["base"], 16)
            assert base in all_bases, (
                f"usable_arenas entry 0x{base:x} not seen by list_arenas()"
            )

    def test_usable_arenas_nfreepools_ascending(
        self, sequential_arenas_proc: tuple[int, list[int]]
    ) -> None:
        """usable_arenas linked list is sorted by nfreepools ascending.

        CPython maintains this invariant on every alloc/free.  Once we can read
        nfreepools directly from arena_objects via disassembly-located
        usable_arenas, this is a hard invariant, not an approximation.
        Skipped until list_usable_arenas is implemented.
        """
        pid, _ = sequential_arenas_proc
        with MemWalkCollector(pid) as col:
            if not hasattr(col, "list_usable_arenas"):
                pytest.skip("list_usable_arenas not yet implemented")
            usable = col.list_usable_arenas()  # type: ignore[attr-defined]
        if len(usable) < 2:
            pytest.skip("need at least 2 arenas in usable_arenas to check ordering")
        nfree_seq = [int(a["nfreepools"]) for a in usable]
        for i in range(1, len(nfree_seq)):
            assert nfree_seq[i] >= nfree_seq[i - 1], (
                f"usable_arenas not sorted by nfreepools at position {i}: {nfree_seq}"
            )

    def test_nfreepools_approx_decreases_with_arenaindex(
        self, sequential_arenas_proc: tuple[int, list[int]]
    ) -> None:
        """Older arenas (lower arenaindex) should be more full than newer ones.

        CPython's usable_arenas sorts by nfreepools ascending: the most-full
        arenas are served first.  In a process that fills arenas sequentially,
        the oldest arena should have the fewest free pools (most full) and the
        newest should have the most free pools.

        This is a "sketchy" validation: it asserts a weak ordering trend rather
        than a strict sort, because other allocations (interpreter startup,
        imports) may partially fill all arenas.
        """
        pid, first_addrs = sequential_arenas_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_size = snap.pool_size

        # Recover arenaindex for each batch's first float (same logic as above)
        def addr_to_arena(addr: int) -> object:  # returns ArenaSummary | None
            pool_addr = addr & ~(pool_size - 1)
            pool = next(
                (
                    p
                    for sc in snap.size_classes
                    for p in sc.pools
                    if p.address == pool_addr
                ),
                None,
            )
            if pool is None:
                return None
            return next(
                (a for a in snap.arenas if a.arena_index == pool.arena_index), None
            )

        batch_arenas = [addr_to_arena(a) for a in first_addrs]
        assert all(a is not None for a in batch_arenas), (
            "could not locate all batch arenas in snapshot"
        )
        nfree = [self._nfreepools(a, pool_size) for a in batch_arenas]  # type: ignore[arg-type]
        # The oldest arena (first batch, lowest arenaindex) should have fewer or
        # equal free pools than the newest (last batch, highest arenaindex).
        assert nfree[0] <= nfree[-1], (
            f"expected oldest arena to be most full: "
            f"nfreepools by batch = {nfree} (arenaindices: "
            f"{[a.arena_index for a in batch_arenas]})"  # type: ignore[union-attr]
        )


class TestPhantomPools:
    """A small anonymous mmap at a pool-aligned address coalesced with a pymalloc
    arena can produce a phantom "pool" at the mmap's base address.  The mmap
    content (e.g. glibc malloc metadata with fd/bk pointers at offsets +16/+24)
    may partially satisfy the maxnextoffset invariant, inserting a bogus pool
    into the snapshot.  Detecting this requires the arenaindex consistency check.
    """

    @pytest.mark.skipif(
        sys.platform != "linux", reason="VMA coalescing is Linux-specific"
    )
    def test_no_phantom_pool_at_small_mmap_base(
        self, phantom_proc: tuple[int, int] | None
    ) -> None:
        """No phantom pool should appear at the base of a small pool-aligned mmap.

        If the alignment retry failed (phantom_proc is None), skip.
        The arenaindex consistency check must reject the phantom, even if
        maxnextoffset happens to match a valid size class.
        """
        if phantom_proc is None:
            pytest.skip("could not obtain a pool-aligned mmap in 64 tries")
        pid, small_addr = phantom_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        # The small_addr must NOT appear as a pool address in the snapshot
        phantom = next(
            (
                p
                for sc in snap.size_classes
                for p in sc.pools
                if p.address == small_addr
            ),
            None,
        )
        assert phantom is None, (
            f"phantom pool detected at 0x{small_addr:x} (small mmap base): "
            f"szidx={phantom.szidx if phantom else '?'}, "
            f"arenaindex={phantom.arena_index if phantom else '?'}"
        )

    @pytest.mark.skipif(
        sys.platform != "linux", reason="VMA coalescing is Linux-specific"
    )
    def test_all_pools_pass_arenaindex_consistency(
        self, phantom_proc: tuple[int, int] | None
    ) -> None:
        """In the phantom-pool scenario, every pool's arenaindex is self-consistent.

        This is the positive complement of test_no_phantom_pool: if a phantom
        somehow slips through maxnextoffset validation, the arenaindex check
        must still catch it.
        """
        if phantom_proc is None:
            pytest.skip("could not obtain a pool-aligned mmap in 64 tries")
        pid, _ = phantom_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        arena_size = snap.arena_size
        arena_ranges: dict[int, tuple[int, int]] = {
            a.arena_index: (a.base_address, a.base_address + arena_size)
            for a in snap.arenas
        }
        for sc in snap.size_classes:
            for pool in sc.pools:
                if pool.arena_index not in arena_ranges:
                    continue
                base, top = arena_ranges[pool.arena_index]
                assert base <= pool.address < top, (
                    f"pool 0x{pool.address:x} (szidx={pool.szidx}) has "
                    f"arenaindex={pool.arena_index} → arena [0x{base:x}, 0x{top:x}): "
                    f"address is outside the arena — likely a phantom pool"
                )


class TestFreePools:
    """Verify that emptied pools appear in the snapshot and are correctly characterised.

    When all blocks in a pool are freed, CPython puts the pool on the arena's
    freepools singly-linked list by overwriting the pool's first 8 bytes
    (the ref.count union) with the old list head.  maxnextoffset at offset 44
    is NOT touched, so scanAllPools still finds these pools via the invariant
    check.  ref.count appears as 0 (or as the low 32 bits of the old head
    pointer, which is 0 for the terminal entry).  The pool is available for
    reuse by any size class; the arena is not released until every pool is on
    freepools.
    """

    def test_freed_pool_appears_with_zero_ref_count(
        self, freepools_proc: tuple[int, int, list[int]]
    ) -> None:
        """The emptied pool is still visible in the snapshot with ref_count == 0."""
        pid, pool_addr, _ = freepools_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, f"freed pool 0x{pool_addr:x} missing from snapshot"
        assert found.ref_count == 0, (
            f"expected ref_count=0 for freed pool, got {found.ref_count}"
        )

    def test_freed_pool_maxnextoffset_still_valid(
        self, freepools_proc: tuple[int, int, list[int]]
    ) -> None:
        """maxnextoffset survives the freepools overwrite and matches the size class."""
        pid, pool_addr, _ = freepools_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None
        expected_max = snap.pool_size - found.block_size
        assert found.maxnextoffset == expected_max, (
            f"maxnextoffset {found.maxnextoffset} != expected {expected_max} "
            f"(pool_size={snap.pool_size}, block_size={found.block_size})"
        )

    def test_freed_pool_blocks_are_not_live(
        self, freepools_proc: tuple[int, int, list[int]]
    ) -> None:
        """Every block in the freed pool is FREE or UNBORN, never LIVE."""
        pid, pool_addr, block_addrs = freepools_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None
        live_blocks = [
            a for a in block_addrs if found.block_state(a) == BlockState.LIVE
        ]
        assert live_blocks == [], (
            f"{len(live_blocks)} blocks still LIVE in freed pool: "
            + ", ".join(f"0x{a:x}" for a in live_blocks[:5])
        )

    def test_arena_retained_while_other_pools_live(
        self, freepools_proc: tuple[int, int, list[int]]
    ) -> None:
        """The arena containing the freed pool is still present in list_arenas().

        CPython only returns an arena to the OS when every pool in it is on
        freepools.  One freed pool in an otherwise-active arena must not
        cause the arena to disappear.
        """
        pid, pool_addr, _ = freepools_proc
        with MemWalkCollector(pid) as col:
            snap = col.collect()
            arenas = col.list_arenas()
        arena_addr = pool_addr & ~(snap.arena_size - 1)
        list_bases = {int(a["base"], 16) for a in arenas}
        assert arena_addr in list_bases, (
            f"arena 0x{arena_addr:x} missing from list_arenas() even though "
            f"it still has active pools"
        )


class TestStaletypeNames:
    """When ob_type is unmapped (e.g. a freed object whose .so type was unloaded),
    the tool should report what it knows rather than just "unmapped".

    Today's test uses a freed object whose type pointer is still mapped (the
    extension stays loaded) — we verify get_type_name returns a non-empty,
    recognisable name even after the object is freed.  A future test can
    exercise the truly-unmapped case by unloading the extension.
    """

    def test_freed_cext_block_type_name_is_known(
        self,
        stale_type_proc: tuple[int, int, int, str, str],
    ) -> None:
        """get_type_name on a freed C-extension block returns the type name, not empty."""
        pid, obj_addr, ob_type_addr, type_name, module = stale_type_proc
        with MemWalkCollector(pid) as col:
            name = col.get_type_name(obj_addr)
        # The type is still mapped (extension not unloaded), so we expect the
        # real name.  Accept both qualified and unqualified forms.
        assert type_name.split(".")[-1] in name, (
            f"expected '{type_name}' somewhere in get_type_name result, got {name!r}"
        )

    def test_repr_block_freed_object_does_not_crash(
        self,
        stale_type_proc: tuple[int, int, int, str, str],
    ) -> None:
        """repr_block on a freed object address must not raise, even if content is stale."""
        pid, obj_addr, _, _, _ = stale_type_proc
        with MemWalkCollector(pid) as col:
            result = col.repr_block(obj_addr)
        # May return None (object is freed/stale) or a tuple — must not raise.
        assert result is None or isinstance(result, tuple)

    def test_symbolize_unmapped_type_ptr_includes_module(
        self,
        stale_type_proc: tuple[int, int, int, str, str],
    ) -> None:
        """symbolize_addresses on the ob_type pointer names the extension module.

        Even when the address is in a .so that is still loaded, the symbol
        resolution should include the module path.  This is the building block
        for "now-unloaded foo.so:data+0xf00 (TypeName)" formatting.
        """
        pid, _, ob_type_addr, type_name, module = stale_type_proc
        with MemWalkCollector(pid) as col:
            syms = col.symbolize_addresses([ob_type_addr])
        info = syms.get(ob_type_addr)
        # If mapped: should have path pointing to the extension .so
        if info is not None:
            path = info.get("path", "") or info.get("module", "")
            short_module = module.split(".")[0]  # e.g. "_decimal"
            assert short_module in path or short_module in str(info), (
                f"expected module '{short_module}' in symbol info for ob_type, got {info}"
            )


class TestGILFreeRepr:
    """GIL-free repr: read raw memory fields of builtin types without the Python API.

    Each test verifies that repr_block_raw(pid, addr) can reconstruct the
    expected value from the target process's memory using only struct offsets,
    with no ctypes.cast or Python API calls.  Tests are skipped until the
    collector implements repr_block_raw.

    Safety protocol (see REWRITE.md §9):
      1. Read raw bytes at addr.
      2. Validate ob_refcnt > 0 and ob_type is a plausible pointer.
      3. ONLY THEN proceed to type-specific field reads.
    The post-cast ob_type re-read adds no safety — validation must come from
    the raw hexdump data BEFORE any ctypes.cast call.
    """

    def _skip_if_not_implemented(self, col: MemWalkCollector) -> None:
        if not hasattr(col, "repr_block_raw"):
            pytest.skip("repr_block_raw not yet implemented")

    def test_float_raw_repr_matches_value(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """repr_block_raw on a float block returns the correct double value.

        PyFloatObject layout (non-GC, block_start == id(x)):
          +0  ob_refcnt (u64)
          +8  ob_type*  (u64)
          +16 ob_fval   (f64)  ← the value
        """
        pid, addrs, gc_map = prim_proc
        assert not gc_map["float"]
        addr = addrs["float"]
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            result = col.repr_block_raw(addr)  # type: ignore[attr-defined]
        assert result is not None
        # 1.5 is the value in the fixture; accept either the float or its repr
        assert "1.5" in str(result), f"expected 1.5 in repr, got {result!r}"

    def test_str_raw_repr_reads_inline_ascii(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """repr_block_raw on a compact-ASCII str reads inline data at +40.

        PyASCIIObject layout (non-GC, block_start == id(x)):
          +0   ob_refcnt  (u64)
          +8   ob_type*   (u64)
          +16  length     (i64)
          +24  hash       (i64)
          +32  state      (u32)  bit 6 = ascii, bit 5 = compact
          +40  → inline char data
        """
        pid, addrs, gc_map = prim_proc
        assert not gc_map["str"]
        addr = addrs["str"]
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            result = col.repr_block_raw(addr)  # type: ignore[attr-defined]
        assert result is not None
        assert "zap" in str(result), f"expected 'zap' in str repr, got {result!r}"

    def test_bytes_raw_repr_reads_inline_data(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """repr_block_raw on a bytes block reads inline ob_val at +32.

        PyBytesObject layout (non-GC, block_start == id(x)):
          +0   ob_refcnt  (u64)
          +8   ob_type*   (u64)
          +16  ob_size    (i64)
          +24  ob_shash   (i64)
          +32  ob_val[0]  (inline bytes; ob_val[ob_size] == 0x00 guaranteed)
        Bytes beyond ob_val[ob_size] to end of block are uninitialized pool memory.
        """
        pid, addrs, gc_map = prim_proc
        assert not gc_map["bytes"]
        addr = addrs["bytes"]
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            result = col.repr_block_raw(addr)  # type: ignore[attr-defined]
        assert result is not None
        assert "zap" in str(result), f"expected b'zap' in bytes repr, got {result!r}"

    def test_list_raw_repr_reads_ob_size(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """repr_block_raw on a list block reads ob_size from block_start+32.

        PyListObject layout (GC, block_start == id(x) - 16):
          +0   _gc_next  (u64)
          +8   _gc_prev  (u64)
          +16  ob_refcnt (u64)
          +24  ob_type*  (u64)
          +32  ob_size   (i64)  ← list length
          +40  ob_item*  (u64)  ← heap pointer to PyObject*[] (NOT inline)
          +48  allocated (i64)
        """
        pid, addrs, gc_map = prim_proc
        assert gc_map["list"]
        addr = addrs["list"]
        block_start = addr - GC_HEADER_SIZE
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            # Must pass block_start, not id(x), for GC objects
            result = col.repr_block_raw(block_start)  # type: ignore[attr-defined]
        assert result is not None
        # list(range(50)) has 50 elements
        assert "50" in str(result), f"expected 50 in list repr, got {result!r}"

    def test_stale_block_repr_returns_stale_marker(
        self, stale_type_proc: tuple[int, int, int, str, str]
    ) -> None:
        """repr_block_raw on a freed block returns a [freed] or [stale] marker.

        After the object is freed, ob_refcnt at block_start+0 is 0 (or a free-
        list pointer's low 32 bits).  The raw-bytes pre-check (step 3 of the
        safety protocol) must catch this before any ctypes.cast attempt.
        """
        pid, obj_addr, _, _, _ = stale_type_proc
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            result = col.repr_block_raw(obj_addr)  # type: ignore[attr-defined]
        # Must not raise; result must signal that the block is not live
        assert (
            result is None
            or "stale" in str(result).lower()
            or "freed" in str(result).lower()
        ), f"expected stale/freed marker for freed block, got {result!r}"


class TestCStringBlocks:
    """Clarify when raw byte data ends up in pymalloc pools and what null guarantees hold.

    PyBytesObject (small, ≤ ~479 bytes) stores its data inline — the whole
    PyBytesObject block lives in pymalloc.  Raw C char[] buffers from extensions
    using PyMem_Malloc(n ≤ 512) also land in pools but have no header.
    """

    def test_small_bytes_block_is_in_pymalloc(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """PyBytesObject for a small string is in a pymalloc pool.

        'zap' * 100 = 300 bytes, well within the 512-byte pymalloc limit.
        sizeof(PyBytesObject) = 33 bytes → block_size = 512 bytes (szidx=31)
        if 300+33 > 256; or the next power-of-16 above 333.
        """
        pid, addrs, gc_map = prim_proc
        assert not gc_map["bytes"]
        addr = addrs["bytes"]
        with MemWalkCollector(pid) as col:
            snap = col.collect()
        pool_addr = addr & ~(snap.pool_size - 1)
        found = next(
            (p for sc in snap.size_classes for p in sc.pools if p.address == pool_addr),
            None,
        )
        assert found is not None, (
            f"PyBytesObject at 0x{addr:x} not found in any pymalloc pool; "
            f"expected pool at 0x{pool_addr:x}"
        )

    def test_bytes_null_terminator_at_ob_size(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """PyBytesObject guarantees ob_val[ob_size] == 0x00.

        This is the only guaranteed null byte; bytes beyond ob_size to the end
        of the block are uninitialized pool memory.  This test uses ctypes to
        verify the guarantee holds for a live object (with pre-flight raw check).
        """
        import ctypes
        import struct

        pid, addrs, gc_map = prim_proc
        assert not gc_map["bytes"]
        addr = addrs["bytes"]

        # Pre-flight: read raw bytes to validate before ctypes.cast
        header = (ctypes.c_uint8 * 32).from_address(addr)
        raw = bytes(header)
        ob_refcnt, ob_type = struct.unpack_from("<QQ", raw, 0)
        assert ob_refcnt > 0, "bytes ob_refcnt is 0 — cannot validate"
        assert 0x1000 < ob_type < 0xFFFF_FFFF_FFFF, "ob_type is implausible"
        ob_size = struct.unpack_from("<q", raw, 16)[0]
        assert 0 < ob_size < 512, f"ob_size={ob_size} out of expected range"

        # The null terminator is at offset 32 (ob_val start) + ob_size
        null_addr = addr + 32 + ob_size
        null_byte = ctypes.c_uint8.from_address(null_addr).value
        assert null_byte == 0, (
            f"expected null at ob_val[{ob_size}]=0x{null_addr:x}, got 0x{null_byte:02x}"
        )


class TestGILDisplay:
    """GIL state display and pre-acquisition.

    These tests are skipped until the collector implements:
      - col.get_gil_state()  → {"locked": bool, "last_holder_tid": int | None}
      - col.acquire_gil()    → GIL is now held by the agent; returns a token
      - col.release_gil(tok) → releases the GIL
    See REWRITE.md §11 for the disassembly-based location algorithm.
    """

    def _skip_if_not_implemented(self, col: MemWalkCollector) -> None:
        if not hasattr(col, "get_gil_state"):
            pytest.skip("get_gil_state not yet implemented")

    def test_gil_state_is_bool(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """get_gil_state returns a dict with a boolean 'locked' field."""
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            state = col.get_gil_state()  # type: ignore[attr-defined]
        assert isinstance(state, dict), f"expected dict, got {type(state)}"
        assert "locked" in state, f"'locked' key missing from {state}"
        assert isinstance(state["locked"], bool), (
            f"'locked' should be bool, got {type(state['locked'])}"
        )

    def test_acquire_and_release_gil_round_trip(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """acquire_gil/release_gil leave GIL in the same state as before.

        While we hold the GIL, Python threads in the target are frozen.
        After release_gil, the locked field should return to the value it had
        before we intervened (typically free, since the target is blocking on
        stdin in the fixture).
        """
        pid, _addrs, _gc = prim_proc
        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            if not hasattr(col, "acquire_gil"):
                pytest.skip("acquire_gil not yet implemented")
            before = col.get_gil_state()  # type: ignore[attr-defined]
            tok = col.acquire_gil()  # type: ignore[attr-defined]
            during = col.get_gil_state()  # type: ignore[attr-defined]
            col.release_gil(tok)  # type: ignore[attr-defined]
            after = col.get_gil_state()  # type: ignore[attr-defined]
        assert during["locked"] is True, "GIL should appear locked while agent holds it"
        assert after["locked"] == before["locked"], (
            f"GIL locked state changed after round-trip: {before} → {after}"
        )

    def test_incref_increments_refcount(
        self, prim_proc: tuple[int, dict[str, int], dict[str, bool]]
    ) -> None:
        """With the GIL held, col.incref(addr) / col.decref(addr) round-trip correctly.

        We read ob_refcnt before and after via raw memory read (no ctypes.cast).
        Pre-flight check: ob_refcnt > 0 and ob_type is plausible before incref.

        col.incref / col.decref must call Py_IncRef / Py_DecRef (the exported C
        functions), never do raw ob_refcnt ± 1 arithmetic.  CPython 3.12+
        immortal objects have ob_refcnt == _Py_IMMORTAL_REFCNT (0xFFFFFFFF on
        64-bit); Py_IncRef silently skips arithmetic for them, so the refcount
        would not change — skip the assertion in that case.
        """
        import ctypes
        import struct

        _Py_IMMORTAL_REFCNT = 0xFFFF_FFFF  # (Py_ssize_t)UINT_MAX, 64-bit CPython 3.12+

        pid, addrs, gc_map = prim_proc
        addr = addrs["float"]

        with MemWalkCollector(pid) as col:
            self._skip_if_not_implemented(col)
            if not hasattr(col, "acquire_gil") or not hasattr(col, "incref"):
                pytest.skip("acquire_gil / incref not yet implemented")

            # Pre-flight: validate from raw bytes before any API call
            raw = (ctypes.c_uint8 * 16).from_address(addr)
            ob_refcnt_before, ob_type = struct.unpack_from("<QQ", bytes(raw), 0)
            assert ob_refcnt_before > 0, "float ob_refcnt is 0 before incref"
            assert 0x1000 < ob_type < 0xFFFF_FFFF_FFFF, "ob_type implausible"
            if ob_refcnt_before >= _Py_IMMORTAL_REFCNT:
                pytest.skip("float is an immortal object; Py_IncRef is a no-op for it")

            tok = col.acquire_gil()  # type: ignore[attr-defined]
            col.incref(addr)  # type: ignore[attr-defined]
            raw2 = (ctypes.c_uint8 * 8).from_address(addr)
            ob_refcnt_after = struct.unpack_from("<Q", bytes(raw2), 0)[0]
            col.decref(addr)  # type: ignore[attr-defined]
            col.release_gil(tok)  # type: ignore[attr-defined]

        assert ob_refcnt_after == ob_refcnt_before + 1, (
            f"expected ob_refcnt to increase by 1: "
            f"{ob_refcnt_before} → {ob_refcnt_after}"
        )
