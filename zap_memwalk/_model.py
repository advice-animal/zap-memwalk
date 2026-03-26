"""Data model for pymalloc arena/pool/block snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

POOL_OVERHEAD = 48  # sizeof(pool_header) rounded up to 16-byte alignment


class BlockState(Enum):
    LIVE = "live"
    FREE = "free"
    UNBORN = "unborn"  # address >= pool_header.nextoffset (never carved off)


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    address: int  # pool_header address in target process
    arena_index: int  # index into allarenas[]
    szidx: int  # 0-31, block size class
    block_size: int  # (szidx + 1) * 16
    ref_count: int  # pool_header.ref.count (allocated blocks)
    nextoffset: int  # pool_header.nextoffset (byte watermark from pool base)
    maxnextoffset: int  # pool_header.maxnextoffset
    free_addresses: frozenset[int]  # from walking pool_header.freeblock list

    @property
    def pool_size(self) -> int:
        # CPython invariant: maxnextoffset = POOL_SIZE - block_size
        return self.maxnextoffset + self.block_size

    @property
    def total_blocks(self) -> int:
        return (self.pool_size - POOL_OVERHEAD) // self.block_size

    @property
    def free_blocks(self) -> int:
        return self.total_blocks - self.ref_count

    @property
    def fill_pct(self) -> float:
        t = self.total_blocks
        return 100.0 * self.ref_count / t if t > 0 else 0.0

    def block_state(self, block_addr: int) -> BlockState:
        """State of the block at the given absolute address."""
        offset = block_addr - self.address
        if offset >= self.nextoffset:
            return BlockState.UNBORN
        if block_addr in self.free_addresses:
            return BlockState.FREE
        return BlockState.LIVE

    def iter_block_addresses(self) -> list[int]:
        """All block addresses from POOL_OVERHEAD to pool_size (carved and unborn)."""
        addrs = []
        for off in range(POOL_OVERHEAD, self.pool_size, self.block_size):
            addrs.append(self.address + off)
        return addrs


@dataclass
class SizeClassSummary:
    szidx: int
    block_size: int
    pools: list[PoolSnapshot] = field(default_factory=list)

    @property
    def pool_count(self) -> int:
        return len(self.pools)

    @cached_property
    def used_blocks(self) -> int:
        return sum(p.ref_count for p in self.pools)

    @cached_property
    def total_blocks(self) -> int:
        return sum(p.total_blocks for p in self.pools)

    @cached_property
    def fill_pct(self) -> float:
        t = self.total_blocks
        return 100.0 * self.used_blocks / t if t > 0 else 0.0


@dataclass
class ArenaSummary:
    """Pools grouped by their arena_index (pool_header.arenaindex field)."""

    arena_index: int
    base_address: (
        int  # min(pool.address) for this arena — best approximation of arena base
    )
    pools: list[PoolSnapshot] = field(default_factory=list)

    @property
    def pool_count(self) -> int:
        return len(self.pools)

    @cached_property
    def used_blocks(self) -> int:
        return sum(p.ref_count for p in self.pools)

    @cached_property
    def total_blocks(self) -> int:
        return sum(p.total_blocks for p in self.pools)

    @cached_property
    def fill_pct(self) -> float:
        t = self.total_blocks
        return 100.0 * self.used_blocks / t if t > 0 else 0.0


@dataclass
class MemorySnapshot:
    pid: int
    ts: float
    pool_size: int
    arena_size: int
    py_version: tuple[int, int]
    size_classes: list[SizeClassSummary]  # 32 entries, indices 0-31

    @cached_property
    def arenas(self) -> list[ArenaSummary]:
        """Build arena list grouped by pool.arena_index, sorted by arena_index."""
        buckets: dict[int, list[PoolSnapshot]] = {}
        for sc in self.size_classes:
            for pool in sc.pools:
                buckets.setdefault(pool.arena_index, []).append(pool)
        result = []
        for idx in sorted(buckets):
            pools = buckets[idx]
            result.append(
                ArenaSummary(
                    arena_index=idx,
                    base_address=min(p.address for p in pools),
                    pools=pools,
                )
            )
        return result

    @property
    def total_pools(self) -> int:
        return sum(sc.pool_count for sc in self.size_classes)

    @property
    def total_used_blocks(self) -> int:
        return sum(sc.used_blocks for sc in self.size_classes)

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "ts": self.ts,
            "pool_size": self.pool_size,
            "arena_size": self.arena_size,
            "py_version": list(self.py_version),
            "size_classes": [
                {
                    "szidx": sc.szidx,
                    "block_size": sc.block_size,
                    "pool_count": sc.pool_count,
                    "used_blocks": sc.used_blocks,
                    "total_blocks": sc.total_blocks,
                    "fill_pct": round(sc.fill_pct, 1),
                }
                for sc in self.size_classes
                if sc.pool_count > 0
            ],
        }


class BlockAgeTracker:
    """Track first-seen timestamps for live block addresses across snapshots.

    Colors live blocks from yellow (new) through orange to red (old).
    """

    def __init__(self) -> None:
        self._first_seen: dict[int, float] = {}

    def update(self, live_addrs: set[int], now: float) -> None:
        for addr in live_addrs:
            if addr not in self._first_seen:
                self._first_seen[addr] = now
        gone = self._first_seen.keys() - live_addrs
        for addr in gone:
            del self._first_seen[addr]

    def age(self, addr: int, now: float) -> float:
        first = self._first_seen.get(addr)
        return (now - first) if first is not None else 0.0

    def color(self, addr: int, now: float) -> str:
        a = self.age(addr, now)
        if a < 2.0:
            return "yellow"
        elif a < 10.0:
            return "orange1"
        elif a < 60.0:
            return "red"
        else:
            return "bold red"
