"""Entry point for ``python -m zap_memwalk`` / ``zap-memwalk``."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="zap-memwalk",
        description="Interactively walk CPython pymalloc arenas in a live process.",
    )
    ap.add_argument("pid", type=int, help="target process PID")
    ap.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        metavar="S",
        help="seconds between auto-refreshes (default: 1.0)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print one snapshot as JSON and exit",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="print a text summary and exit (no TUI)",
    )
    ap.add_argument(
        "--size-json",
        type=int,
        metavar="BYTES",
        help="dump JSON for all blocks in the first pool of the matching size class, then exit",
    )
    ap.add_argument(
        "--addr-json",
        type=lambda s: int(s, 0),
        metavar="ADDR",
        help="dump JSON for the single pymalloc block containing ADDR (hex or decimal), or null",
    )
    ap.add_argument(
        "--debuginfod",
        choices=("false", "cached", "true"),
        default="false",
        metavar="MODE",
        help=(
            "debuginfod symbol enrichment on Linux: "
            "'false' (default) = off; "
            "'cached' = use only already-downloaded debuginfod files; "
            "'true' = fetch from DEBUGINFOD_URLS if needed"
        ),
    )
    args = ap.parse_args()

    from zap_memwalk._collector import MemWalkCollector  # noqa: PLC0415
    from zap_memwalk._tui import MemWalkTUI  # noqa: PLC0415

    try:
        with MemWalkCollector(args.pid, debuginfod=args.debuginfod) as col:
            if args.json:
                snap = col.collect()
                print(json.dumps(snap.to_dict(), indent=2))
                return
            if args.once:
                snap = col.collect()
                _print_text(snap)
                return
            if args.size_json is not None:
                snap = col.collect()
                szidx = (args.size_json + 15) // 16 - 1
                szidx = max(0, min(szidx, 31))
                sc = snap.size_classes[szidx]
                if not sc.pools:
                    print(json.dumps([]))
                    return
                pool = sc.pools[0]
                pool_raw = col.read_pool(pool.address)
                blocks = _pool_blocks_json(col, pool, pool_raw)
                print(json.dumps(blocks, indent=2))
                return
            if args.addr_json is not None:
                snap = col.collect()
                pool_size = snap.pool_size
                pool_addr = args.addr_json & ~(pool_size - 1)
                found_pool = None
                for sc in snap.size_classes:
                    for p in sc.pools:
                        if p.address == pool_addr:
                            found_pool = p
                            break
                    if found_pool:
                        break
                if found_pool is None:
                    sym_label = f"0x{args.addr_json:x}"
                    try:
                        info = col.symbolize_addresses([args.addr_json]).get(
                            args.addr_json
                        )
                        if info is not None:
                            mod, offset, symbol = (
                                info["module"],
                                info["offset"],
                                info.get("symbol"),
                            )
                            sym_label = (
                                f"{mod}!{symbol}"
                                if symbol
                                else (mod if offset == 0 else f"{mod}+0x{offset:x}")
                            )
                    except Exception:
                        pass
                    print(
                        json.dumps(
                            {"error": "not in any pymalloc pool", "symbol": sym_label}
                        )
                    )
                    return
                pool_raw = col.read_pool(found_pool.address)
                block_size = (pool_raw.get("szidx", found_pool.szidx) + 1) * 16
                off = args.addr_json - pool_addr
                from zap_memwalk._model import POOL_OVERHEAD  # noqa: PLC0415

                if off < POOL_OVERHEAD or (off - POOL_OVERHEAD) % block_size != 0:
                    print("null")
                    return
                blocks = _pool_blocks_json(col, found_pool, pool_raw)
                target = f"0x{args.addr_json:x}"
                match = next((b for b in blocks if b["addr"] == target), None)
                print(json.dumps(match, indent=2))
                return
            MemWalkTUI(col, interval=args.interval).run()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


def _print_text(snap: object) -> None:
    from zap_memwalk._model import MemorySnapshot  # noqa: PLC0415

    assert isinstance(snap, MemorySnapshot)
    print(
        f"PID {snap.pid}  Python {snap.py_version[0]}.{snap.py_version[1]}"
        f"  {snap.pool_size // 1024}KiB pools  {snap.arena_size // 1048576}MiB arenas"
    )
    print(
        f"{'szidx':>6} {'bytes':>6} {'pools':>6} {'used':>10} {'total':>10} {'fill%':>7}"
    )
    print("-" * 52)
    for sc in snap.size_classes:
        if sc.pool_count == 0:
            continue
        print(
            f"{sc.szidx:>6} {sc.block_size:>6} {sc.pool_count:>6}"
            f" {sc.used_blocks:>10,} {sc.total_blocks:>10,} {sc.fill_pct:>6.1f}%"
        )
    print(
        f"\ntotal: {snap.total_used_blocks:,} blocks in use across {snap.total_pools} pools"
    )


def _pool_blocks_json(col: object, pool: object, pool_raw: dict) -> list[dict]:
    """Return a list of block-info dicts for every slot in the pool."""
    import struct

    from zap_memwalk._collector import MemWalkCollector  # noqa: PLC0415
    from zap_memwalk._model import POOL_OVERHEAD  # noqa: PLC0415

    assert isinstance(col, MemWalkCollector)

    raw = bytes(pool_raw.get("raw", b"") or b"")
    szidx = pool_raw.get("szidx", pool.szidx)
    block_size = (szidx + 1) * 16
    nextoffset = pool_raw.get("nextoffset", pool.nextoffset)
    free_set = frozenset(int(a, 16) for a in pool_raw.get("freeAddrs", []))

    # First pass: collect blocks and born ob_type pointers for batch RPC.
    slots: list[dict] = []
    ob_type_ptrs: list[tuple[int, int]] = []  # (addr, ob_type_ptr)
    for off in range(POOL_OVERHEAD, len(raw), block_size):
        addr = pool.address + off
        blk = raw[off : off + block_size]
        if off >= nextoffset:
            state = "unborn"
        elif addr in free_set:
            state = "free"
        else:
            state = "live"
        ob_type_ptr = struct.unpack_from("<Q", blk, 8)[0] if len(blk) >= 16 else 0
        if state != "unborn" and ob_type_ptr:
            ob_type_ptrs.append((addr, ob_type_ptr))
        slots.append(
            {
                "addr": f"0x{addr:x}",
                "size_class": block_size,
                "state": state,
                "_blk": blk,
                "_ob_type_ptr": ob_type_ptr,
            }
        )

    # Batch-resolve type names and symbols.
    unique_ptrs = list({p for _, p in ob_type_ptrs})
    try:
        type_names: dict[int, str] = (
            col.resolve_type_names(unique_ptrs) if unique_ptrs else {}
        )
    except Exception:
        type_names = {}
    try:
        sym_results: dict[int, dict | None] = (
            col.symbolize_addresses(unique_ptrs) if unique_ptrs else {}
        )
    except Exception:
        sym_results = {}

    def _fmt_sym(ptr: int, info: dict | None) -> str:
        if info is None:
            return f"<unmapped 0x{ptr:x}>"
        mod = info["module"]
        offset = info["offset"]
        symbol = info.get("symbol")
        if symbol:
            return f"{mod}!{symbol}"
        return mod if offset == 0 else f"{mod}+0x{offset:x}"

    # Second pass: build final dicts.
    result = []
    for slot in slots:
        blk = slot.pop("_blk")
        ob_type_ptr = slot.pop("_ob_type_ptr")
        slot["type"] = type_names.get(ob_type_ptr, "") if ob_type_ptr else ""
        slot["hex"] = " ".join(f"{b:02x}" for b in blk[:16])
        slot["ob_type_symbol"] = (
            _fmt_sym(ob_type_ptr, sym_results.get(ob_type_ptr)) if ob_type_ptr else None
        )
        result.append(slot)
    return result


if __name__ == "__main__":
    main()
