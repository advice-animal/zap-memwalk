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
    args = ap.parse_args()

    from zap_memwalk._collector import MemWalkCollector  # noqa: PLC0415
    from zap_memwalk._tui import MemWalkTUI  # noqa: PLC0415

    try:
        with MemWalkCollector(args.pid) as col:
            if args.json:
                snap = col.collect()
                print(json.dumps(snap.to_dict(), indent=2))
                return
            if args.once:
                snap = col.collect()
                _print_text(snap)
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


if __name__ == "__main__":
    main()
