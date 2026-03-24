"""Tests for __main__.py entry point (coverage + basic sanity)."""

from __future__ import annotations

import subprocess
import sys

from tests.conftest import frida_mark


@frida_mark
def test_once_prints_summary(idle_proc):
    """--once should print a text table and exit cleanly."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "-m",
            "zap_memwalk",
            "--once",
            str(idle_proc.pid),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "PID" in result.stdout
    assert "Python" in result.stdout
    assert "total:" in result.stdout


@frida_mark
def test_json_output(idle_proc):
    """--json should print valid JSON and exit cleanly."""
    import json

    result = subprocess.run(
        [sys.executable, "-m", "zap_memwalk", "--json", str(idle_proc.pid)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["pid"] == idle_proc.pid
    assert "size_classes" in data


@frida_mark
def test_addr_json_live_float(known_float_proc):
    """--addr-json on a live float block returns the expected JSON fields."""
    import json

    pid, addr = known_float_proc
    result = subprocess.run(
        [sys.executable, "-m", "zap_memwalk", "--addr-json", str(addr), str(pid)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data is not None, "expected a block dict, got null"
    assert data["addr"] == f"0x{addr:x}"
    assert data["state"] == "live"
    assert data["type"] == "float"
    assert (
        data["size_class"] == 32
    )  # PyFloatObject is 24 bytes → rounds to 32-byte block (szidx=1)
    assert len(data["hex"].split()) == 16
    assert data["ob_type_symbol"] is not None
    assert (
        "float" in data["ob_type_symbol"].lower() or "Float" in data["ob_type_symbol"]
    )


@frida_mark
def test_addr_json_not_in_pool():
    """--addr-json on a C-global type object returns an error object with a resolved symbol."""
    import json

    from tests.conftest import _ready_with_line, _spawn, _teardown

    # id(float) is PyFloat_Type — a C global in libpython, never allocated by pymalloc.
    proc = _spawn("import sys; print(id(float), flush=True); sys.stdin.read()")
    addr = int(_ready_with_line(proc))
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "zap_memwalk",
                "--addr-json",
                str(addr),
                str(proc.pid),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        _teardown(proc)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "error" in data
    assert "symbol" in data
    assert data["symbol"] != f"0x{addr:x}", "expected a resolved symbol, got raw hex"
