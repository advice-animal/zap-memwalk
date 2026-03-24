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
