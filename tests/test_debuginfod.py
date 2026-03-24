"""Unit tests for debuginfod helpers in _collector.py — no Frida, no subprocess."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from zap_memwalk._collector import (
    MemWalkCollector,
    _debuginfod_cache_dir,
    _elf_build_id,
    _run_eu_addr2line,
)

# ── _elf_build_id ─────────────────────────────────────────────────────────────


def _write_elf_with_build_id(path: Path, build_id_bytes: bytes) -> None:
    """Write a minimal 64-bit LE ELF with a single PT_NOTE containing the build-id."""
    endian = "<"

    # We'll lay out:
    #   0x00: ELF header (64 bytes)
    #   0x40: PT_NOTE program header (56 bytes)
    #   0x78: Note data

    note_name = b"GNU\x00"  # 4 bytes
    namesz = len(note_name)
    descsz = len(build_id_bytes)
    ntype = 3  # NT_GNU_BUILD_ID

    note = struct.pack(endian + "III", namesz, descsz, ntype)
    note += note_name
    # pad namesz to 4-byte boundary (already aligned)
    note += build_id_bytes
    # pad descsz to 4-byte boundary
    note += b"\x00" * ((4 - descsz % 4) % 4)

    note_offset = 0x78
    note_size = len(note)

    # ELF header (64 bytes)
    e_ident = (
        b"\x7fELF"  # magic
        + b"\x02"  # EI_CLASS: 64-bit
        + b"\x01"  # EI_DATA: little-endian
        + b"\x01"  # EI_VERSION
        + b"\x00"  # EI_OSABI
        + b"\x00" * 8
    )
    # e_type=ET_DYN(3), e_machine=EM_X86_64(62), e_version=1
    elf_hdr = e_ident
    elf_hdr += struct.pack(endian + "HHI", 3, 62, 1)  # type, machine, version
    elf_hdr += struct.pack(endian + "Q", 0)  # e_entry
    elf_hdr += struct.pack(endian + "Q", 0)  # e_phoff placeholder
    elf_hdr += struct.pack(endian + "Q", 0)  # e_shoff
    elf_hdr += struct.pack(endian + "IHH", 0, 64, 56)  # flags, ehsize, phentsize
    e_phoff = 0x40
    elf_hdr += struct.pack(endian + "HH", 1, 0)  # e_phnum=1, e_shentsize
    elf_hdr += struct.pack(endian + "HH", 0, 0)  # e_shnum, e_shstrndx
    # patch e_phoff back in at offset 32
    elf_hdr_b = bytearray(elf_hdr)
    struct.pack_into(endian + "Q", elf_hdr_b, 32, e_phoff)

    # PT_NOTE program header (56 bytes for 64-bit)
    # p_type=4(PT_NOTE), p_flags=4, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align
    ph = struct.pack(
        endian + "IIQQQQQQ",
        4,
        4,
        note_offset,
        0,
        0,
        note_size,
        note_size,
        4,
    )

    data = bytes(elf_hdr_b) + ph + b"\x00" * (note_offset - 0x40 - len(ph)) + note
    path.write_bytes(data)


class TestElfBuildId:
    def test_extracts_build_id(self, tmp_path):
        elf = tmp_path / "lib.so"
        build_id = bytes.fromhex("deadbeef01020304")
        _write_elf_with_build_id(elf, build_id)
        result = _elf_build_id(str(elf))
        assert result == "deadbeef01020304"

    def test_non_elf_returns_none(self, tmp_path):
        f = tmp_path / "notelf"
        f.write_bytes(b"this is not an ELF file")
        assert _elf_build_id(str(f)) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert _elf_build_id(str(tmp_path / "missing.so")) is None

    def test_truncated_elf_returns_none(self, tmp_path):
        f = tmp_path / "short.so"
        f.write_bytes(b"\x7fELF" + b"\x00" * 10)
        assert _elf_build_id(str(f)) is None

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_real_python_binary(self):
        """The running Python interpreter has a build-id on Linux."""
        result = _elf_build_id(sys.executable)
        # Some interpreters are static or lack a build-id, so allow None
        assert result is None or (isinstance(result, str) and len(result) >= 16)


# ── _debuginfod_cache_dir ─────────────────────────────────────────────────────


def test_debuginfod_cache_dir_default():
    with patch.dict(os.environ, {}, clear=True):
        # Without XDG_CACHE_HOME, should use ~/.cache/debuginfod_client
        os.environ.pop("XDG_CACHE_HOME", None)
        d = _debuginfod_cache_dir()
        assert d.name == "debuginfod_client"
        assert "cache" in str(d).lower() or str(d).startswith(str(Path.home()))


def test_debuginfod_cache_dir_xdg(tmp_path):
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        d = _debuginfod_cache_dir()
        assert d == tmp_path / "debuginfod_client"


# ── _run_eu_addr2line ─────────────────────────────────────────────────────────


class TestRunEuAddr2line:
    def test_returns_empty_when_not_found(self):
        with patch("shutil.which", return_value=None):
            assert _run_eu_addr2line([0x1000], "/lib/foo.so") == {}

    def test_parses_symbol_names(self, tmp_path):
        fake_output = "PyFloat_Type\n/usr/lib/libpython3.13.so:42\n"
        fake_eu = tmp_path / "eu-addr2line"
        fake_eu.write_text("#!/bin/sh\necho 'PyFloat_Type'\necho '/lib/x.so:1'\n")
        fake_eu.chmod(0o755)

        with patch("shutil.which", return_value=str(fake_eu)):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout=fake_output, returncode=0)
                result = _run_eu_addr2line([0x1A2B], "/lib/foo.so")
        assert result == {0x1A2B: "PyFloat_Type"}

    def test_skips_unknown_symbol(self, tmp_path):
        fake_output = "??\n??:0\n"
        with patch("shutil.which", return_value="/usr/bin/eu-addr2line"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout=fake_output, returncode=0)
                result = _run_eu_addr2line([0xDEAD], "/lib/foo.so")
        assert result == {}

    def test_handles_subprocess_exception(self):
        with patch("shutil.which", return_value="/usr/bin/eu-addr2line"):
            with patch("subprocess.run", side_effect=OSError("no such file")):
                assert _run_eu_addr2line([0x100], "/lib/foo.so") == {}

    def test_passes_extra_env(self):
        captured: dict = {}

        def fake_run(args, **kwargs):
            captured["env"] = kwargs.get("env")
            return MagicMock(stdout="sym\n/f:1\n", returncode=0)

        with patch("shutil.which", return_value="/usr/bin/eu-addr2line"):
            with patch("subprocess.run", side_effect=fake_run):
                _run_eu_addr2line(
                    [0x10], "/lib/foo.so", extra_env={"DEBUGINFOD_URLS": ""}
                )
        assert captured["env"].get("DEBUGINFOD_URLS") == ""


# ── MemWalkCollector.__init__ validation ─────────────────────────────────────


def test_collector_rejects_invalid_debuginfod():
    with pytest.raises(ValueError, match="debuginfod"):
        MemWalkCollector(pid=1234, debuginfod="yes")


def test_collector_accepts_valid_debuginfod_modes():
    for mode in ("false", "cached", "true"):
        col = MemWalkCollector(pid=1234, debuginfod=mode)
        assert col._debuginfod == mode


# ── _resolve_debug_file ───────────────────────────────────────────────────────


class TestResolveDebugFile:
    def _col(self, mode: str) -> MemWalkCollector:
        return MemWalkCollector(pid=1234, debuginfod=mode)

    def test_true_mode_returns_path_if_exists(self, tmp_path):
        f = tmp_path / "libfoo.so"
        f.write_bytes(b"")
        assert self._col("true")._resolve_debug_file(str(f)) == str(f)

    def test_true_mode_returns_none_if_missing(self, tmp_path):
        assert self._col("true")._resolve_debug_file(str(tmp_path / "nope.so")) is None

    def test_cached_mode_returns_none_when_no_build_id(self, tmp_path):
        f = tmp_path / "notelf.so"
        f.write_bytes(b"not an elf")
        assert self._col("cached")._resolve_debug_file(str(f)) is None

    def test_cached_mode_returns_none_when_cache_missing(self, tmp_path):
        elf = tmp_path / "lib.so"
        _write_elf_with_build_id(elf, bytes.fromhex("aabbccdd"))
        with patch(
            "zap_memwalk._collector._debuginfod_cache_dir",
            return_value=tmp_path / "cache",
        ):
            assert self._col("cached")._resolve_debug_file(str(elf)) is None

    def test_cached_mode_returns_path_when_cache_present(self, tmp_path):
        build_id = "aabbccdd11223344"
        elf = tmp_path / "lib.so"
        _write_elf_with_build_id(elf, bytes.fromhex(build_id))
        cache_dir = tmp_path / "cache" / "debuginfod_client" / build_id
        cache_dir.mkdir(parents=True)
        debug_file = cache_dir / "debuginfo"
        debug_file.write_bytes(b"fake debug")
        with patch(
            "zap_memwalk._collector._debuginfod_cache_dir",
            return_value=tmp_path / "cache" / "debuginfod_client",
        ):
            result = self._col("cached")._resolve_debug_file(str(elf))
        assert result == str(debug_file)
