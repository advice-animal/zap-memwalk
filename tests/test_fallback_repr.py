"""Unit tests for _fallback_repr.py — pure functions, no process needed."""

from __future__ import annotations

import struct

import pytest
from zap_memwalk._fallback_repr import (
    cstr_hint,
    extract_cstr,
    fallback_repr_from_raw,
    hex_bytes,
)

# ── helpers ────────────────────────────────────────────────────────────────────


def _buf(size: int = 64) -> bytearray:
    return bytearray(size)


# ── fallback_repr_from_raw: str ────────────────────────────────────────────────


class TestFallbackStr:
    def _make(self, text: str, data_off: int = 40, extra: int = 0) -> bytes:
        """Build a minimal PyASCIIObject-shaped buffer for *text*."""
        n = len(text)
        buf = _buf(data_off + n + extra)
        struct.pack_into("<q", buf, 16, n)  # length
        struct.pack_into("<I", buf, 32, 0x60)  # state: compact(0x20) | ascii(0x40)
        buf[data_off : data_off + n] = text.encode("ascii")
        return bytes(buf)

    def test_ascii_at_offset_40(self):
        blk = self._make("hello", data_off=40)
        assert fallback_repr_from_raw(blk, "str") == "'hello'"

    def test_ascii_at_offset_48(self):
        """Offset 40 contains null bytes (wstr ptr) so we fall through to offset 48."""
        n = 3
        buf = _buf(64)
        struct.pack_into("<q", buf, 16, n)
        struct.pack_into("<I", buf, 32, 0x60)
        # offset 40–47: leave as zero (null bytes → skip)
        buf[48 : 48 + n] = b"abc"
        assert fallback_repr_from_raw(bytes(buf), "str") == "'abc'"

    def test_too_short_returns_none(self):
        assert fallback_repr_from_raw(b"\x00" * 30, "str") is None

    def test_length_zero_returns_none(self):
        buf = _buf(64)
        struct.pack_into("<q", buf, 16, 0)
        struct.pack_into("<I", buf, 32, 0x60)
        assert fallback_repr_from_raw(bytes(buf), "str") is None

    def test_length_too_large_returns_none(self):
        buf = _buf(64)
        struct.pack_into("<q", buf, 16, 9999)  # way beyond buffer
        struct.pack_into("<I", buf, 32, 0x60)
        assert fallback_repr_from_raw(bytes(buf), "str") is None

    def test_non_compact_ascii_returns_none(self):
        buf = _buf(64)
        struct.pack_into("<q", buf, 16, 3)
        struct.pack_into("<I", buf, 32, 0x20)  # compact but not ascii
        buf[40:43] = b"abc"
        assert fallback_repr_from_raw(bytes(buf), "str") is None

    def test_high_byte_in_payload_returns_none(self):
        buf = _buf(64)
        struct.pack_into("<q", buf, 16, 3)
        struct.pack_into("<I", buf, 32, 0x60)
        buf[40:43] = b"\x80\x81\x82"  # high bytes — not ASCII
        assert fallback_repr_from_raw(bytes(buf), "str") is None


# ── fallback_repr_from_raw: bytes ─────────────────────────────────────────────


class TestFallbackBytes:
    def test_happy_path(self):
        buf = _buf(48)
        struct.pack_into("<q", buf, 16, 5)  # ob_size = 5
        buf[32:37] = b"hello"
        assert fallback_repr_from_raw(bytes(buf), "bytes") == "b'hello'"

    def test_empty_bytes(self):
        buf = _buf(48)
        struct.pack_into("<q", buf, 16, 0)
        assert fallback_repr_from_raw(bytes(buf), "bytes") == "b''"

    def test_too_short_returns_none(self):
        assert fallback_repr_from_raw(b"\x00" * 20, "bytes") is None

    def test_ob_size_negative_returns_none(self):
        buf = _buf(48)
        struct.pack_into("<q", buf, 16, -1)
        assert fallback_repr_from_raw(bytes(buf), "bytes") is None

    def test_ob_size_beyond_buffer_returns_none(self):
        buf = _buf(48)
        struct.pack_into("<q", buf, 16, 9999)
        assert fallback_repr_from_raw(bytes(buf), "bytes") is None


# ── fallback_repr_from_raw: float ─────────────────────────────────────────────


class TestFallbackFloat:
    def test_positive(self):
        buf = _buf(32)
        struct.pack_into("<d", buf, 16, 3.14)
        assert fallback_repr_from_raw(bytes(buf), "float") == "3.14"

    def test_negative(self):
        buf = _buf(32)
        struct.pack_into("<d", buf, 16, -2.5)
        assert fallback_repr_from_raw(bytes(buf), "float") == "-2.5"

    def test_too_short_returns_none(self):
        assert fallback_repr_from_raw(b"\x00" * 10, "float") is None


# ── fallback_repr_from_raw: int (3.12+) ───────────────────────────────────────


class TestFallbackInt312:
    PY = (3, 12)

    def _buf_with_tag(self, lv_tag: int, digit: int = 0) -> bytes:
        buf = _buf(32)
        struct.pack_into("<Q", buf, 16, lv_tag)
        struct.pack_into("<I", buf, 24, digit)
        return bytes(buf)

    def test_zero(self):
        assert fallback_repr_from_raw(self._buf_with_tag(1), "int", self.PY) == "0"

    def test_compact_positive(self):
        assert fallback_repr_from_raw(self._buf_with_tag(0, 42), "int", self.PY) == "42"

    def test_compact_negative(self):
        # lv_tag=2 → sign bit set
        assert fallback_repr_from_raw(self._buf_with_tag(2, 7), "int", self.PY) == "-7"

    def test_multi_digit(self):
        # lv_tag=24 → ndigits = 24>>3 = 3
        assert (
            fallback_repr_from_raw(self._buf_with_tag(24), "int", self.PY)
            == "int (3-digit)"
        )

    def test_ndigits_too_large_returns_none(self):
        # ndigits = 10001 → beyond the 10000 guard
        lv_tag = 10001 << 3
        assert (
            fallback_repr_from_raw(self._buf_with_tag(lv_tag), "int", self.PY) is None
        )

    def test_too_short_returns_none(self):
        assert fallback_repr_from_raw(b"\x00" * 10, "int", self.PY) is None


# ── fallback_repr_from_raw: int (3.10 / 3.11) ────────────────────────────────


class TestFallbackInt310:
    PY = (3, 10)

    def _buf_with_size(self, ob_size: int, digit: int = 0) -> bytes:
        buf = _buf(32)
        struct.pack_into("<q", buf, 16, ob_size)
        struct.pack_into("<I", buf, 24, digit)
        return bytes(buf)

    def test_zero(self):
        assert fallback_repr_from_raw(self._buf_with_size(0), "int", self.PY) == "0"

    def test_single_digit_positive(self):
        assert (
            fallback_repr_from_raw(self._buf_with_size(1, 99), "int", self.PY) == "99"
        )

    def test_single_digit_negative(self):
        assert (
            fallback_repr_from_raw(self._buf_with_size(-1, 99), "int", self.PY) == "-99"
        )

    def test_multi_digit(self):
        assert (
            fallback_repr_from_raw(self._buf_with_size(3), "int", self.PY)
            == "int (3-digit)"
        )

    def test_ndigits_too_large_returns_none(self):
        assert (
            fallback_repr_from_raw(self._buf_with_size(10001), "int", self.PY) is None
        )

    def test_too_short_returns_none(self):
        assert fallback_repr_from_raw(b"\x00" * 10, "int", self.PY) is None


# ── fallback_repr_from_raw: list / tuple ──────────────────────────────────────


class TestFallbackListTuple:
    # list/tuple are GC-tracked: PyGC_Head (16 bytes) sits before the PyObject
    # in the pymalloc block, so ob_size is at block+32 (GC_HEAD + ob_refcnt + ob_type).
    @pytest.mark.parametrize("type_name", ["list", "tuple"])
    def test_happy_path(self, type_name):
        buf = _buf(48)
        struct.pack_into("<q", buf, 32, 7)  # ob_size at GC_HEAD(16) + PyObject(16)
        assert fallback_repr_from_raw(bytes(buf), type_name) == f"{type_name}[7]"

    @pytest.mark.parametrize("type_name", ["list", "tuple"])
    def test_too_short_returns_none(self, type_name):
        assert fallback_repr_from_raw(b"\x00" * 10, type_name) is None

    @pytest.mark.parametrize("type_name", ["list", "tuple"])
    def test_negative_size_returns_none(self, type_name):
        buf = _buf(48)
        struct.pack_into("<q", buf, 32, -1)
        assert fallback_repr_from_raw(bytes(buf), type_name) is None


# ── fallback_repr_from_raw: dict ──────────────────────────────────────────────


class TestFallbackDict:
    # dict is GC-tracked: ma_used is at block+32.
    def test_happy_path(self):
        buf = _buf(48)
        struct.pack_into("<q", buf, 32, 4)  # ma_used at GC_HEAD(16) + PyObject(16)
        assert fallback_repr_from_raw(bytes(buf), "dict") == "dict{4}"

    def test_too_short_returns_none(self):
        assert fallback_repr_from_raw(b"\x00" * 10, "dict") is None

    def test_negative_used_returns_none(self):
        buf = _buf(48)
        struct.pack_into("<q", buf, 32, -1)
        assert fallback_repr_from_raw(bytes(buf), "dict") is None


# ── fallback_repr_from_raw: set / frozenset ───────────────────────────────────


class TestFallbackSet:
    # set/frozenset are GC-tracked: 'used' is at block+40 (fill at +32, used at +40).
    @pytest.mark.parametrize("type_name", ["set", "frozenset"])
    def test_happy_path(self, type_name):
        buf = _buf(64)
        struct.pack_into(
            "<q", buf, 40, 3
        )  # used at GC_HEAD(16) + PyObject(16) + fill(8)
        assert fallback_repr_from_raw(bytes(buf), type_name) == f"{type_name}{{3}}"

    @pytest.mark.parametrize("type_name", ["set", "frozenset"])
    def test_too_short_returns_none(self, type_name):
        assert fallback_repr_from_raw(b"\x00" * 10, type_name) is None

    @pytest.mark.parametrize("type_name", ["set", "frozenset"])
    def test_negative_used_returns_none(self, type_name):
        buf = _buf(64)
        struct.pack_into("<q", buf, 40, -1)
        assert fallback_repr_from_raw(bytes(buf), type_name) is None


# ── fallback_repr_from_raw: frame ─────────────────────────────────────────────


def test_frame_returns_fixed_string():
    assert fallback_repr_from_raw(b"\x00" * 64, "frame") == "(frame)"


# ── fallback_repr_from_raw: module + unknown ──────────────────────────────────


def test_module_returns_fixed_string():
    assert fallback_repr_from_raw(b"\x00" * 64, "module") == "(module)"


def test_unknown_type_returns_none():
    assert fallback_repr_from_raw(b"\x00" * 64, "nonexistent") is None


# ── cstr_hint ─────────────────────────────────────────────────────────────────


class TestCstrHint:
    def test_printable_in_first_8(self):
        assert cstr_hint(b"Hello!! ") == "cstr"

    def test_printable_in_bytes_8_to_15(self):
        data = b"\x00" * 8 + b"World!!! "
        assert cstr_hint(data) == "cstr"

    def test_non_printable_returns_empty(self):
        assert cstr_hint(b"\x00" * 16) == ""

    def test_too_short_returns_empty(self):
        assert cstr_hint(b"Hi") == ""  # < 8 bytes

    def test_control_char_in_first_8_falls_through_to_second_window(self):
        # first 8 bytes contain a non-printable; second window is printable
        data = b"\x01" * 8 + b"abcdefgh"
        assert cstr_hint(data) == "cstr"


# ── extract_cstr ──────────────────────────────────────────────────────────────


class TestExtractCstr:
    def test_start_at_0_with_null(self):
        # All 8 bytes printable → start=0; null terminates after the window.
        data = b"hello   " + b"\x00more"
        assert extract_cstr(data) == "hello   "  # reads to null at index 8

    def test_start_at_8_with_null(self):
        # First 8 bytes non-printable; bytes 8-15 all printable → start=8.
        # Null terminator sits after the 8-byte window.
        data = b"\x01" * 8 + b"world!!!" + b"\x00xx"
        assert extract_cstr(data) == "world!!!"  # reads to null at index 16

    def test_no_null_reads_to_end(self):
        data = b"12345678"
        assert extract_cstr(data) == "12345678"

    def test_neither_window_printable_returns_none(self):
        assert extract_cstr(b"\x00" * 16) is None

    def test_empty_string_at_offset_0(self):
        data = b" " * 8 + b"\x00more"  # first 8 all printable, then null
        # start=0, find null: no null in first 8, then at index 8
        result = extract_cstr(data)
        assert result == "        "  # 8 spaces


# ── hex_bytes ─────────────────────────────────────────────────────────────────


class TestHexBytes:
    def test_single_row(self):
        out = hex_bytes(b"\x41\x42\x43")  # "ABC"
        assert "41 42 43" in out
        assert "ABC" in out
        assert out.startswith("0000")

    def test_non_printable_shown_as_dot(self):
        out = hex_bytes(b"\x00\x01\x02")
        assert "..." in out

    def test_multi_row(self):
        out = hex_bytes(bytes(range(32)))
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[1].startswith("0010")

    def test_empty(self):
        assert hex_bytes(b"") == ""
