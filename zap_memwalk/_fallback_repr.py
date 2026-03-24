"""Block-level repr helpers that work from raw bytes only (no Frida RPC needed).

All functions are pure: they take bytes (and optionally py_version) and return
strings.  No process attachment, no GIL.
"""

from __future__ import annotations

import struct as _s


def fallback_repr_from_raw(
    blk_bytes: bytes,
    type_name: str,
    py_version: tuple[int, int] = (3, 10),
) -> str | None:
    """Best-effort repr from raw block bytes for types with inline data.

    Works even on freed blocks (ob_type at offset 8 is stale but still usable
    as a type hint; the inline payload is intact).

    str       — PyASCIIObject: length at +16, state at +32, ASCII data at +40/+48
    bytes     — PyBytesObject: ob_size at +16, data at +32
    float     — PyFloatObject: ob_fval (double) at +16
    int       — PyLongObject: ob_size/lv_tag at +16, digit[0] at +24
    list/tuple — ob_size (item count) at +16
    dict      — ma_used at +16
    set/frozenset — used at +24
    module    — name resolved via name_hints; returns plain '(freed module)' here
    """
    try:
        if type_name == "str":
            if len(blk_bytes) < 41:
                return None
            length = _s.unpack_from("<q", blk_bytes, 16)[0]
            if not (0 < length < len(blk_bytes) - 36):
                return None
            state = _s.unpack_from("<I", blk_bytes, 32)[0]
            if (state & 0x60) != 0x60:  # must be compact + ascii
                return None
            for data_off in (40, 48):  # 3.12+ then ≤3.11
                end = data_off + length
                if end <= len(blk_bytes):
                    payload = blk_bytes[data_off:end]
                    # Null bytes mean we hit the wstr pointer slot (≤3.11) — skip.
                    if b"\x00" not in payload and all(b < 128 for b in payload):
                        return "(freed) " + repr(
                            payload.decode("ascii", errors="replace")
                        )
            return None
        elif type_name == "bytes":
            if len(blk_bytes) < 33:
                return None
            ob_size = _s.unpack_from("<q", blk_bytes, 16)[0]
            if not (0 <= ob_size <= len(blk_bytes) - 32):
                return None
            return "(freed) " + repr(bytes(blk_bytes[32 : 32 + ob_size]))
        elif type_name == "float":
            if len(blk_bytes) < 24:
                return None
            val = _s.unpack_from("<d", blk_bytes, 16)[0]
            return f"(freed) {val!r}"
        elif type_name == "int":
            if len(blk_bytes) < 28:
                return None
            if py_version >= (3, 12):
                # lv_tag: bits 0-2 = flags; SIGN_ZERO=1, SIGN_NEGATIVE=2
                # compact = lv_tag < 16; digit at +24
                lv_tag = _s.unpack_from("<Q", blk_bytes, 16)[0]
                if lv_tag == 1:  # SIGN_ZERO
                    return "(freed) 0"
                if lv_tag < 16:  # compact single-digit
                    digit = _s.unpack_from("<I", blk_bytes, 24)[0]
                    sign = -1 if (lv_tag & 2) else 1
                    return f"(freed) {sign * digit}"
                ndigits = lv_tag >> 3
                if 0 < ndigits <= 10000:
                    return f"(freed) int ({ndigits}-digit)"
                return None
            else:
                # 3.10-3.11: ob_size is signed; abs = ndigits, sign = sign of value
                ob_size = _s.unpack_from("<q", blk_bytes, 16)[0]
                if ob_size == 0:
                    return "(freed) 0"
                n = abs(ob_size)
                if n <= 10000:
                    if n == 1:
                        digit = _s.unpack_from("<I", blk_bytes, 24)[0]
                        return f"(freed) {-digit if ob_size < 0 else digit}"
                    return f"(freed) int ({n}-digit)"
                return None
        elif type_name in ("list", "tuple"):
            if len(blk_bytes) < 24:
                return None
            ob_size = _s.unpack_from("<q", blk_bytes, 16)[0]
            if 0 <= ob_size <= 10**8:
                return f"(freed) {type_name}[{ob_size}]"
            return None
        elif type_name == "dict":
            if len(blk_bytes) < 24:
                return None
            ma_used = _s.unpack_from("<q", blk_bytes, 16)[0]
            if 0 <= ma_used <= 10**8:
                return f"(freed) dict{{{ma_used}}}"
            return None
        elif type_name in ("set", "frozenset"):
            if len(blk_bytes) < 32:
                return None
            used = _s.unpack_from("<q", blk_bytes, 24)[0]
            if 0 <= used <= 10**8:
                return f"(freed) {type_name}{{{used}}}"
            return None
        elif type_name == "module":
            return "(freed module)"
    except Exception:
        pass
    return None


def cstr_hint(data: bytes) -> str:
    """Return 'cstr' if bytes 0–7 or bytes 8–15 are all printable ASCII.

    Checking the ob_type slot (bytes 8–15) catches freed blocks whose
    free-list pointer has overwritten bytes 0–7 but left string content
    intact at offset 8.  A real ob_type pointer is never printable ASCII.
    """
    if len(data) >= 8 and all(0x20 <= b <= 0x7E for b in data[:8]):
        return "cstr"
    if len(data) >= 16 and all(0x20 <= b <= 0x7E for b in data[8:16]):
        return "cstr"
    return ""


def extract_cstr(data: bytes) -> str | None:
    """Extract null-terminated string from a block identified as cstr.

    Determines start offset (0 or 8) based on which window triggered the
    cstr hint, reads to the first null byte (or end of block).
    """
    if len(data) >= 8 and all(0x20 <= b <= 0x7E for b in data[:8]):
        start = 0
    elif len(data) >= 16 and all(0x20 <= b <= 0x7E for b in data[8:16]):
        start = 8
    else:
        return None
    end = data.find(b"\x00", start)
    if end == -1:
        end = len(data)
    return data[start:end].decode("utf-8", errors="replace")


def hex_bytes(data: bytes, width: int = 16) -> str:
    """Format *data* as two-column hex + ASCII, *width* bytes per row."""
    rows = []
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        rows.append(f"{i:04x}  {hex_part:<{width * 3 - 1}}  {asc_part}")
    return "\n".join(rows)
