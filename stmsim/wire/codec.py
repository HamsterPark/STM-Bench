"""Byte-level codec for the controller TCP protocol, server side.

The *client* side of this protocol is the controller client's send/response pair
(v1.0.9) with the compatibility patch applied. This module is its mirror image:

* :func:`decode_args` reads a request body the way ``Nanonis.send`` writes it
  (``handleString`` / ``handleArrayString`` / ``handleArrayPrepend`` / ``handleArray`` /
  ``handle2DArray`` / scalar ``struct.pack``);
* :func:`encode_returns` writes a reply body the way the patched ``parseGeneralResponse``
  reads it, and appends the error section.

Format characters follow the checked-in wire-format reference::

    scalars   H h I i f d            big-endian struct
    2X        rows, cols (two preceding int fields) then rows*cols raw X
    *X        raw X elements, count = previous int field      (reply: 8 B for d, else 4 B)
    -*X       same as *X (explicitly "no length prepended")
    +*X       self-contained: int32 count + elements
    **X       raw X elements, count = FIRST field of the reply (Variables[0])
    +*c       self-contained string: int32 len + bytes
    *-c       raw string bytes, len = previous int field
    *+c       string array: each int32 len + bytes; the two preceding int fields are
              (total bytes, count) — the client advances by the BYTES field
    **c       string array, count = first field
    *2c       rows × cols strings, rows/cols = two preceding int fields

Request-side quirk: for a ``c`` argument the client picks the encoding by the Python
type it was given — ``str`` → ``int32 len + bytes``, ``list`` → ``int32 4*n, int32 n,
then n × (int32 len + bytes)``. The format string does not say which, so
:func:`decode_args` accepts an explicit ``array_string_args`` set and otherwise uses a
plausibility test.

Wire frame::

    request : name.ljust(32, '\0') | uint32 body_size | uint16 send_response_back | 2 zero | body
    reply   : name (32 B, **verbatim echo**)          | uint32 body_size | 4 zero          | body
"""
from __future__ import annotations

import struct
from typing import Any, Sequence

HEADER_LEN = 40
NAME_LEN = 32
ERROR_HEADER_LEN = 8

_SCALAR = {"H": 2, "h": 2, "I": 4, "i": 4, "f": 4, "d": 8, "b": 1, "B": 1}
_INT_FMTS = {"H", "h", "I", "i", "b", "B"}


def _pack(fmt: str, v: Any) -> bytes:
    if fmt in _INT_FMTS:
        v = int(v)
    else:
        v = float(v)
    return struct.pack(">" + fmt, v)


def _elem_size_reply(fmt: str, kind: str) -> int:
    """Byte stride the *client* uses when reading arrays of ``fmt``.

    ``*X``/``-*X`` go through ``decodeArrayPrepended`` (8 for d, else 4);
    ``**X`` through the patched ``decodeArray`` (calcsize); ``+*X``/``*+X`` use calcsize.
    """
    if kind == "star":
        return 8 if fmt == "d" else 4
    return struct.calcsize(">" + fmt)


def _s(v: Any) -> bytes:
    if isinstance(v, bytes):
        return v
    return str(v if v is not None else "").encode("utf-8")


# ────────────────────────────── request decoding ──────────────────────────────


class _Reader:
    def __init__(self, body: bytes):
        self.b = body
        self.pos = 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.b):
            raise ValueError(f"request body too short: need {n} at {self.pos}, have {len(self.b)}")
        out = self.b[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt: str):
        return struct.unpack(">" + fmt, self.take(_SCALAR[fmt]))[0]

    def remaining(self) -> int:
        return len(self.b) - self.pos


def _looks_like_string_array(r: _Reader) -> bool:
    """Peek: ``int32 4n, int32 n, then n strings`` that fit inside the body?"""
    if r.remaining() < 8:
        return False
    nbytes, n = struct.unpack(">ii", r.b[r.pos:r.pos + 8])
    if n < 0 or nbytes != 4 * n or n > 10_000:
        return False
    pos = r.pos + 8
    for _ in range(n):
        if pos + 4 > len(r.b):
            return False
        ln = struct.unpack(">i", r.b[pos:pos + 4])[0]
        if ln < 0 or pos + 4 + ln > len(r.b):
            return False
        pos += 4 + ln
    return True


def decode_args(body: bytes, arg_fmts: Sequence[str], *,
                array_string_args: Sequence[int] = ()) -> list:
    """Decode a request body into Python values, one per format entry.

    ``array_string_args`` — indexes (into ``arg_fmts``) of ``c`` fields that are string
    *arrays* (e.g. ``Scan.PropsSet`` modules names). Others are decided by plausibility.
    """
    r = _Reader(body)
    vals: list = []
    for idx, fmt in enumerate(arg_fmts):
        if fmt in _SCALAR:
            vals.append(r.scalar(fmt))
        elif fmt[0] == "2":
            rows = r.scalar("i")
            cols = r.scalar("i")
            et = fmt[1]
            n = rows * cols
            data = [struct.unpack(">" + et, r.take(_SCALAR[et]))[0] for _ in range(n)]
            vals.append([data[i * cols:(i + 1) * cols] for i in range(rows)])
        elif "c" in fmt:
            if idx in array_string_args or _looks_like_string_array(r):
                r.scalar("i")  # 4*n
                n = r.scalar("i")
                out = []
                for _ in range(n):
                    ln = r.scalar("i")
                    out.append(r.take(ln).decode("utf-8", errors="replace"))
                vals.append(out)
            else:
                ln = r.scalar("i")
                vals.append(r.take(ln).decode("utf-8", errors="replace"))
        elif fmt.startswith("+*"):
            et = fmt[2]
            n = r.scalar("i")
            vals.append([struct.unpack(">" + et, r.take(_SCALAR[et]))[0] for _ in range(n)])
        elif fmt.startswith("-*") or fmt.startswith("*"):
            et = fmt[2] if fmt.startswith("-*") else fmt[1]
            # no length on the wire: count comes from the previous int argument
            prev_ints = [v for v in vals if isinstance(v, int) and not isinstance(v, bool)]
            size = _SCALAR[et]
            if prev_ints and prev_ints[-1] * size <= r.remaining():
                n = prev_ints[-1]
            else:
                n = r.remaining() // size
            vals.append([struct.unpack(">" + et, r.take(size))[0] for _ in range(n)])
        else:
            raise ValueError(f"unsupported argument format {fmt!r}")
    return vals


# ────────────────────────────── reply encoding ──────────────────────────────


def fill_counts(values: Sequence[Any], ret_fmts: Sequence[str]) -> list:
    """Replace ``None`` in count/size slots with numbers derived from the arrays.

    Handlers may pass ``None`` for the int fields that only exist to size a following
    array/string; this fills them so module code never hand-computes byte totals.
    """
    vals = list(values)
    n = len(ret_fmts)
    for i, fmt in enumerate(ret_fmts):
        if fmt in _SCALAR:
            continue
        if fmt in ("*+c",):
            strs = [_s(x) for x in (vals[i] or [])]
            if i >= 1 and vals[i - 1] is None:
                vals[i - 1] = len(strs)
            if i >= 2 and vals[i - 2] is None:
                vals[i - 2] = sum(4 + len(x) for x in strs)
        elif fmt == "*-c":
            if i >= 1 and vals[i - 1] is None:
                vals[i - 1] = len(_s(vals[i]))
        elif fmt[0] == "2" or fmt == "*2c":
            arr = vals[i]
            rows = len(arr) if arr is not None else 0
            cols = len(arr[0]) if rows else 0
            if i >= 1 and vals[i - 1] is None:
                vals[i - 1] = cols
            if i >= 2 and vals[i - 2] is None:
                vals[i - 2] = rows
        elif fmt.startswith("**"):
            if vals[0] is None:
                vals[0] = len(vals[i] or [])
        elif fmt.startswith("+*"):
            pass  # self-contained
        elif fmt.startswith("*+") or fmt.startswith("-*") or fmt.startswith("*"):
            if i >= 1 and vals[i - 1] is None:
                vals[i - 1] = len(vals[i] or [])
    for i, fmt in enumerate(ret_fmts):
        if fmt in _SCALAR and vals[i] is None:
            vals[i] = 0
    return vals


def encode_returns(values: Sequence[Any], ret_fmts: Sequence[str]) -> bytes:
    """Encode declared return fields (no error section)."""
    vals = fill_counts(values, ret_fmts)
    if len(vals) != len(ret_fmts):
        raise ValueError(f"{len(vals)} values for {len(ret_fmts)} return formats")
    out = bytearray()
    for i, fmt in enumerate(ret_fmts):
        v = vals[i]
        if fmt in _SCALAR:
            out += _pack(fmt, v)
        elif fmt[0] == "2":
            et = fmt[1]
            for row in (v or []):
                for x in row:
                    out += _pack(et, x)
        elif fmt == "*+c" or fmt == "**c":
            for x in (v or []):
                b = _s(x)
                out += struct.pack(">i", len(b)) + b
        elif fmt == "*-c":
            out += _s(v)
        elif fmt == "*2c":
            for row in (v or []):
                for x in row:
                    b = _s(x)
                    out += struct.pack(">i", len(b)) + b
        elif fmt == "+*c":
            b = _s(v)
            out += struct.pack(">i", len(b)) + b
        elif fmt.startswith("+*"):
            et = fmt[2]
            arr = list(v or [])
            out += struct.pack(">i", len(arr))
            for x in arr:
                out += _pack(et, x)
        elif fmt.startswith("**"):
            et = fmt[2]
            for x in (v or []):
                out += _pack(et, x)
        elif fmt.startswith("*+"):
            et = fmt[2]
            for x in (v or []):
                out += _pack(et, x)
        elif fmt.startswith("-*"):
            et = fmt[2]
            for x in (v or []):
                out += _pack(et, x)
        elif fmt.startswith("*"):
            et = fmt[1]
            stride = _elem_size_reply(et, "star")
            for x in (v or []):
                b = _pack(et, x)
                out += b.rjust(stride, b"\x00") if len(b) < stride else b
        else:
            raise ValueError(f"unsupported return format {fmt!r}")
    return bytes(out)


def error_section(status: int = 0, description: str = "") -> bytes:
    d = description.encode("utf-8") if description else b""
    return struct.pack(">ii", int(status), len(d)) + d


def encode_reply_body(values: Sequence[Any], ret_fmts: Sequence[str]) -> bytes:
    return encode_returns(values, ret_fmts) + error_section(0, "")


def encode_error_body(status: int, description: str) -> bytes:
    """Error-only body: the length identity ``len == 8 + desc_len`` must hold."""
    if not description:
        description = f"controller error status {status}"
    return error_section(status, description)


# ────────────────────────────── framing ──────────────────────────────


def parse_request_header(hdr: bytes) -> tuple[bytes, str, int, bool]:
    """→ (raw 32-byte name field, command string, body_size, send_response_back)."""
    if len(hdr) != HEADER_LEN:
        raise ValueError(f"header must be {HEADER_LEN} bytes, got {len(hdr)}")
    raw_name = hdr[:NAME_LEN]
    name = raw_name.rstrip(b"\x00").decode("utf-8", errors="replace")
    body_size = struct.unpack(">I", hdr[32:36])[0]
    send_back = struct.unpack(">H", hdr[36:38])[0] != 0
    return raw_name, name, body_size, send_back


def build_reply_frame(raw_name: bytes, body: bytes) -> bytes:
    """Header echoes the request's 32-byte name field verbatim (the client compares)."""
    if len(raw_name) != NAME_LEN:
        raw_name = raw_name[:NAME_LEN].ljust(NAME_LEN, b"\x00")
    return raw_name + struct.pack(">I", len(body)) + b"\x00\x00\x00\x00" + body


def build_request_frame(command: str, body: bytes, send_response_back: bool = True) -> bytes:
    """Client-side framing (used by tests and the in-process object client)."""
    return (command.ljust(NAME_LEN, "\x00").encode("utf-8")
            + struct.pack(">I", len(body))
            + struct.pack(">H", 1 if send_response_back else 0)
            + b"\x00\x00" + body)
