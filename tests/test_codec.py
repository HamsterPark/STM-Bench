"""Wire codec round-trips, with MAST's patched ``nanonis_spm`` client as the oracle.

Two directions:

* reply: we encode with :mod:`stmsim.wire.codec`, the *real* patched
  ``Nanonis.parseGeneralResponse`` decodes — values must come back equal;
* request: the real ``Nanonis.send`` frames a request into a fake transport, we decode
  the body — arguments must come back equal.

Also: the full loopback server, with the real client end-to-end over TCP.
"""
from __future__ import annotations

import socket
import struct

import numpy as np
import pytest

from stmsim.wire import codec
from stmsim.wire.errors import ModuleNotRunning
from stmsim.wire.server import Dispatcher, WireServer
from stmsim.wire.spec import commands

from tests.conftest import requires_mast


class _ReplyTransport:
    """Fake socket: swallows the request, hands back one prepared reply frame."""

    def __init__(self, reply: bytes):
        self._reply = reply
        self.sent = b""
        self.pos = 0

    def sendall(self, data):
        self.sent += bytes(data)

    def send(self, data):
        self.sendall(data)

    def recv(self, n):
        chunk = self._reply[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk

    def gettimeout(self):
        return 5.0

    def settimeout(self, t):
        pass


def _client():
    import mast.core.nanonis_patch  # noqa: F401 — applies the patch on import
    from nanonis_spm import Nanonis as Client   # MAST's client class
    return Client


def _oracle_decode(command: str, values, ret_fmts):
    """Encode with our codec; decode with the real patched client. Returns Variables."""
    Client = _client()
    body = codec.encode_reply_body(values, ret_fmts)
    frame = codec.build_reply_frame(command.ljust(32, "\x00").encode(), body)
    nn = Client(_ReplyTransport(frame))
    err, _raw, variables = nn.quickSend(command, [], [], list(ret_fmts))
    assert err == "", err
    return variables


def _oracle_encode_request(command: str, args, arg_fmts) -> bytes:
    """Frame a request with the real client; return the body bytes it put on the wire."""
    Client = _client()
    ok_reply = codec.build_reply_frame(command.ljust(32, "\x00").encode(),
                                       codec.encode_reply_body([], []))
    tr = _ReplyTransport(ok_reply)
    nn = Client(tr)
    nn.quickSend(command, list(args), list(arg_fmts), [])
    raw_name, name, size, _ = codec.parse_request_header(tr.sent[:40])
    assert name == command
    return tr.sent[40:40 + size]


# ────────────────────────────── reply direction ──────────────────────────────

@requires_mast
@pytest.mark.parametrize("command,values", [
    ("Bias.Get", [1.25]),
    ("FolMe.XYPosGet", [1.5e-7, -2.5e-7]),
    ("Scan.FrameGet", [1e-7, 2e-7, 5e-8, 5e-8, 12.5]),
    ("ZCtrl.StatusGet", [2]),
])
def test_scalar_replies_decode_with_real_client(command, values):
    spec = commands()[command]
    got = _oracle_decode(command, values, spec.ret_fmts)
    for fmt, want, have in zip(spec.ret_fmts, values, got):
        if fmt in ("f",):
            assert have == pytest.approx(want, rel=1e-6)
        elif fmt == "d":
            assert have == pytest.approx(want, rel=1e-12)
        else:
            assert have == want


@requires_mast
def test_scan_buffer_get_star_i_array_unpacked():
    # returns: i (count), *i (channels), i pixels, i lines
    got = _oracle_decode("Scan.BufferGet", [None, [0, 14, 30], 256, 256],
                         commands()["Scan.BufferGet"].ret_fmts)
    assert got[0] == 3 and got[1] == [0, 14, 30] and got[2:] == [256, 256]


@requires_mast
def test_signals_names_get_string_array():
    names = ["Current (A)", "Bias (V)", "Z (m)", "Amplitude (m)"]
    got = _oracle_decode("Signals.NamesGet", [None, None, names],
                         commands()["Signals.NamesGet"].ret_fmts)
    assert got[2] == names and got[1] == len(names)
    assert got[0] == sum(4 + len(n) for n in names)


@requires_mast
def test_util_session_path_get_singular_string():
    path = "E:/stm_bench/run_1/session"
    got = _oracle_decode("Util.SessionPathGet", [None, path], commands()["Util.SessionPathGet"].ret_fmts)
    assert got[1] == path and got[0] == len(path)


@requires_mast
def test_frame_data_grab_2d_float_array():
    spec = commands()["Scan.FrameDataGrab"]  # i, *-c, i, i, 2f, I
    rows, cols = 4, 6
    frame = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) * 1e-12
    got = _oracle_decode("Scan.FrameDataGrab", [None, "Z (m)", None, None, frame.tolist(), 1], spec.ret_fmts)
    assert got[1] == "Z (m)" and got[2] == rows and got[3] == cols
    arr = np.asarray(got[4], dtype=np.float32).reshape(rows, cols)
    assert np.allclose(arr, frame)
    assert got[5] == 1


@requires_mast
def test_osci2t_data_get_double_arrays():
    spec = commands()["Osci2T.DataGet"]  # d, d, i, *d, i, *d
    a = np.linspace(-1e-12, 1e-12, 256).tolist()
    b = np.linspace(0, 1e-9, 256).tolist()
    got = _oracle_decode("Osci2T.DataGet", [0.0, 0.0005, None, a, None, b], spec.ret_fmts)
    assert got[1] == pytest.approx(0.0005)
    assert got[2] == 256 and got[4] == 256
    assert np.allclose(got[3], a) and np.allclose(got[5], b)


@requires_mast
def test_osci_timebase_get_i_i_starf():
    spec = commands()["Osci1T.TimebaseGet"]  # i, i, *f
    tb = [6.4, 2.56, 1.28, 0.64, 0.256, 0.128]
    got = _oracle_decode("Osci1T.TimebaseGet", [5, None, tb], spec.ret_fmts)
    assert got[0] == 5 and got[1] == 6 and np.allclose(got[2], tb, rtol=1e-6)


@requires_mast
def test_marks_points_get_double_star_families():
    spec = commands()["Marks.PointsGet"]  # i, **f, **f, i, **c, **I, **I
    xs, ys = [1e-9, 2e-9], [3e-9, 4e-9]
    got = _oracle_decode("Marks.PointsGet", [None, xs, ys, 7, ["a", "bb"], [1, 2], [3, 4]], spec.ret_fmts)
    assert got[0] == 2 and np.allclose(got[1], xs) and np.allclose(got[2], ys)
    assert got[3] == 7 and got[4] == ["a", "bb"] and got[5] == [1, 2] and got[6] == [3, 4]


@requires_mast
def test_scan_props_get_with_mast_2d_strings():
    spec = commands()["Scan.PropsGet"]
    modules = ["Bias", "Z-Controller", "Scan"]
    counts = [2, 3, 1]
    params = [["Bias (V)", "Calibration", ""], ["Setpoint", "P", "I"], ["Speed", "", ""]]
    vals = [1, 0, 1, None, "series", None, "comment", None, None, modules, None, counts, None, None, params, 0]
    got = _oracle_decode("Scan.PropsGet", vals, spec.ret_fmts)
    assert got[0:3] == [1, 0, 1]
    assert got[4] == "series" and got[6] == "comment"
    assert got[9] == modules and got[11] == counts
    assert got[12] == 3 and got[13] == 3
    assert [list(r) for r in np.asarray(got[14]).reshape(3, 3)] == params
    assert got[15] == 0


@requires_mast
def test_error_only_body_reaches_client_as_text():
    Client = _client()
    body = codec.encode_error_body(1, "NeedModule: Cannot access the 'OsciHR' module. Please make sure it is running.")
    assert len(body) == 8 + (len(body) - 8)
    status, ln = struct.unpack(">ii", body[:8])
    assert status == 1 and ln == len(body) - 8
    # (a) a verb with no declared returns: text comes through verbatim
    frame = codec.build_reply_frame("Osci2T.Run".ljust(32, "\x00").encode(), body)
    err, _raw, vars_ = Client(_ReplyTransport(frame)).quickSend("Osci2T.Run", [], [], [])
    assert "NeedModule" in err and vars_ == []
    # (b) declared fields overrun the error-only body -> MAST's salvage path (2026-08-09 shape)
    frame = codec.build_reply_frame("Osci2T.TimebaseGet".ljust(32, "\x00").encode(), body)
    err, _raw, vars_ = Client(_ReplyTransport(frame)).quickSend("Osci2T.TimebaseGet", [], [], ["i", "i", "*f"])
    assert "NeedModule" in err and vars_ == []
    # (c) a single int32 return fits inside the error body: the patched client reports a
    #     layout mismatch instead of the text. That is MAST-against-real-Client behaviour
    #     (OsciHRProbe only checks "any error"), so the simulator stays faithful.
    frame = codec.build_reply_frame("OsciHR.SamplesGet".ljust(32, "\x00").encode(), body)
    err, _raw, vars_ = Client(_ReplyTransport(frame)).quickSend("OsciHR.SamplesGet", [], [], ["i"])
    assert err != ""


# ────────────────────────────── request direction ──────────────────────────────

@requires_mast
@pytest.mark.parametrize("command,args", [
    ("Bias.Set", [0.02]),
    ("ZCtrl.Withdraw", [1, -1]),
    ("Motor.StartMove", [4, 10, 0, 1]),
    ("TipShaper.PropsSet", [0.05, 2, 0.02, -5e-10, 0.1, 0.02, 0.5, 5e-10, 0.1, 0.5, 1]),
    ("Scan.FrameDataGrab", [14, 1]),
    ("FolMe.XYPosGet", [1]),
])
def test_scalar_requests_decode(command, args):
    spec = commands()[command]
    body = _oracle_encode_request(command, list(args), spec.arg_fmts)
    got = codec.decode_args(body, spec.arg_fmts)
    for fmt, want, have in zip(spec.arg_fmts, args, got):
        if fmt == "f":
            assert have == pytest.approx(want, rel=1e-6)
        else:
            assert have == want


@requires_mast
def test_signals_vals_get_prepended_int_array_request():
    spec = commands()["Signals.ValsGet"]  # +*i, I
    body = _oracle_encode_request("Signals.ValsGet", [[0, 14, 30], 1], spec.arg_fmts)
    assert codec.decode_args(body, spec.arg_fmts) == [[0, 14, 30], 1]


@requires_mast
def test_bias_spectr_start_string_request():
    spec = commands()["BiasSpectr.Start"]  # I, +*c
    body = _oracle_encode_request("BiasSpectr.Start", [1, "sts_"], spec.arg_fmts)
    assert codec.decode_args(body, spec.arg_fmts) == [1, "sts_"]


@requires_mast
def test_scan_props_set_string_array_request():
    spec = commands()["Scan.PropsSet"]  # I I I +*c +*c +*c I
    args = [0, 0, 1, "ser", "cmt", ["Bias", "Z-Controller"], 0]
    body = _oracle_encode_request("Scan.PropsSet", list(args), spec.arg_fmts)
    got = codec.decode_args(body, spec.arg_fmts, array_string_args=(5,))
    assert got == args
    # and the plausibility fallback finds it too
    assert codec.decode_args(body, spec.arg_fmts) == args


@requires_mast
def test_zctrl_setpnt_set_and_scan_buffer_set_arrays():
    # ZCtrl.SetpntSet: a single float32
    spec = commands()["ZCtrl.SetpntSet"]
    assert tuple(spec.arg_fmts) == ("f",)
    body = _oracle_encode_request("ZCtrl.SetpntSet", [50e-12], spec.arg_fmts)
    assert codec.decode_args(body, spec.arg_fmts)[0] == pytest.approx(50e-12, rel=1e-6)

    # Scan.BufferSet: the wire body is  Number of channels (int) · Channel indexes (int[])
    # · Pixels (int) · Lines (int).  The upstream nanonis_spm method takes THREE Python
    # arguments and declares ('+*i', 'i', 'i'): the '+' means the client itself prepends
    # the int32 count in front of the array, so "Number of channels" is never a
    # caller-supplied argument.  MAST's patch does not touch Scan_BufferSet, and MAST calls
    # it as Scan_BufferSet(channels, 0, 0) (ConfigureScan) / (channels, pixels, lines)
    # (SetScanBuffer).  The local registry preserves this contract; an earlier
    # version of this test expected an explicit leading count and skipped when it was
    # absent, which hid the contract instead of asserting it.
    spec = commands()["Scan.BufferSet"]
    assert tuple(spec.arg_fmts) == ("+*i", "i", "i"), spec.arg_fmts
    # Parameter labels are local metadata; only types and byte order are on the wire.
    args = [[0, 14, 30], 256, 256]
    body = _oracle_encode_request("Scan.BufferSet", list(args), spec.arg_fmts)
    assert len(body) == 4 + 3 * 4 + 4 + 4                 # count + 3 indexes + pixels + lines
    assert struct.unpack(">i", body[:4])[0] == 3           # the client wrote the count
    assert struct.unpack(">3i", body[4:16]) == (0, 14, 30)
    assert struct.unpack(">ii", body[16:24]) == (256, 256)
    assert codec.decode_args(body, spec.arg_fmts) == args
    # MAST's ConfigureScan shape: channels only, pixels/lines left at 0 ("keep")
    body0 = _oracle_encode_request("Scan.BufferSet", [[14], 0, 0], spec.arg_fmts)
    assert len(body0) == 4 + 4 + 4 + 4
    assert codec.decode_args(body0, spec.arg_fmts) == [[14], 0, 0]


# ────────────────────────────── end-to-end over TCP ──────────────────────────────

@requires_mast
def test_loopback_server_with_real_client():
    d = Dispatcher(unloaded_modules={"OsciHR"})
    state = {"bias": 0.1}

    @d.handles("Bias_Get")
    def _bias_get():
        return state["bias"]

    @d.handles("Bias_Set")
    def _bias_set(v):
        state["bias"] = float(v)

    @d.handles("Signals_NamesGet")
    def _names():
        return [None, None, ["Current (A)", "Z (m)"]]

    Client = _client()
    with WireServer(d, ports=[0]) as srv:
        port = srv.bound_ports[0]
        sock = socket.create_connection(("127.0.0.1", port), timeout=2)
        try:
            nn = Client(sock)
            assert nn.Bias_Get()[2][0] == pytest.approx(0.1)
            assert nn.Bias_Set(0.02)[0] == ""
            assert nn.Bias_Get()[2][0] == pytest.approx(0.02)
            assert nn.Signals_NamesGet()[2][2] == ["Current (A)", "Z (m)"]
            # not-loaded module → error (NeedModule text when the client can extract it), link stays up
            err = nn.quickSend("OsciHR.Run", [], [], [])[0]
            assert "NeedModule" in err
            # unimplemented verb of a loaded module → error, link still up
            err2 = nn.quickSend("ZCtrl.OnOffGet", [], [], ["I"])[0]
            assert err2 != "" and "NeedModule" not in err2
            assert nn.Bias_Get()[2][0] == pytest.approx(0.02)
        finally:
            sock.close()


def test_dispatcher_unknown_command_is_error_body():
    d = Dispatcher()
    body = d.call("No.SuchVerb", b"")
    status, ln = struct.unpack(">ii", body[:8])
    assert status != 0 and ln == len(body) - 8 and b"not recognized" in body


def test_fill_counts_two_ints_before_string_array():
    vals = codec.fill_counts([None, None, ["ab", "c"]], ["i", "i", "*+c"])
    assert vals == [(4 + 2) + (4 + 1), 2, ["ab", "c"]]


def test_module_not_running_text_has_need_module():
    assert "NeedModule" in ModuleNotRunning("Osci2T").description
