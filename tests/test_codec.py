"""Wire codec and dispatcher pieces that need no controller client: an unknown command is
an error body, counts are filled in before a string array, and a module that is not
running says NeedModule."""
from __future__ import annotations

import struct

from stmsim.wire import codec
from stmsim.wire.errors import ModuleNotRunning
from stmsim.wire.server import Dispatcher


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
