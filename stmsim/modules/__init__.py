"""Bind every implemented controller module to a world → a ready :class:`Dispatcher`."""
from __future__ import annotations

from ..physics.world import World
from ..wire.server import Dispatcher
from .core_modules import (bind_bias, bind_current, bind_datalog, bind_folme, bind_lockin,
                           bind_marks, bind_piezo, bind_safetip, bind_signals, bind_util,
                           bind_zctrl)
from .osci_module import bind_osci
from .pll_module import bind_pll, bind_pllfreqswp
from .scan_module import bind_scan
from .spectr_module import bind_biasspectr, bind_zspectr
from .tipwork_module import bind_autoapproach, bind_motor, bind_tipshaper


def build_dispatcher(world: World) -> Dispatcher:
    loaded = world.rig.modules_loaded or None
    d = Dispatcher(loaded_modules=loaded, unloaded_modules=world.rig.modules_not_loaded)
    d.world = world      # lets the call log stamp clock state / mark undelivered replies
    binders = [bind_bias, bind_zctrl, bind_current, bind_folme, bind_piezo, bind_signals, bind_util,
               bind_safetip, bind_lockin, bind_marks, bind_datalog, bind_scan, bind_tipshaper,
               bind_motor, bind_autoapproach, bind_biasspectr, bind_zspectr, bind_osci]
    # binding is harmless while the module is unavailable (the gate fires first), but building
    # the PLL state on an STM rig is not: it would make signal 17 report physics it has none of
    if "PLL" in (world.rig.modules_loaded or ()):
        binders += [bind_pll, bind_pllfreqswp]
    for bind in binders:
        bind(d, world)
    return d
