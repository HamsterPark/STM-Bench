"""TipShaper (poke), Motor, AutoApproach."""
from __future__ import annotations

import time

from ..physics.world import World
from ..wire.errors import BadArguments, WireError

_DIRECTIONS = {0: "X+", 1: "X-", 2: "Y+", 3: "Y-", 4: "Z+", 5: "Z-"}


def bind_tipshaper(d, w: World):
    props = {"switch_off_delay": 0.05, "change_bias": 2, "bias_v": 0.02, "tip_lift_m": -0.5e-9,
             "lift_time_1_s": 0.1, "bias_lift_v": 0.0, "bias_settling_s": 0.5,
             "lift_height_m": 0.5e-9, "lift_time_2_s": 0.1, "end_wait_s": 0.5, "restore_feedback": 1}

    @d.handles("TipShaper_PropsSet")
    def _set(sod, change_bias, bias_v, lift, t1, bias_lift, settle, lift_h, t2, end_wait, restore):
        props.update(switch_off_delay=float(sod), change_bias=int(change_bias), bias_v=float(bias_v),
                     tip_lift_m=float(lift), lift_time_1_s=float(t1), bias_lift_v=float(bias_lift),
                     bias_settling_s=float(settle), lift_height_m=float(lift_h), lift_time_2_s=float(t2),
                     end_wait_s=float(end_wait), restore_feedback=int(restore))

    d.register("TipShaper_PropsGet", lambda: [props["switch_off_delay"], props["change_bias"], props["bias_v"],
                                              props["tip_lift_m"], props["lift_time_1_s"], props["bias_lift_v"],
                                              props["bias_settling_s"], props["lift_height_m"], props["lift_time_2_s"],
                                              props["end_wait_s"], props["restore_feedback"]])

    @d.handles("TipShaper_Start")
    def _start(wait, timeout_ms):
        if w.withdrawn:
            raise WireError("Tip Shaper: tip is withdrawn (Z controller off at the retract limit)")
        # controller three-valued booleans: 1 = True, 2 = False
        p = dict(props)
        p["change_bias"] = (props["change_bias"] == 1)
        p["restore_feedback"] = (props["restore_feedback"] != 2)
        w.tip_shaper_start(p)
        if int(wait):
            total = (props["switch_off_delay"] + props["lift_time_1_s"] + props["bias_settling_s"]
                     + props["lift_time_2_s"] + props["end_wait_s"] + 0.3)
            time.sleep(min(total, 4.0))


def bind_motor(d, w: World):
    supported_counter = bool(w.rig.get("motor.step_counter_supported", False))

    @d.handles("Motor_StartMove")
    def _move(direction, steps, group, wait):
        dname = _DIRECTIONS.get(int(direction))
        if dname is None:
            raise BadArguments("Motor.StartMove", f"direction {direction}")
        n = int(steps)
        w.motor_move(dname, n)
        if int(wait):
            time.sleep(min(n / max(w.coarse.freq_hz, 1.0), 4.0))

    d.register("Motor_StopMove", lambda: None)

    def _nomodule():
        raise WireError("NeedModule: Cannot access 'Motor Control' Module. Please make sure it is running.")

    if supported_counter:
        d.register("Motor_PosGet", lambda group, timeout: [0.0, 0.0, -w.coarse.coarse_gap_m])
        d.register("Motor_StepCounterGet", lambda rx, ry, rz: [w.coarse.xy_steps["X+"] - w.coarse.xy_steps["X-"],
                                                                w.coarse.xy_steps["Y+"] - w.coarse.xy_steps["Y-"],
                                                                w.coarse.steps_done])
    else:
        d.register("Motor_PosGet", lambda group, timeout: _nomodule())
        d.register("Motor_StepCounterGet", lambda rx, ry, rz: _nomodule())
    d.register("Motor_FreqAmpGet", lambda axis: [w.coarse.freq_hz, w.coarse.amp_v])

    @d.handles("Motor_FreqAmpSet")
    def _fa(freq, amp, axis):
        w.coarse.freq_hz, w.coarse.amp_v = float(freq), float(amp)


def bind_autoapproach(d, w: World):
    d.register("AutoApproach_Open", lambda: None)

    @d.handles("AutoApproach_OnOffSet")
    def _onoff(on):
        if int(on):
            w.auto_approach_start()
        else:
            w.auto_approach_stop()

    d.register("AutoApproach_OnOffGet", lambda: 1 if w.auto_approach_poll() else 0)
