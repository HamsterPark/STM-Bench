"""Which action goes where on the panels, in what order, under which short label (no Qt).

Only the arrangement lives here. What a field is - number, integer, choice, boolean - and its
default, range and whether it is required come from :data:`stmbench.instrument.ACTIONS` when the
window is built, so a change to the action table shows up on the screen. The arrangement must cover the
table exactly (``tests/test_gui_layout.py``): every action, every parameter, nothing else.

A panel is a stack of groups; a group is a few lines; a line is a row of cells. A cell is one
parameter's field (:class:`F`), the button that runs an action (:class:`Run`), the On / Off pair of an
action that switches something (:class:`OnOff`), or one button per allowed value that runs the action
at once (:class:`Choices`). Labels say what a control is, with its unit; they carry no advice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class F:
    """One parameter: a text field, or a segmented selector when the parameter is a boolean or a choice."""
    action: str
    param: str
    en: str
    zh: str
    unit: str = ""


@dataclass(frozen=True)
class Run:
    """The button that runs ``action`` with its group's fields (its object name is ``run.<action>``)."""
    action: str
    en: str
    zh: str


@dataclass(frozen=True)
class OnOff:
    """On and Off buttons (``run.<action>.on`` / ``.off``) that run an action whose ``on`` parameter
    switches something, with the fields beside them."""
    action: str
    en: str
    zh: str


@dataclass(frozen=True)
class Choices:
    """One button per allowed value of ``param`` (``run.<action>.<value>``); a click runs the action."""
    action: str
    param: str
    en: str
    zh: str
    unit: str = ""


Cell = Union[F, Run, OnOff, Choices]


@dataclass(frozen=True)
class Group:
    key: str
    en: str
    zh: str
    lines: tuple[tuple[Cell, ...], ...]


OPERATE: tuple[Group, ...] = (
    Group("scan", "Scan", "扫描", (
        (F("scan", "center_x_nm", "centre x", "中心 x", "nm"),
         F("scan", "center_y_nm", "centre y", "中心 y", "nm"),
         F("scan", "size_nm", "size", "边长", "nm"),
         F("scan", "pixels", "pixels", "像素")),
        (F("scan", "angle_deg", "angle", "角度", "°"),
         F("scan", "line_time_s", "line time", "行时间", "s"),
         F("scan", "direction", "slow axis", "慢轴"),
         Run("scan", "Scan", "扫描")),
    )),
    Group("junction", "Junction", "隧道结", (
        (F("set_bias", "bias_v", "bias", "偏压", "V"),
         Run("set_bias", "Set bias", "设偏压"),
         F("set_setpoint", "current_pa", "setpoint", "设定点", "pA"),
         Run("set_setpoint", "Set setpoint", "设设定点")),
        (OnOff("feedback", "Z feedback", "Z 反馈"),
         Choices("set_current_range", "full_scale_na", "preamp range", "前放量程", "nA")),
    )),
    Group("tip", "Tip position", "针尖位置", (
        (F("move_tip", "x_nm", "tip x", "针尖 x", "nm"),
         F("move_tip", "y_nm", "tip y", "针尖 y", "nm"),
         F("move_tip", "speed_nm_s", "speed", "速度", "nm/s"),
         Run("move_tip", "Move tip", "移动针尖")),
        (F("drift_comp", "vx_nm_per_min", "drift vx", "漂移 vx", "nm/min"),
         F("drift_comp", "vy_nm_per_min", "drift vy", "漂移 vy", "nm/min"),
         F("drift_comp", "vz_pm_per_min", "drift vz", "漂移 vz", "pm/min"),
         OnOff("drift_comp", "Drift comp.", "漂移补偿")),
    )),
    Group("conditioning", "Tip conditioning", "针尖修整", (
        (F("pulse", "bias_v", "pulse bias", "脉冲偏压", "V"),
         F("pulse", "width_ms", "width", "脉宽", "ms"),
         F("pulse", "z_hold", "hold Z", "保持 Z"),
         Run("pulse", "Pulse", "脉冲")),
        (F("poke", "depth_nm", "depth", "深度", "nm"),
         F("poke", "bias_v", "poke bias", "压针偏压", "V"),
         F("poke", "dwell_s", "dwell", "停留", "s"),
         Run("poke", "Poke", "压针")),
    )),
    Group("coarse", "Approach and coarse motion", "进针与粗动", (
        (F("coarse_move", "direction", "direction", "方向"),
         F("coarse_move", "steps", "steps", "步数"),
         Run("coarse_move", "Coarse move", "粗动")),
        (F("approach", "timeout_s", "timeout", "超时", "s"),
         Run("approach", "Approach", "进针"),
         Run("withdraw", "Withdraw", "撤针"),
         F("wait", "seconds", "wait", "等待", "s"),
         Run("wait", "Wait", "等待")),
    )),
)

SPECTROSCOPY: tuple[Group, ...] = (
    Group("sts", "Bias spectroscopy (STS)", "偏压谱 STS", (
        (F("sts", "v_start", "start", "起点", "V"),
         F("sts", "v_end", "end", "终点", "V"),
         F("sts", "points", "points", "点数"),
         F("sts", "settle_ms", "settle", "稳定", "ms")),
        (F("sts", "integration_ms", "integration", "积分", "ms"),
         F("sts", "z_offset_nm", "z offset", "Z 偏移", "nm"),
         F("sts", "backward", "sweep back", "回扫"),
         Run("sts", "Run STS", "测 STS")),
    )),
    Group("zspec", "Z spectroscopy", "Z 谱", (
        (F("zspec", "sweep_nm", "sweep", "扫程", "nm"),
         F("zspec", "points", "points", "点数"),
         F("zspec", "z_offset_nm", "z offset", "Z 偏移", "nm"),
         F("zspec", "sweeps", "sweeps", "平均次数")),
        (F("zspec", "settle_ms", "settle", "稳定", "ms"),
         F("zspec", "integration_ms", "integration", "积分", "ms"),
         F("zspec", "backward", "sweep back", "回扫"),
         Run("zspec", "Run Z spectrum", "测 Z 谱")),
    )),
    Group("lockin", "Lock-in", "锁相放大器", (
        (F("lockin", "amplitude_mv", "amplitude", "调制幅度", "mV"),
         OnOff("lockin", "Lock-in", "锁相")),
    )),
    Group("pll", "qPlus oscillation (PLL)", "qPlus 振荡 PLL", (
        (F("pll", "amplitude_pm", "amplitude", "振幅", "pm"),
         OnOff("pll", "Oscillation", "振荡")),
    )),
)

TABS: tuple[tuple[str, tuple[Group, ...]], ...] = (("operate", OPERATE), ("spectroscopy", SPECTROSCOPY))


def cells() -> list[tuple[str, Cell]]:
    """Every cell with the key of its group, in panel order."""
    return [(g.key, c) for _, groups in TABS for g in groups for line in g.lines for c in line]


def covered_params() -> list[tuple[str, str]]:
    """The ``(action, parameter)`` pairs the arrangement gives a control to (an On / Off pair is the
    ``on`` parameter's control)."""
    out: list[tuple[str, str]] = []
    for _, c in cells():
        if isinstance(c, (F, Choices)):
            out.append((c.action, c.param))
        elif isinstance(c, OnOff):
            out.append((c.action, "on"))
    return out


def runners() -> list[str]:
    """The actions the arrangement can run (each by a button, or by an On / Off pair, or by choices)."""
    return [c.action for _, c in cells() if isinstance(c, (Run, OnOff, Choices))]


def group_of_action() -> dict[str, str]:
    """The group each action's reply line belongs to."""
    return {c.action: key for key, c in cells()}
