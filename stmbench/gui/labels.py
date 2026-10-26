"""The window's strings as ``(English, Chinese)`` pairs (no Qt).

``--lang en`` shows the English; ``--lang zh`` (the default) shows the Chinese followed by its English,
as the web console does. A label says what its control is, with its unit, and nothing more: the
window gives no advice. The action parameters' labels live in :mod:`stmbench.gui.layout`.
"""
from __future__ import annotations

Pair = tuple[str, str]


def both(en: str, zh: str, lang: str = "zh") -> str:
    """One string for ``lang``: English alone, or the Chinese followed by its English."""
    if lang == "en" or not zh or zh == en:
        return en
    return f"{zh} {en}"


def tr(key: str, lang: str = "zh") -> str:
    return both(*S[key], lang)


S: dict[str, Pair] = {
    # window and status strip
    "app": ("STM-Bench console", "STM-Bench 控制台"),
    "ready": ("READY", "就绪"),
    "busy": ("BUSY", "运行中"),
    "finished": ("FINISHED", "已结束"),
    "queued": ("queued", "排队"),
    "time": ("Instrument time", "仪器时间"),
    "actions": ("Actions", "动作"),
    "bias": ("Bias", "偏压"),
    "setpoint": ("Setpoint", "设定点"),
    "feedback": ("Z feedback", "Z 反馈"),
    "current": ("Current", "电流"),
    "z": ("Z", "Z"),
    "range": ("Range", "量程"),
    "tip": ("Tip x, y", "针尖 x, y"),
    "withdrawn": ("Withdrawn", "撤针"),
    "lockin": ("Lock-in", "锁相"),
    "oscillation": ("Oscillation", "振荡"),
    "no_call": ("No call yet.", "还没有操作。"),
    "refused": ("refused", "被拒绝"),
    "stays_open": ("This program stays open; it is closed from outside.", "此程序保持打开，由外部关闭。"),
    # tabs
    "tab_task": ("Task", "任务"),
    "tab_manual": ("Manual", "手册"),
    "tab_report": ("Report", "报告"),
    "tab_data": ("Data", "数据"),
    "tab_log": ("Log", "日志"),
    "tab_operate": ("Operate", "操作"),
    "tab_spectroscopy": ("Spectroscopy", "谱学"),
    # report
    "report_title": ("Report a result", "报告结果"),
    "rep_value": ("value", "数值"),
    "rep_x": ("x (nm)", "x (nm)"),
    "rep_y": ("y (nm)", "y (nm)"),
    "rep_frame": ("frame the position was read in", "位置读自哪一帧"),
    "rep_evidence": ("evidence (file names, comma-separated)", "证据（文件名，逗号分隔）"),
    "rep_viewed": ("+ viewed", "+ 当前"),
    "rep_unit": ("unit", "单位"),
    "rep_sigma": ("sigma", "不确定度"),
    "rep_note": ("note", "备注"),
    "rep_submit": ("Submit report", "提交报告"),
    "finish_title": ("Finish the episode", "结束实验"),
    "finish": ("Finish episode", "结束实验"),
    "finish_ask": ("No actions or reports after this. Finish now?", "结束后不能再操作或报告。确定结束？"),
    "finish_yes": ("Confirm finish", "确认结束"),
    "finish_no": ("Cancel", "取消"),
    "over": ("The episode is over; the record is written.", "实验已结束，记录已写入。"),
    # data
    "frames": ("Frames", "帧"),
    "spectra": ("Spectra", "谱"),
    "col_name": ("name", "名称"),
    "col_start": ("start", "开始"),
    "col_size": ("size nm", "边长 nm"),
    "col_pixels": ("px", "像素"),
    "col_centre": ("centre nm", "中心 nm"),
    "col_angle": ("angle", "角度"),
    "col_channels": ("channels", "通道"),
    "col_kind": ("kind", "类型"),
    "col_tip": ("tip x, y nm", "针尖 x, y nm"),
    "col_points": ("points", "点数"),
    "col_time": ("time", "时间"),
    # viewer
    "ch_z": ("Z", "Z"),
    "ch_current": ("Current", "电流"),
    "ch_didv": ("dI/dV", "dI/dV"),
    "dir_forward": ("Forward", "正向"),
    "dir_backward": ("Backward", "反向"),
    "flat_line": ("Line", "逐行"),
    "flat_plane": ("Plane", "平面"),
    "flat_none": ("None", "无"),
    "tool_marker": ("Marker", "标记"),
    "tool_profile": ("Profile", "剖面"),
    "clear_markers": ("Clear markers", "清除标记"),
    "lbl_channel": ("Channel", "通道"),
    "lbl_direction": ("Direction", "方向"),
    "lbl_flatten": ("Flatten", "拉平"),
    "lbl_tool": ("Tool", "工具"),
    "lbl_x": ("x axis", "横轴"),
    "lbl_y": ("y axis", "纵轴"),
    "no_view": ("Nothing shown.", "没有显示的内容。"),
    "mk_num": ("#", "#"),
    "mk_x": ("x nm", "x nm"),
    "mk_y": ("y nm", "y nm"),
    "mk_value": ("value", "值"),
    "mk_frame": ("frame", "帧"),
    "mk_dist": ("to previous nm", "距上一个 nm"),
    "mk_tip": ("Tip ← marker", "针尖 ← 标记"),
    "mk_scan": ("Scan centre ← marker", "扫描中心 ← 标记"),
    "mk_report": ("Report ← marker", "报告 ← 标记"),
    "mk_none": ("No marker selected.", "没有选中的标记。"),
    "profile": ("Profile", "剖面"),
    "trace": ("Reply", "回复"),
    "last_frames": ("Last frames", "最近的帧"),
    # log
    "log_empty": ("Nothing yet.", "还没有操作。"),
    "log_running": ("running", "运行中"),
    # on/off, yes/no
    "on": ("On", "开"),
    "off": ("Off", "关"),
    "yes": ("yes", "是"),
    "no": ("no", "否"),
}
