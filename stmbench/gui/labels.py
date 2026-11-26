"""The window's strings as ``(English, Chinese)`` pairs (no Qt).

``--lang en`` shows the English; ``--lang zh`` (the default) shows the Chinese followed by its English,
as the web console does. A label says what its control is, with its unit, and nothing more: the
window gives no advice about the experiment (its only hints say how to use the viewer). The action parameters' labels live in :mod:`stmbench.gui.layout`.
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


def one(key: str, lang: str = "zh") -> str:
    """One language only (the Chinese in ``zh``, else the English): for a line of many short readings,
    which in both languages at once would not fit the width of the window."""
    en, zh = S[key]
    return zh if lang != "en" and zh else en


def two_lines(key: str, lang: str = "zh") -> str:
    """A tab or column title: English alone, or the Chinese above its English."""
    en, zh = S[key]
    return en if lang == "en" or not zh or zh == en else zh + chr(10) + en


#: Under the Tip group's during-move fields in the English window (screen trial 22: "labelled keep and not explained"): what the
#: console does with them, in the words of the screen's own labels ("Move tip", "keep"); no advice on values or on when to fill them
DURING_HINT_EN = "Filled: set before Move tip, set back after. Blank: keep. Reposition ignores them and speed."

#: What the console does with the Tip group's "during move" fields, said in front of the instrument's manual (the manual itself
#: is the instrument's, the text an MCP agent gets, and stays byte for byte). Facts only: no value and no occasion is named.
MOVE_NOTE: Pair = (
    "Console: the fields move range and move setpoint keep their values; while either is filled, every Move tip uses "
    "them (blank: keep). Move tip first checks every value (one invalid: nothing runs), then sets the filled ones, "
    "moves the tip, and sets each setting it changed back to the value the "
    "instrument reported before; a setting that did not change is not set back. Each call is logged as if clicked. If a "
    "call is refused, the settings already changed are set back and the band shows the refusal; a refused set-back stops "
    "there. Reposition moves the tip to x and y alone, without the move range, the move setpoint and the speed.",
    "控制台：移动时量程、移动时设定点两栏的值会一直保留，只要填着，每次移动针尖都会用（留空＝不变）。填了任一栏，移动针尖先检查所有值"
    "（有一个不合法就都不运行），再设上已填的项，移动针尖，然后把改过的项恢复为之前仪器报告的值；没有变的不再恢复。"
    "每个调用照常记入日志。调用被拒时，已改的设置同样恢复，红条显示拒绝；恢复被拒则就地停止。仅移位只按 x、y 移动针尖，不用移动时量程、移动时设定点和速度。",
)


#: What the viewer's Flatten modes do (stmbench.render.display_array defines them; the console only says so). Facts only.
FLATTEN_NOTE: Pair = (
    "Flatten: Line subtracts each row's median (tilt along a row stays); Plane subtracts one fitted plane "
    "(row-to-row steps stay); None subtracts nothing.",
    "拉平：逐行 减去每一行的中位数（行内倾斜保留）；平面 减去整幅拟合平面（行间台阶保留）；无 不减。",
)


def with_console_notes(manual: str, lang: str = "en") -> str:
    """The manual the Manual tab shows: the console's notes in the window's language, then the instrument's manual byte for
    byte. The notes come first because the reader of screen trial 22 never scrolled to the end of the manual to find them."""
    i = 0 if lang == "en" else 1
    return "\n\n".join([MOVE_NOTE[i], FLATTEN_NOTE[i], manual])


S: dict[str, Pair] = {
    # window and status strip
    "app": ("STM-Bench console", "控制台"),
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
    "tip": ("Tip", "针尖"),
    "withdrawn": ("Withdrawn", "撤针"),
    "lockin": ("Lock-in", "锁相"),
    "oscillation": ("Oscillation", "振荡"),
    "drift": ("Drift comp.", "漂移补偿"),
    "no_call": ("No call yet.", "还没有操作。"),
    "running": ("running", "运行中"),
    "cancel_queued": ("Cancel queued", "取消排队"),
    "cancelled": ("cancelled", "已取消"),
    "cancelled_note": ("cancelled queued", "已取消排队"),
    "not_run": ("not run: an earlier call was refused", "未执行:前面的调用被拒绝了"),
    # a refusal holds the operator's calls until the reader acknowledges it (the band on the status strip)
    "held": ("HELD", "挂起"),
    "not_run_held": ("not run: the refusal is not acknowledged", "未执行:拒绝尚未确认"),
    "refused_head": ("REFUSED", "被拒绝"),
    "error_head": ("ERROR", "出错"),
    "held_note": ("Calls that change the instrument or the record are held until this is acknowledged.",
                  "确认之前，会改变仪器或记录的调用都被挂起。"),
    "band_not_run": ("Not run", "未执行"),
    "ack": ("Acknowledge", "知道了"),
    "acknowledged": ("acknowledged the refusal of", "已确认被拒绝的调用"),
    "field_range": ("range", "范围"),
    "field_one_of": ("one of", "取值"),
    "recent_title": ("Recent results", "最近结果"),
    "stays_open": ("This program stays open; it is closed from outside.", "此程序保持打开，由外部关闭。"),
    # the instrument's facts (top of the Manual tab)
    "facts_title": ("Instrument facts", "仪器参数"),
    "manual_wide": ("Wide view", "宽屏阅读"),
    "manual_narrow": ("Narrow view", "窄屏阅读"),
    "fact_material": ("Material", "材料"),
    "fact_temperature": ("Temperature", "温度"),
    "fact_piezo_xy": ("Piezo XY range", "压电 XY 量程"),
    "fact_piezo_z": ("Piezo Z range", "压电 Z 量程"),
    "fact_bias_range": ("Bias range", "偏压范围"),
    "fact_current_ranges": ("Current ranges (full scale)", "电流量程（满量程）"),
    "fact_budget": ("Time budget", "时间预算"),
    "fact_qplus_f0": ("qPlus f0", "qPlus 共振频率 f0"),
    "fact_qplus_k": ("qPlus k", "qPlus 弹性常数 k"),
    "fact_qplus_q": ("qPlus Q", "qPlus 品质因数 Q"),
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
    # the report form's fields keep the names ReportResult and the rule card give them; these say what each is
    "rep_value": ("", "数值"),
    "rep_x_nm": ("nm", "nm"),
    "rep_y_nm": ("nm", "nm"),
    "rep_frame": ("the frame of the position", "位置读自哪一帧"),
    "rep_evidence": ("", ""),                          # beside the button: the placeholder says what goes in
    "rep_evidence_hint": ("files, comma-separated", "文件名，逗号分隔"),
    "rep_unit": ("", "单位"),
    "rep_sigma": ("", "不确定度"),
    "rep_note": ("", "备注"),
    "rep_viewed": ("shown file", "显示中的文件"),
    "rep_no_position": ("the picked claim takes no position: nothing copied", "所选项不需要位置:未复制"),
    "rep_nothing_shown": ("no frame or spectrum is shown: no file name to add", "查看器里没有显示帧或谱:没有可加入的文件名"),
    "rep_submit": ("Submit report", "提交报告"),
    "finish": ("Finish episode", "结束实验"),
    "finish_ask": ("No actions or reports after this. Finish now?", "结束后不能再操作或报告。确定结束？"),
    "finish_yes": ("Confirm finish", "确认结束"),
    "finish_no": ("Cancel", "取消"),
    "over": ("The episode is over; the record is written.", "实验已结束，记录已写入。"),
    # data
    "frames": ("Frames", "帧"),
    "spectra": ("Spectra", "谱"),
    "move_traces": ("Move traces", "移动轨迹"),
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
    "clear_markers": ("Clear image markers", "清除图像标记"),
    "lbl_flatten": ("Flatten", "拉平"),
    "lbl_contrast": ("Contrast", "对比度"),
    "lbl_zoom": ("Zoom", "缩放"),
    "zoom_fit": ("Fit", "全图"),
    "zoom_2": ("×2", "×2"),
    "zoom_4": ("×4", "×4"),
    "zoom_8": ("×8", "×8"),
    "tool_pan": ("Pan", "平移"),
    "contrast_auto": ("auto", "自动"),
    "contrast_full": ("full", "全幅"),
    "contrast_visible": ("visible", "可见区"),
    "lbl_tool": ("Tool", "工具"),
    "lbl_x": ("x axis", "横轴"),
    "lbl_y": ("y axis", "纵轴"),
    # the two columns only a move trace has (the other columns are symbols: V, I, dI/dV, z)
    "col_dist": ("distance", "距离"),
    "col_t": ("time", "时间"),
    "no_view": ("Nothing shown.", "没有显示的内容。"),
    "scan_running": ("scan running…", "扫描进行中…"),
    # the readout line's usage hints, shown while it has no reading: how the viewer works, never about the experiment
    # each says what the pointer does with the tool picked, and that the wheel zooms about the pointer (p3check1, finding 1:
    # a sighted agent zoomed about the centre with the buttons and then had to pan to a ring at the edge)
    "hint_image": ("Hover to read; click to mark, drag to pan; wheel to zoom about the pointer",
                   "悬停读数，单击放标记，放大后拖动平移，滚轮以指针处为中心缩放"),
    "hint_image_profile": ("Hover to read; drag a line to plot along it; wheel to zoom about the pointer",
                           "悬停读坐标和数值，拖动画剖面线，滚轮以指针处为中心缩放"),
    "hint_image_pan": ("Hover to read; drag to move the zoomed view; wheel to zoom about the pointer",
                       "悬停读坐标和数值，拖动平移，滚轮以指针处为中心缩放"),
    "hint_plot": ("Hover over the plot to read a point; click to pin it; wheel to zoom",
                  "悬停查看数值，点击固定读数，滚轮缩放"),
    "no_frame_click": ("no frame shown: nothing marked", "没有显示的帧：未标记"),
    "delete_marker": ("Delete marker", "删除标记"),
    "mk_num": ("#", "#"),
    "mk_x": ("x nm", "x nm"),
    "mk_y": ("y nm", "y nm"),
    "mk_value": ("value", "值"),
    "mk_frame": ("frame", "帧"),
    "mk_start": ("start", "起始"),
    "mk_dist": ("to prev. nm", "距前 nm"),
    "mk_tip": ("Tip ← marker", "针尖 ← 标记"),
    "mk_scan": ("Scan centre ← marker", "扫描中心 ← 标记"),
    "mk_report": ("Report ← marker", "报告 ← 标记"),
    "mk_none": ("No marker selected.", "没有选中的标记。"),
    "profile": ("Profile", "剖面"),
    "trace": ("Reply", "回复"),
    # the small plot beside the marker table is a thumbnail of the big one (screen trial 23: its axes were 65 px wide)
    "plot_enlarge": ("Click to enlarge", "点击放大"),
    "plot_return": ("Click to go back", "点击返回"),
    "plot_back": ("← Back to view", "← 返回视图"),
    # log
    "log_empty": ("Nothing yet.", "还没有操作。"),
    "log_running": ("running", "运行中"),
    # on/off, yes/no
    "on": ("On", "开"),
    "off": ("Off", "关"),
    "yes": ("yes", "是"),
    "no": ("no", "否"),
    "keep": ("keep", "保持"),
    # after a report of a claim whose card line asks to cite data, sent with no evidence files (screen trial 24): what was sent, nothing more
    "no_evidence_note": ("no evidence files named", "没有列出证据文件"),
    # the viewer, added after the screen trials of 2026-10-27/28 (gui-viewer)
    "clear_readings": ("Clear plot readings", "清除曲线读数"),
}
