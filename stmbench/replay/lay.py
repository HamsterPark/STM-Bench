"""Plain-language text for the replay: what each tool call does, what state the tip is in,
what each paper is about and what each reported number means.

Everything here is presentation. It never feeds a verdict, and it describes the tip from the
ledger's truth — which the model never saw — so it belongs in the replay only.
"""
from __future__ import annotations

import math
import re
from typing import Any

# ── units ───────────────────────────────────────────────────────────────────
_SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0,
       "k": 1e3, "M": 1e6, "G": 1e9}
_SI_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([fpnuµμmkMG]?)")


def si(value: Any) -> float | None:
    """A tool argument as a number: ``40e-9``, ``"40n"``, ``"8.355p"``, ``"0.4"`` (the
    SI-prefixed strings MAST accepts). ``None`` when it is not a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    m = _SI_RE.match(str(value))
    if not m:
        return None
    return float(m.group(1)) * _SI[m.group(2)]


def num(x: float | None, digits: int = 3) -> str:
    """Up to ``digits`` significant figures, no trailing zeros."""
    if x is None or not math.isfinite(x):
        return "?"
    if x == 0:
        return "0"
    d = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
    s = f"{x:.{d}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def nm(x_m: float | None, digits: int = 3) -> str:
    return f"{num(x_m * 1e9, digits)} nm" if x_m is not None else "?"


def with_prefix(x: float | None, unit: str) -> str:
    """``8e-3, "V"`` → ``8 mV``; ``85e-9, "A"`` → ``85 nA``."""
    if x is None:
        return "?"
    a = abs(x)
    for f, p in ((1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, "")):
        if a < f * 1000:
            return f"{num(x / f)} {p}{unit}"
    return f"{num(x)} {unit}"


def duration(sim_s: float) -> str:
    """Instrument time in words: ``45 秒`` / ``8 分钟`` / ``1 小时 12 分``."""
    s = max(0.0, float(sim_s))
    if s < 90:
        return f"{int(round(s))} 秒"
    m = int(round(s / 60))
    if m < 60:
        return f"{m} 分钟"
    return f"{m // 60} 小时 {m % 60} 分"


# ── what the page says before anything happens ──────────────────────────────
INTRO = {
    "title": "这是在看什么？",
    "paragraphs": [
        "扫描隧道显微镜（STM）用一根极细的金属针，在样品表面上方大约一纳米处扫描，"
        "通过针尖与表面之间的隧穿电流测量表面形貌和原子尺度起伏。",
        "针尖状态会影响图像：针尖钝了，图就发糊；针尖末端分了叉，图上每个东西都有重影；"
        "针尖撞过表面，图上会出现一道道横纹。",
        "真实的实验里，谁也看不见针尖，科学家只能从图像去猜它现在好不好。接受测试的 AI 也一样：它只看得到自己扫出来的图。",
        "这个页面提供完整状态视图：左边是 AI 看到的和做的；右边显示评估期间记录的针尖状态、"
        "样品状态和判定所需的参考信息。",
    ],
}

#: family → the paper in plain words. ``goal`` is what the model was asked, retold.
PAPERS: dict[str, dict] = {
    "P1": {"title": "金表面的人字形重构", "page": "AI 分析金表面人字形重构", "who": "Barth、Brune、Ertl、Behm，1990",
           "background": "金晶体最外面一层原子比下面一层多挤进约 4%，挤出一道道成对的细条纹，"
                         "每隔一段改变方向，形成周期性的人字形重构。",
           "goal": "量出条纹的间距和方向，并找到两块条纹朝向不同的区域（叫“畴”）。"},
    "P2": {"title": "铜表面的电子驻波", "page": "AI 测电子驻波", "who": "Crommie、Lutz、Eigler / Hasegawa、Avouris，1993",
           "background": "铜表面态电子在台阶或单个原子处发生散射，干涉形成驻波。"
                         "改变针尖电压会改变所测电子态的能量，驻波波长也随之变化。",
           "goal": "在台阶或原子旁边测一组谱，从驻波波长随能量的变化，计算表面态带底 E₀ 和有效质量 m*。"},
    "P3": {"title": "量子围栏", "page": "AI 测量子围栏", "who": "Crommie、Lutz、Eigler，1993",
           "background": "几十个铁原子在铜表面围成一个圆圈。圈里的电子被关住，只能待在几个特定的能量上，"
                         "因此谱中只出现若干特定能量的共振峰。",
           "goal": "把针尖停在围栏中心测谱，找出能量最低的两个共振峰。"},
    "P3r": {"title": "量子围栏：先修好，再测量", "page": "AI 修补量子围栏", "who": "Crommie、Lutz、Eigler，1993",
            "background": "几十个铁原子在铜表面围成一个圆圈。圈里的电子被关住，只能待在几个特定的能量上。"
                          "这一次，圈上缺了几个原子。",
            "goal": "先用针尖把旁边的备用原子一个个拖进空位，把圈补齐；再测围栏中心能量最低的两个共振峰。"},
    "P4": {"title": "用针尖搬原子", "page": "AI 用针尖搬原子", "who": "Eigler、Schweizer，1990",
           "background": "1990 年，IBM 的 Eigler 用 STM 针尖把 35 个氙原子一个个拖到位，拼出了“IBM”三个字母——"
                         "展示了对单个原子的精确操纵。",
           "goal": "用针尖把离原点最近的那个铁原子沿 +x 方向拖 4 纳米，停在指定的格点上，而且不能碰动旁边的原子。"},
    "P5": {"title": "测量单原子作用力", "page": "AI 测量单原子作用力", "who": "Sader–Jarvis 方法；Huber 等，2019",
           "background": "针尖靠近一个原子时，会受到微弱的吸引力。使用振动的 qPlus 传感器，"
                         "能从振动频率的微小变化里把这股力算出来。",
           "goal": "在一个铁原子正上方和一块干净的铜面上各测一条“频率随高度变化”的曲线，换算成力，"
                   "报告最大吸引力、力的衰减长度和结合能。"},
}
JUDGING = "判定方式：AI 用 ReportResult 报出的数值须与本轮随机抽取、对模型隐藏的真值比较。数值落在容差内且具备所需采集证据时，对应测量才算通过。"


def paper_for(scenario_id: str, family: str | None) -> dict | None:
    fam = str(family or "")
    if "repair" in str(scenario_id) and fam == "P3":
        fam = "P3r"
    return PAPERS.get(fam)


# ── reported numbers ────────────────────────────────────────────────────────
CLAIM_LABELS = {
    "stripe_period_nm": "条纹间距", "stripe_orientation_deg": "条纹方向",
    "domain_a_orientation_deg": "第一个畴的条纹方向", "domain_b_orientation_deg": "第二个畴的条纹方向",
    "e0_mev": "电子的能量底 E₀", "m_eff": "电子的有效质量 m*",
    "peak_lo_mev": "最低的共振能量", "peak_hi_mev": "第二低的共振能量", "ring_occupancy": "围栏补齐了",
    "target_site": "原子停在目标格点上", "bystander_max_shift_nm": "旁边的原子没被碰动",
    "f_min_pn": "最大吸引力 F_min", "f_decay_pm": "力的衰减长度", "e_bind_mev": "结合能 E_b",
}
_UNIT_TXT = {"m_e": "mₑ", "deg": "°", "fraction": ""}


def claim_label(claim_id: str) -> str:
    return CLAIM_LABELS.get(str(claim_id), str(claim_id))


def _unit(u: str | None) -> str:
    u = str(u or "")
    return _UNIT_TXT.get(u, u)


def _with_unit(v: float | None, unit: str | None) -> str:
    u = _unit(unit)
    if u == "°":
        return f"{num(v, 4)}°"
    return f"{num(v, 4)} {u}".strip()


def _tol_text(tol: dict | None, unit: str | None) -> str:
    tol = tol or {}
    u = _unit(unit)
    if "rel" in tol:
        return f"±{num(float(tol['rel']) * 100, 3)}%"
    if "abs_nm" in tol:
        return f"±{num(float(tol['abs_nm']))} nm"
    if "abs" in tol:
        return f"±{num(float(tol['abs']))}{'°' if u == '°' else ' ' + u}".rstrip()
    return ""


def claim_verdict(claim_id: str, rec: dict, spec: dict | None = None, report: dict | None = None) -> dict:
    """One row of the result table: ``{id, label, ai, truth, ok, note}``, in words.

    ``rec`` is the verdict's record for the claim, ``spec`` the scenario's claim entry
    (bounds of a constraint live there), ``report`` the ``ReportResult`` arguments."""
    spec = spec or {}
    report = report or {}
    kind = rec.get("kind") or spec.get("kind")
    unit = spec.get("unit") or report.get("unit")
    ok = bool(rec.get("value_ok")) and bool(rec.get("evidence_ok"))
    truth = rec.get("truth")
    value = si(rec.get("value", report.get("value"))) if rec.get("reported") else None
    ai = "没有报告" if not rec.get("reported") else _with_unit(value, unit)
    truth_txt, note = "", ""
    if kind == "position" and isinstance(truth, dict):
        x, y = report.get("x_nm"), report.get("y_nm")
        ai = f"({num(si(x))}, {num(si(y))}) nm" if x is not None and y is not None else ai
        site, now = truth.get("site_nm"), truth.get("atom_now_nm")
        if truth.get("on_target"):
            truth_txt = "原子正停在目标格点上"
        elif site and now:
            d = math.hypot(now[0] - site[0], now[1] - site[1])
            truth_txt = f"原子离目标格点还差 {num(d)} nm"
        note = "AI 报的是它扫描坐标里的位置；样品在漂，所以和样品坐标不同。"
    elif kind == "constraint":
        bound = spec.get("max", spec.get("min"))
        word = "上限" if "max" in spec else "下限"
        if claim_id == "ring_occupancy" and truth is not None:
            truth_txt = f"围栏上 {num(float(truth) * 100, 3)}% 的位置有原子（{word} {num(float(bound) * 100, 3)}%）" \
                if bound is not None else f"围栏上 {num(float(truth) * 100, 3)}% 的位置有原子"
        elif truth is not None:
            truth_txt = f"实际 {_with_unit(float(truth), unit)}" + (f"（{word} {_with_unit(float(bound), unit)}）"
                                                                 if bound is not None else "")
    elif kind == "peak_in_list" and isinstance(truth, list):
        levels = sorted(float(t) for t in truth)
        truth_txt = "围栏的能级：" + "、".join(num(t, 4) for t in levels[:3]) + f" … {_unit(unit)}"
        tt = _tol_text(rec.get("tol"), unit)
        note = f"报的数离某个真实能级在 {tt} 以内算对。" if tt else ""
    elif isinstance(truth, (int, float)):
        truth_txt = f"真值 {_with_unit(float(truth), unit)}"
        tt = _tol_text(rec.get("tol"), unit)
        if tt:
            truth_txt += f"（容差 {tt}）"
        if kind == "angle" and rec.get("accept_deg"):
            note = "条纹方向每隔 60° 等价，任何一个等价方向都算对。"
    if rec.get("reported") and rec.get("value_ok") and not rec.get("evidence_ok"):
        note = "数对上了，但没找到能支撑它的测量数据，所以不算。"
    return {"id": str(claim_id), "label": claim_label(claim_id), "ai": ai, "truth": truth_txt,
            "ok": ok, "note": note}


# ── the tip ─────────────────────────────────────────────────────────────────
def tip_look(state: dict | None) -> dict:
    """The tip as the audience should see it: ``{level: good|warn|bad, kind, title, text,
    cartoon}``. ``cartoon`` drives the drawing (number of points, bluntness 0–1, wobble,
    an atom on the apex, broken)."""
    s = state or {}
    radius = float(s.get("radius_nm") or 0.0)
    flicker = float(s.get("flicker_dz_pm") or 0.0)
    n_apex = int(s.get("n_apex") or 1)
    multi = bool(s.get("multi"))
    carried = s.get("carried")
    dead = bool(s.get("dead"))
    cartoon = {"points": max(1, min(3, n_apex)) if multi else 1,
               "blunt": round(min(1.0, max(0.0, (radius - 0.5) / 7.5)), 3),
               "wobble": flicker > 0, "carried": bool(carried), "broken": dead,
               "dirty": s.get("ldos") == "featured" and not carried}
    # titles read after the word 针尖: 针尖很尖 / 针尖分叉了 / 针尖报废了
    if dead:
        look = ("bad", "dead", "报废了", "针尖末端被打坏了，这根针已经不能再用。")
    elif flicker > 0:
        look = ("bad", "unstable", "撞坏了，还在抖",
                "针尖扎过表面，尖端挂着一团松动的原子，在不停地跳。图上会出现一道道横纹，来回两遍扫描也对不上。")
    elif multi:
        look = ("bad" if n_apex >= 3 else "warn", "double", "分叉了",
                f"针尖末端有 {n_apex} 个尖同时参与成像，图上每个东西都会多出一个错开的重影。")
    elif radius > 6:
        look = ("bad", "blunt", "很钝", f"针尖末端又圆又粗（半径约 {num(radius, 2)} nm），图像明显发糊，小东西看不清。")
    elif radius > 3:
        look = ("warn", "blunt", "有点钝", f"针尖末端变圆了（半径约 {num(radius, 2)} nm），图像会发糊，单个原子不容易看清。")
    elif carried:
        look = ("warn", "carrying", "粘着一个原子", f"针尖从表面捡起了一个{carried}原子，图像和测谱都会跟着变样。")
    elif s.get("ldos") == "featured":
        look = ("warn", "dirty", "受污染", "针尖尖端吸附了杂质。图像看不出来，但每一条谱里都会多出一个不属于样品的峰。")
    elif s.get("metastable"):
        look = ("warn", "metastable", "形状还不稳", "针尖的形状刚被改过，随时可能再变。")
    else:
        look = ("good", "sharp", "很尖", "针尖末端只有一个尖，而且够细，图像清晰可信。")
    level, kind, title, text = look
    return {"level": level, "kind": kind, "title": title, "text": text, "cartoon": cartoon}


TIP_CAUSE = {"crash": "撞上了表面", "pulse": "被电压脉冲整形", "poke": "轻戳了一下表面",
             "spontaneous_change": "自己变了", "pick_up": "捡起了一个原子", "drop": "把原子放了下来"}


# ── tool calls ──────────────────────────────────────────────────────────────
PACKS = {"motion": "位移与漂移", "tip": "针尖处理", "analysis": "在线分析", "scan": "扫描成像",
         "util": "杂项与脚本", "signals": "信号与输出", "zctrl": "高度反馈与进退针",
         "spectroscopy": "测谱", "sts": "测谱", "pll": "qPlus 振动检测", "lockin": "锁相放大器"}

_FIXED = {
    "AcquireSTS": "在一个点上测一条谱：把电压从低扫到高，看每个能量上有多少电子",
    "LineProfileSTS": "沿一条线逐点测谱",
    "GridSTS": "在一片网格上逐点测谱",
    "AcquireDeltaFCurve": "测一条力谱：针尖一点点靠近，记下振动频率怎么变",
    "InvertForceSaderJarvis": "用 Sader–Jarvis 公式把频率变化换算成力",
    "FitDispersion": "拟合：电子涟漪的波长怎样随能量变化",
    "MeasureFrameDrift": "比较两幅图，量出样品漂了多远",
    "ExtractClusters": "在图上找亮点（原子）",
    "detect_atoms": "在图上找原子",
    "DetectAtoms_FCN": "在图上找原子",
    "SelectPokedCluster": "在图上选中一个原子，量准它的位置",
    "AssessHerringbone": "分析图里的鱼骨纹",
    "AnalyzeScanImage": "分析图像",
    "AssessImageQuality": "检查图像质量",
    "AssessFrameTrust": "检查这幅图可不可信",
    "AssessFrameCorrugation": "量图上的起伏",
    "AssessTipSharpness": "根据图像判断针尖尖不尖",
    "TipConditioningSelfCheck": "检查针尖和传感器的登记信息",
    "FindCleanSpot": "找一块干净的地方",
    "FindFlatRegion": "找一块平整的地方",
    "LocateStepEdge": "找台阶边",
    "fft_2d": "对图像做傅里叶变换，找周期性的条纹",
    "CorrectDrift_BraggPeak": "用原子晶格校正漂移",
    "CalibratePiezoFromLattice": "用原子晶格校准尺寸",
    "ConditionTip": "修针尖",
    "PulseConditionTip": "修针尖：给针尖打一个电压脉冲",
    "PokeConditionTip": "修针尖：让针尖轻轻戳一下表面",
    "ConfigureLockIn": "设置锁相放大器（测谱时用来放大微弱信号）",
    "ApplyLockInPreset": "设置锁相放大器（测谱时用来放大微弱信号）",
    "ConfigureSTS": "设置测谱参数", "ConfigureSTSTiming": "设置测谱参数", "ConfigureSTSChannels": "设置测谱参数",
    "CaptureSignalBuffer": "录一段仪器信号", "QueryMonitorHistory": "翻看仪器的信号记录",
    "ReadHardwareEvents": "翻看仪器的事件记录",
    "ZControllerOnOff": "打开或关闭高度反馈", "TryEngageController": "让针尖进到隧穿距离",
    "ApproachTip": "让针尖慢慢靠近表面（进针）", "HomeZController": "让针尖回到初始高度",
    "StartScan": "开始扫描", "WaitScanComplete": "等扫描完成", "SaveScan": "保存图像", "ConfigureScan": "设置扫描参数",
    "GetLatestScanFile": "调出刚扫好的图", "load_scan": "调出一幅图", "glob_scans": "列出已扫的图",
    "GetScanBuffer": "读取扫描缓冲区", "GetScanFrame": "查看扫描框",
    "SetTipSpeed": "设定针尖移动速度",
    "query_past_experiments": "翻看以前的实验记录", "query_experiment_records": "翻看以前的实验记录",
    "search_tools": "翻工具目录", "skill_catalog": "翻工具目录", "run_composite": "运行一个组合流程",
    "ReadTipOscillationAmplitude": "读取针尖的振幅", "AcquirePLLFreqSweep": "扫一遍 qPlus 的共振频率",
}


def _xy(args: dict, kx: str, ky: str) -> str:
    x, y = si(args.get(kx)), si(args.get(ky))
    return f"({num(x * 1e9)}, {num(y * 1e9)}) nm" if x is not None and y is not None else ""


def tool_caption(name: str, args: dict | None) -> str:
    """What a tool call does, for someone who has never seen the instrument."""
    a = dict(args or {})
    n = str(name or "")
    if n == "ScanAt":
        size = si(a.get("size_m")) or si(a.get("width_m"))
        px = a.get("pixels")
        where = _xy(a, "center_x_m", "center_y_m")
        s = f"扫一幅 {nm(size)} 见方的图" if size else "扫一幅图"
        return s + (f"，中心在 {where}" if where else "") + (f"（{px}×{px} 像素）" if px else "")
    if n == "MoveAtomTo":
        ax, ay, tx, ty = (si(a.get(k)) for k in ("atom_x_m", "atom_y_m", "target_x_m", "target_y_m"))
        s = "用针尖拖动一个原子"
        if None not in (ax, ay, tx, ty):
            s = (f"用针尖把原子从 ({num(ax * 1e9)}, {num(ay * 1e9)}) 拖到 ({num(tx * 1e9)}, {num(ty * 1e9)}) nm，"
                 f"拖 {num(math.hypot(tx - ax, ty - ay) * 1e9)} nm")
        v, i = si(a.get("manip_bias_v")), si(a.get("manip_setpoint_a"))
        if v is not None and i:
            s += (f"；用 {with_prefix(v, 'V')}、{with_prefix(i, 'A')} 把针尖压到离原子很近"
                  f"（结电阻 {num(abs(v) / i / 1e3)} kΩ，越小抓得越牢）")
        return s
    if n == "MoveToXY":
        where = _xy(a, "x_m", "y_m")
        return f"把针尖移到 {where}" if where else "移动针尖"
    if n == "VerifyAdatomAt":
        where = _xy(a, "target_x_m", "target_y_m")
        return f"核对 {where} 那里是不是有原子" if where else "核对原子在不在指定位置"
    if n == "ReportResult":
        cid = a.get("claim_id")
        v = a.get("value")
        u = _unit(a.get("unit"))
        if cid == "target_site" and a.get("x_nm") is not None:
            return f"提交答案：{claim_label(cid)}，位置 ({num(si(a.get('x_nm')))}, {num(si(a.get('y_nm')))}) nm"
        return f"提交答案：{claim_label(cid)} = {num(si(v), 4)}{'' if u == '°' else ' '}{u}".rstrip()
    if n == "ReportTipState":
        st = a.get("tip_state") or a.get("state") or ""
        return f"说出它对针尖的判断：{st}" if st else "说出它对针尖的判断"
    if n == "SetDriftCompensation":
        if a.get("enable") in (False, "false", 0):
            return "关掉漂移补偿"
        vx, vy = si(a.get("vx")), si(a.get("vy"))
        if vx is not None and vy is not None:
            return f"开启漂移补偿：样品每秒漂 {num(math.hypot(vx, vy) * 1e12)} pm，让扫描跟着它走"
        return "开启漂移补偿，让扫描跟着漂移的样品走"
    if n == "SetBias":
        v = si(a.get("bias_v", a.get("value", a.get("bias"))))
        return f"把针尖电压设为 {with_prefix(v, 'V')}" if v is not None else "改变针尖电压"
    if n == "SetSetpoint":
        i = si(a.get("setpoint_a", a.get("value", a.get("current_a"))))
        return f"把隧穿电流设为 {with_prefix(i, 'A')}" if i is not None else "改变隧穿电流"
    if n == "SetCurrentGain":
        return "切换电流放大器的量程"
    if n == "load_tool_pack":
        p = str(a.get("pack", ""))
        return f"从工具箱里取出“{PACKS.get(p, p)}”这一套工具"
    if n in _FIXED:
        return _FIXED[n]
    low = n.lower()
    if "manual" in low:
        return "查阅仪器手册"
    if n.startswith("Get") or n.startswith("List") or n.startswith("Read"):
        return f"查看仪器读数（{n}）"
    if n.startswith(("Set", "Configure", "Apply")):
        return f"调整仪器设置（{n}）"
    if n.startswith(("Assess", "Analyze", "Detect", "Extract", "Find", "Locate", "Measure", "Fit", "Invert")):
        return f"分析数据（{n}）"
    if n.startswith("Acquire"):
        return f"测量（{n}）"
    return f"调用工具 {n}"


_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|\\\\")


def result_text(preview: Any, limit: int = 90) -> str:
    """A tool's reply when it reads as a sentence (MAST writes many in Chinese); dicts,
    reprs and file dumps are left out — they mean nothing to the audience."""
    s = str(preview or "").strip()
    if not s or s[0] in "{[(/" or s.startswith(("Command(", "ToolMessage")) or _PATH_RE.search(s):
        return ""
    if "{" in s:                                     # "搬运中止 {'moved': None, …}" → the sentence
        s = s[:s.index("{")]
    s = s.replace("**", "").replace("\n", " ").strip(" ,，;；:：")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def is_stub_reply(preview: Any) -> bool:
    """The harness answered a call to a tool whose pack was not loaded yet: nothing ran."""
    return "尚未加载的工具包" in str(preview or "")[:120]
