"""The scoring rules an agent is shown: generated from the scenario file, never hand-written.

Everything the judge checks for a paper scenario - what to report, in which unit and
coordinate convention, the tolerance, the acquisition evidence, the instrument budget and the
overrun rule - is stated in the task text by :func:`task_text`, from the same YAML fields the
judge reads. A requirement that is not on the rule card is not a requirement. (Before the card,
P4's verification frame was a judge rule the task never mentioned, and P1's task promised a
2 h budget while the overrun rule used the file's 3 h.)

:data:`RULES_VERSION` is recorded in every ledger the card was shown in and is passed to the
judge, which applies the version-1 rules to such episodes (:mod:`stmbench.trackB.claims`).

The card is Chinese, like the task text it is appended to (AGENTS.md: Chinese belongs in
scenario task text).
"""
from __future__ import annotations

import re

#: version of the rule card and of the judging rules it announces
RULES_VERSION = 1

#: the overrun rule (stmbench.harness.episode.apply_budget_overrun): beyond this factor of the
#: scenario's budget an episode is not a reproduction
OVERRUN_TOLERANCE = 1.05

#: a hand-written budget sentence in a task, which the card now states instead
_BUDGET_SENTENCE = re.compile(r"预算[:：][^。\n]*仪器时间[。]?")

_UNIT = {"nm": "nm", "deg": "°", "meV": "meV", "m_e": "m_e", "pN": "pN", "pm": "pm", "fraction": ""}

_AFTER = {"adatom_hop": "最后一次原子跳动（拖动原子、或在低结电阻下成像带动原子）"}


def _num(x) -> str:
    return f"{float(x):g}"


def _unit(c: dict) -> str:
    return _UNIT.get(str(c.get("unit", "")), str(c.get("unit", "")))


def _tolerance(c: dict, sc) -> str:
    kind, tol, unit = c.get("kind"), c.get("tol") or {}, _unit(c)
    if kind == "angle":
        out = f"与真值相差不超过 {_num(tol.get('abs', 5))}°（按模 {_num(tol.get('mod', 180))}° 计）"
        if c.get("truth") == "orientation_at":
            out += "；鱼骨条纹呈锯齿形，两条臂中任一条的方向、或两臂的平均方向，都算对"
        return out
    if kind == "peak_in_list":
        out = (f"与围栏中能量最低的 {int(c.get('rank_max', 3))} 个真实共振能级中的任一个"
               f"相差不超过 {_num(tol.get('abs', 0))} {unit}")
        return out
    if kind == "position":
        ini = (sc.initial or {}).get("adatoms") or {}
        d = (float(ini.get("target_dx_nm", 0.0)), float(ini.get("target_dy_nm", 0.0)))
        out = (f"所报位置须在某个被你搬动过的原子 {_num(tol.get('report_nm', 0.5))} nm 以内（按 frame 指定的帧换算到样品上）；"
               f"这个原子出生时 {_num(ini.get('isolation_nm', 3.0))} nm 内没有别的原子，"
               f"并且现在正好占据'出生位置平移 ({_num(d[0])}, {_num(d[1])}) nm 后最近的那个晶格位点'")
        tie = float(tol.get("site_tie_nm", 0.0) or 0.0)
        if tie > 0:
            out += f"（若另有格点离那一点只比最近的远不到 {_num(tie)} nm，占据它也算）"
        return out
    if kind == "constraint":
        lo, hi = c.get("min"), c.get("max")
        bound = (f"不小于 {_num(lo)}" if lo is not None else "") + \
                ("、" if lo is not None and hi is not None else "") + \
                (f"不大于 {_num(hi)}" if hi is not None else "")
        return f"这个量{bound}{(' ' + unit) if unit else ''}"
    abs_tol, rel_tol = tol.get("abs"), tol.get("rel")
    if abs_tol is not None and rel_tol is not None:
        return f"与真值相差不超过 {_num(abs_tol)} {unit}或真值的 {_num(100 * float(rel_tol))}%（取较大者）"
    if rel_tol is not None:
        return f"相对误差不超过 {_num(100 * float(rel_tol))}%"
    return f"与真值相差不超过 {_num(abs_tol or 0)} {unit}"


def _frame_terms(rule: dict) -> str:
    terms = ["在隧穿状态下开着 Z 反馈扫描（恒流形貌；退针或关着反馈扫的不算）", "完整（扫完全部行）"]
    if rule.get("min_fov_nm") is not None:
        terms.append(f"视场（宽、高中较小者）不小于 {_num(rule['min_fov_nm'])} nm")
    if rule.get("max_nm_per_px") is not None:
        terms.append(f"像素尺寸（宽度 ÷ 列数）不大于 {_num(rule['max_nm_per_px'])} nm")
    return "、".join(terms)


def _evidence(c: dict) -> str:
    rule = c.get("evidence") or {}
    kind = rule.get("kind", "frame")
    after = rule.get("after_event")
    after_txt = (f"，并且在{_AFTER.get(after, after)}之后才开始扫描（看开始扫描的时间，不看保存的时间）"
                 if after else "")
    if kind == "frame":
        return f"evidence 中至少引用一张扫描帧，须{_frame_terms(rule)}"
    if kind == "frame_covers":
        return f"所报位置须落在 frame 指定的那张帧里，该帧须{_frame_terms(rule)}"
    if kind == "frame_covers_after":
        if isinstance(rule.get("point"), str):
            return (f"（不需要引用）{_AFTER.get(after, after) if after else '本局开始'}之后，本局至少新扫并保存过一张覆盖围栏中心的帧，"
                    f"须{_frame_terms(rule)}")
        return f"所报位置须落在 frame 指定的那张帧里，该帧须{_frame_terms(rule)}{after_txt}"
    if kind == "spectra_near_scatterer":
        return (f"evidence 中至少引用 {int(rule.get('min_n', 1))} 条 dI/dV 谱，分布在至少 "
                f"{int(rule.get('min_positions', 1))} 个不同位置（位置按 0.5 nm 网格区分）；每条谱离最近的散射体"
                f"（台阶边或吸附原子）不超过 {_num(rule.get('near_nm', 8.0))} nm、至少 {int(rule.get('min_points', 0))} 个点、"
                f"偏压范围覆盖 {_num(rule.get('v_lo_max'))} V 到 {_num(rule.get('v_hi_min'))} V，谱中途没有突变（针尖不稳造成的跳变）")
    if kind == "sts_at_corral_centre":
        return (f"evidence 中至少引用 {int(rule.get('min_n', 1))} 条 dI/dV 谱：取谱处离围栏中心不超过 "
                f"{_num(rule.get('max_centre_dist_nm', 1.0))} nm、至少 {int(rule.get('min_points', 0))} 个点、"
                f"偏压范围覆盖 {_num(rule.get('v_lo_max'))} V 到 {_num(rule.get('v_hi_min'))} V、中途没有突变"
                f"{('，并且在' + _AFTER.get(after, after) + '之后采集') if after else ''}")
    if kind == "zspec_pair":
        a, b = rule.get("adatom") or {}, rule.get("background") or {}
        return (f"evidence 中至少引用 {int(a.get('min_n', 1))} 条在吸附原子正上方（横向偏离不超过 "
                f"{_num(a.get('near_nm_max', 0.15))} nm）测的 Δf(z) 曲线，越过 Δf 极小值后至少再逼近 "
                f"{_num(a.get('df_span_beyond_min_pm_min', 0))} pm，中途没有跳变、没有接触；以及至少 "
                f"{int(b.get('min_n', 1))} 条在离吸附原子 {_num(b.get('near_nm_min', 2.0))} nm 以外的干净表面上测的曲线（没有接触）")
    if kind == "didv_maps":
        return (f"evidence 中至少引用 {int(rule.get('min_n', 1))} 张开着锁相（lockin）的恒流扫描图（dI/dV 图），"
                f"偏压至少 {int(rule.get('min_energies', 1))} 个不同的值（按 10 mV 区分），覆盖 {_num(rule.get('v_lo_max'))} V "
                f"到 {_num(rule.get('v_hi_min'))} V；每张图的中心离最近的散射体（台阶边或吸附原子）不超过 "
                f"{_num(rule.get('near_nm', 10.0))} nm，图须{_frame_terms(rule)}")
    if kind == "any_of":
        alts = [_evidence({"evidence": r}) for r in rule.get("rules") or []]
        return "以下任一组即可——" + "；或者，".join(f"（{chr(0x61 + i)}）{a}" for i, a in enumerate(alts))
    if kind == "events":
        what = "由针尖拖动引起的原子跳动" if rule.get("event") == "adatom_hop" and rule.get("cause") == "folme" \
            else str(rule.get("event"))
        return f"（不需要引用）本局至少发生过 {int(rule.get('min_n', 1))} 次{what}"
    return f"（证据类型 {kind}）"


def _claim_lines(c: dict, sc, n: int) -> list[str]:
    cid, unit = str(c["id"]), _unit(c)
    say = str(c.get("say") or "").strip()
    if c.get("kind") == "constraint":
        head = f"{n}. {cid}（不需要报告）：{say}"
    else:
        head = f"{n}. {cid}：{say}" + (f"，单位 {unit}" if unit and c.get("kind") != "position" else "")
    lines = [head, f"   - 判对条件：{_tolerance(c, sc)}"]
    other = c.get("distinct_from")
    if other:
        if c.get("kind") == "angle":
            lines.append(f"   - 与 {other.get('claim')} 两处的真实条纹方向须相差至少 "
                         f"{_num(other.get('min_sep_deg', other.get('min_sep', 0)))}°（同一个畴的两条臂不算两个畴）")
        else:
            lines.append(f"   - 须与 {other.get('claim')} 对应不同的真实能级")
    if c.get("greater_than"):
        lines.append(f"   - 数值须大于 {c['greater_than']}")
    if c.get("of_claim"):
        lines.append(f"   - 针对的是 {c['of_claim']} 所匹配到的那个原子")
    lines.append(f"   - 证据：{_evidence(c)}")
    return lines


def rule_card(sc) -> str:
    """The rule card of a paper scenario (Chinese), generated from its claims and budget."""
    lines = [f"【评分规则 v{RULES_VERSION}】以下规则由场景文件自动生成，判分只按这些规则进行。",
             "", "要报告的结果（每条用一次 ReportResult；可以重复报告，以最后一次为准；没有报告的结果算未复现）："]
    n = 0
    for c in sc.claims:
        n += 1
        lines += _claim_lines(c, sc, n)
    lines += ["",
              "引用数据：每条报告用 evidence 列出你用到的扫描帧（.sxm 文件名）和谱（.dat 文件名），也可以写 frame:N、"
              "sts:N、zspec:N。判分只在你引用的数据里检查证据要求，没有引用的数据不算。",
              "位置与角度：位置用扫描坐标（nm），x 向右、y 向上，与扫描框的中心坐标是同一套坐标；位置类结果还要用 "
              "frame 写明坐标取自哪一张帧，判分按这张帧逐行记录的漂移把坐标换算到样品上，你不需要自己修正漂移。"
              "角度在同一坐标系里从 +x 轴逆时针量；图像数组的行号向下增大，在像素里量角度时要把符号取反。"]
    budget = sc.budget or {}
    hours = budget.get("sim_hours")
    if hours:
        txt = (f"仪器时间：本题预算 {_num(hours)} 小时仪器时间（按仿真时间计，模型思考时仪器暂停）。用量超过预算的 "
               f"{_num(round(100 * (OVERRUN_TOLERANCE - 1), 6))}% 以上，本次复现不算成功。")
        if budget.get("wire_cmds"):
            txt += f"控制器命令上限 {int(budget['wire_cmds'])} 条。"
        lines.append(txt)
    return "\n".join(lines)


def task_text(sc) -> str:
    """What the agent is told: the scenario's task, then (paper scenarios) its rule card."""
    task = str(sc.task or "").strip()
    if not sc.paper:
        return task
    return f"{task}\n\n{rule_card(sc)}"


def has_budget_sentence(task: str) -> bool:
    """A budget stated by hand in a task (the rule card states it from the file instead)."""
    return bool(_BUDGET_SENTENCE.search(str(task or "")))
