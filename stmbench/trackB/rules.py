"""The scoring rules an agent is shown: generated from the scenario file, never hand-written.

Everything the judge checks for a paper scenario - what to report, in which unit and
coordinate convention, the tolerance, the acquisition evidence, the instrument budget and the
overrun rule - is stated in the task text by :func:`task_text`, from the same YAML fields the
judge reads. A requirement that is not on the rule card is not a requirement. (Before the card,
P4's verification frame was a judge rule the task never mentioned, and P1's task promised a
2 h budget while the overrun rule used the file's 3 h.)

:data:`RULES_VERSION` is recorded in every ledger the card was shown in and is passed to the
judge, which applies the version-1 rules to such episodes (:mod:`stmbench.trackB.claims`).
The grader refuses an export of another version (:mod:`stmbench.instrument.grade`).

The card is written in Chinese (``lang="zh"``, the default: Chinese belongs in scenario task
text, AGENTS.md) or in English (``lang="en"``). One code path states each rule and only its words
differ: every phrase is written for both languages at the same place (:func:`_t`), from the same
values, so the English says what the Chinese says and a rule cannot be added to one language and
not the other. The scenario's own words are fields of its file: ``task`` and each claim's ``say``
in Chinese, ``task_en`` and ``say_en`` in English. They are not hidden draws and feed no random
stream.
"""
from __future__ import annotations

import re

#: version of the rule card and of the judging rules it announces.
#: 1: the card; a report names its ``frame`` and ``evidence`` (the judge takes ``rules_version >= 1``
#: as these rules, so an older ledger keeps the rules it ran under).
#: 2: the judge is unchanged. P4's card says what the judge does and nothing about how to operate: an
#: atom hop is not explained by its causes, a tied lattice site counts "by at most" ``site_tie_nm``, the
#: bystanders and their displacement are defined as ``AdatomRegistry.any_target`` computes them, and the
#: hop the evidence rule needs is one made during a ``move_tip`` action. No card states the controller
#: command limit of the TCP path (``budget.wire_cmds``) any more: the benchmark instrument does not
#: enforce it.
#: A new paper's rules come with its own card and do not change the version (no earlier ledger of that
#: paper exists): P7's (``drag_series`` evidence, ``distinct_atom_from``, the true value of a pull
#: threshold, a move trace cited by ``move_tip``'s file name or ``move:N``) are stated in P7's card only,
#: and every other paper's card and judging are what version 2 says. A version stamps every ledger and
#: export, and ``grade`` refuses another one, so a change that leaves every other card as it was does
#: not bump it.
RULES_VERSION = 2


def _cites_moves(sc) -> bool:
    """Does any claim of ``sc`` rest on cited move traces (a drag series)? Only then does the card's
    general paragraph on citing data say how a move trace is named."""
    return any(str((c.get("evidence") or {}).get("kind")) == "drag_series" for c in sc.claims)

#: the overrun rule (:func:`apply_budget_overrun`): beyond this factor of the scenario's budget
#: an episode is not a reproduction (5 % over is still within budget)
OVERRUN_TOLERANCE = 1.05

#: the partial score an over-budget episode keeps at most
OVERRUN_PARTIAL_CAP = 0.5

#: the languages the card is written in; the first is the default
LANGS = ("zh", "en")

#: a hand-written budget sentence in a task, which the card now states instead
_BUDGET_SENTENCE = re.compile(r"预算[:：][^。\n]*仪器时间[。]?")

_UNIT = {"nm": "nm", "deg": "°", "meV": "meV", "m_e": "m_e", "pN": "pN", "pm": "pm", "fraction": ""}

#: what an ``after_event`` is called on the card. The judge takes the last ``adatom_hop`` event, one
#: lattice-site step of an atom whatever made it (a tip move or imaging), so the card says what a hop
#: is and leaves the causes, which are operating knowledge, out.
_AFTER = {
    "zh": {"adatom_hop": "最后一次原子跳动（指任何原子从一个晶格位点跳到相邻位点，不论是什么使它动的）"},
    "en": {"adatom_hop": "the last atom hop (the last time any atom hopped from one lattice site to a "
                         "neighbouring one, whatever moved it)"},
}


def _check_lang(lang) -> str:
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r} (known: {', '.join(LANGS)})")
    return lang


def _t(lang: str, zh: str, en: str) -> str:
    """One phrase in ``lang``: its Chinese and English wordings sit side by side, so a reader can
    check that they say the same, and both are built from the same values."""
    return zh if lang == "zh" else en


def _num(x) -> str:
    return f"{float(x):g}"


def _amount(x, unit: str) -> str:
    """``0.15 nm``; just the number for a quantity without a unit (the English wording)."""
    return f"{_num(x)} {unit}" if unit else _num(x)


def _count(n: int, one: str, many: str) -> str:
    """``1 spectrum``, ``4 spectra``: an English noun agrees with its number."""
    return f"{n} {one if n == 1 else many}"


def _and(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b, and c``."""
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]


def _unit(c: dict) -> str:
    return _UNIT.get(str(c.get("unit", "")), str(c.get("unit", "")))


def _atom_numbers(sc) -> tuple[str, str, str]:
    """The shift an atom is to be moved by, as ``(dx, dy)``, the isolation radius and the bystander
    radius (nm) of a single-atom layout, as the card writes them. They are read where the world is built
    from (``initial.adatoms``, defaulted as ``Surface.configure_adatoms`` defaults them), so the card
    cannot state another figure than the one ``AdatomRegistry.any_target`` judges with."""
    ini = (sc.initial or {}).get("adatoms") or {}
    shift = f"({_num(ini.get('target_dx_nm', 4.0))}, {_num(ini.get('target_dy_nm', 0.0))})"
    return shift, _num(ini.get("isolation_nm", 3.0)), _num(ini.get("bystander_nm", 10.0))


def _tolerance(c: dict, sc, lang: str) -> str:
    kind, tol, unit = c.get("kind"), c.get("tol") or {}, _unit(c)
    if kind == "angle":
        abs_deg, mod_deg = _num(tol.get("abs", 5)), _num(tol.get("mod", 180))
        out = _t(lang, f"与真值相差不超过 {abs_deg}°（按模 {mod_deg}° 计）",
                 f"the reported value differs from the true value by at most {abs_deg}° "
                 f"(angles are taken modulo {mod_deg}°)")
        if c.get("truth") == "orientation_at":
            out += _t(lang, "；鱼骨条纹呈锯齿形，两条臂中任一条的方向、或两臂的平均方向，都算对",
                      "; the herringbone stripes are zigzag-shaped, so the direction of either of the two "
                      "arms, or the mean direction of the two arms, is accepted as the true direction")
        return out
    if kind == "peak_in_list":
        rank, abs_tol = int(c.get("rank_max", 3)), tol.get("abs", 0)
        return _t(lang, f"与围栏中能量最低的 {rank} 个真实共振能级中的任一个相差不超过 {_num(abs_tol)} {unit}",
                  f"the reported value differs by at most {_amount(abs_tol, unit)} from at least one of the "
                  f"{rank} lowest-energy true resonance levels of the corral")
    if kind == "position":
        shift, isolation_nm, _ = _atom_numbers(sc)
        report_nm = _num(tol.get("report_nm", 0.5))
        # the judge matches the report to the moved atom nearest to the reported place, and only then
        # asks whether that one is within report_nm (AdatomRegistry.any_target, claims_verified)
        out = _t(lang,
                 f"所报位置须在离它最近的、被你搬动过的原子 {report_nm} nm 以内（按 frame 指定的帧换算到样品上）；"
                 f"这个原子本局开始时 {isolation_nm} nm 内没有别的原子，"
                 f"并且现在正好占据'它本局开始时的位置平移 {shift} nm 后最近的那个晶格位点'",
                 f"the reported position, converted to the sample using the frame named in the frame field, "
                 f"must be within {report_nm} nm of the atom you have moved that lies nearest to it; at the "
                 f"start of the episode that atom had no other atom within {isolation_nm} nm, and it now sits "
                 f"exactly on 'the lattice site nearest to the point obtained by shifting its position at the "
                 f"start of the episode by {shift} nm'")
        tie = float(tol.get("site_tie_nm", 0.0) or 0.0)
        if tie > 0:
            # the judge accepts a site farther by <= site_tie_nm (claims_verified), so "at most"
            out += _t(lang, f"（若另有格点离那一点比最近的格点至多远 {_num(tie)} nm，占据它也算）",
                      f" (if another lattice site is farther from that point than the nearest one by at most "
                      f"{_num(tie)} nm, sitting on it counts too)")
        return out
    if kind == "constraint":
        lo, hi = c.get("min"), c.get("max")
        bound = (f"不小于 {_num(lo)}" if lo is not None else "") + \
                ("、" if lo is not None and hi is not None else "") + \
                (f"不大于 {_num(hi)}" if hi is not None else "")
        bounds = ([f"at least {_amount(lo, unit)}"] if lo is not None else []) + \
                 ([f"at most {_amount(hi, unit)}"] if hi is not None else [])
        return _t(lang, f"这个量{bound}{(' ' + unit) if unit else ''}",
                  f"this quantity is {' and '.join(bounds)}")
    abs_tol, rel_tol = tol.get("abs"), tol.get("rel")
    if abs_tol is not None and rel_tol is not None:
        percent = _num(100 * float(rel_tol))
        return _t(lang, f"与真值相差不超过 {_num(abs_tol)} {unit}或真值的 {percent}%（取较大者）",
                  f"the reported value differs from the true value by at most {_amount(abs_tol, unit)} or "
                  f"{percent}% of the true value (whichever is larger)")
    if rel_tol is not None:
        percent = _num(100 * float(rel_tol))
        return _t(lang, f"相对误差不超过 {percent}%", f"the relative error is at most {percent}%")
    return _t(lang, f"与真值相差不超过 {_num(abs_tol or 0)} {unit}",
              f"the reported value differs from the true value by at most {_amount(abs_tol or 0, unit)}")


def _frame_terms(rule: dict, lang: str, after: str | None = None) -> str:
    """What a frame must be, as the list that follows "the frame must". ``after`` (an
    ``after_event`` as the card names it) adds that the scan began after that event."""
    terms = [_t(lang, "在隧穿状态下开着 Z 反馈扫描（恒流形貌；退针或关着反馈扫的不算）",
                "have been scanned in the tunnelling state with the Z feedback on (constant-current "
                "topograph; a scan made with the tip withdrawn or the feedback off does not count)"),
             _t(lang, "完整（扫完全部行）", "be complete (all rows scanned)")]
    if rule.get("min_fov_nm") is not None:
        fov = _num(rule["min_fov_nm"])
        terms.append(_t(lang, f"视场（宽、高中较小者）不小于 {fov} nm",
                        f"have a field of view (the smaller of width and height) of at least {fov} nm"))
    if rule.get("max_nm_per_px") is not None:
        px = _num(rule["max_nm_per_px"])
        terms.append(_t(lang, f"像素尺寸（宽度 ÷ 列数）不大于 {px} nm",
                        f"have a pixel size (width ÷ number of columns) of at most {px} nm"))
    if lang == "zh":
        return "、".join(terms) + (f"，并且在{after}之后才开始扫描（看开始扫描的时间，不看保存的时间）"
                                  if after else "")
    if after:
        terms.append(f"have started scanning only after {after} — what counts is when the scan started, "
                     f"not when it was saved")
    return _and(terms)


def _drag_series(c: dict, lang: str, sc) -> str:
    """The evidence rule of a drag series (``claims.select_drags`` and ``claims._check_drag_series``),
    with every number it reads. The site spacing is the material's (a constant, so a rule can state it)."""
    from stmsim.physics.drag_threshold import WITH_TIP_FRACTION
    from stmsim.physics.surface import MATERIALS

    from .claims import DISTINCT_RESISTANCE_FACTOR

    rule, proto = c.get("evidence") or {}, c.get("protocol") or {}
    length, speed = _num(proto.get("length_nm", 1.0)), _num(proto.get("speed_nm_s", 0.1))
    rel = _num(100 * float(rule.get("protocol_rel", 0.0) or 0.0))
    within, n, res = _num(rule.get("start_within_nm", 0.0)), int(rule.get("min_n", 1)), int(rule.get("min_resistances", 1))
    distinct = _num(round(100 * (DISTINCT_RESISTANCE_FACTOR - 1), 6))
    material = MATERIALS.get(str(getattr(sc, "material", "") or ""))
    spacing = f"{material.nn_m * 1e9:g}" if material else None
    frac = _num(WITH_TIP_FRACTION)
    reach = f"{WITH_TIP_FRACTION * material.nn_m * 1e9:g}" if material else None
    zh_spacing = f"（{material.name} 上的格点间距为 {spacing} nm，所以是 {reach} nm）" if material else ""
    en_spacing = (f" (on {material.name} the site spacing is {spacing} nm, so {reach} nm)") if material else ""
    return _t(lang,
              f"evidence 中引用这个结果所依据的拖动：move_tip 返回的轨迹文件名（.dat），或 move:N；引用的不是移动轨迹的数据在这条规则里不计。"
              f"被引用的一次移动要计入，须同时满足：Z 反馈开着、针尖在隧穿状态（没有退针）、没有超过前置放大器量程、"
              f"移动开始前针尖上没有原子；设定的结电阻 |V|/I（面板上的偏压除以电流设定值）是有限值；"
              f"移动开始时针尖的某个顶点抓住了一个原子，该顶点离这个原子不超过 {within} nm（从抓住这个原子的那个顶点量起）；"
              f"移动长度与 {length} nm 的相对误差不超过 {rel}%，速度与 {speed} nm/s 的相对误差不超过 {rel}%。"
              f"计入的移动须满足：抓住的全是同一个原子；至少 {n} 次；结电阻至少有 {res} 个不同的值"
              f"（把结电阻从小到大排，比当前这一组的第一个值大出 {distinct}% 以上才算新的一个值）；"
              f"你报告的数值被它们夹住：在不高于该数值的结电阻下，至少有一次移动把原子带到了终点，"
              f"并且在不低于该数值的结电阻下，至少有一次没有（原子被留在原处或被提到针尖上，都算没有）。"
              f"“把原子带到终点”指移动结束时，原子停在抓住它的那个顶点的终点处，"
              f"离终点不超过 {frac} 个格点间距{zh_spacing}",
              f"cite in the evidence field the moves this result rests on: the trace file names (.dat) that move_tip "
              f"returned, or move:N; cited data that is not a move trace does not count for this rule. A cited move "
              f"counts only when all of the following hold: the Z feedback was on, the tip was in the tunnelling state "
              f"(not withdrawn), the preamplifier range was not exceeded, and no atom was on the tip before the move; "
              f"the junction resistance |V|/I set (the bias voltage divided by the current setpoint on the panel) was "
              f"finite; an apex of the tip held an atom at the start of the move, within {within} nm of that atom "
              f"(measured from the apex that held the atom); and the length of the move is within {rel}% of {length} nm "
              f"and its speed within {rel}% of {speed} nm/s. The counted moves must all have held the same atom; there "
              f"must be at least {n} of them; their resistances must take at least {res} distinct values (sort the "
              f"resistances; a value is a new one when it exceeds the first value of the current group by more than "
              f"{distinct}%); and the reported value must be bracketed by them: at least one counted move that took the "
              f"atom along to the end point at a resistance at or below the reported value, and at least one that did "
              f"not at a resistance at or above it (an atom left where it was, or lifted onto the tip, counts as not "
              f"taken along). An atom is taken along to the end point when, at the end of the move, it rests at the "
              f"end point of the apex that held it, within {frac} site spacings of it{en_spacing}")


def _drag_truth(c: dict, lang: str) -> str:
    """What a ``drag_threshold`` value is compared with."""
    return _t(lang,
              "该原子在计入的那些移动各自的条件下的拉动阈值（定义见上）；如果各次移动的条件不同"
              "（例如中途针尖变了），取各次移动自己条件下的拉动阈值的几何平均",
              "the pull threshold of that atom (defined above) under the conditions of the counted moves; if "
              "the moves were made under different conditions (for example the tip changed in between), the "
              "geometric mean of the pull thresholds under each move's own conditions")


def _evidence(c: dict, lang: str, sc=None) -> str:
    rule = c.get("evidence") or {}
    kind = rule.get("kind", "frame")
    after = rule.get("after_event")
    since = _AFTER[lang].get(after, after) if after else None
    inside = _t(lang, "所报位置须落在 frame 指定的那张帧里，该帧须",
                "the reported position must fall inside the frame named in the frame field; that frame must ")
    if kind == "frame":
        return _t(lang, f"evidence 中至少引用一张扫描帧，须{_frame_terms(rule, lang)}",
                  f"cite at least one scan frame in the evidence field; that frame must "
                  f"{_frame_terms(rule, lang)}")
    if kind == "frame_covers":
        return f"{inside}{_frame_terms(rule, lang)}"
    if kind == "frame_covers_after":
        if isinstance(rule.get("point"), str):
            terms = _frame_terms(rule, lang)
            return _t(lang, f"（不需要引用）{since or '本局开始'}之后，本局至少新扫并保存过一张覆盖围栏中心的帧，"
                            f"须{terms}",
                      f"(no citation needed) after {since or 'the start of the episode'}, this episode has "
                      f"scanned and saved at least one new frame covering the corral centre; that frame "
                      f"must {terms}")
        return f"{inside}{_frame_terms(rule, lang, since)}"
    if kind == "spectra_near_scatterer":
        n, positions = int(rule.get("min_n", 1)), int(rule.get("min_positions", 1))
        near, points = _num(rule.get("near_nm", 8.0)), int(rule.get("min_points", 0))
        v_lo, v_hi = _num(rule.get("v_lo_max")), _num(rule.get("v_hi_min"))
        return _t(lang,
                  f"evidence 中至少引用 {n} 条 dI/dV 谱，分布在至少 {positions} 个不同位置"
                  f"（位置按 0.5 nm 网格区分）；每条谱离最近的散射体"
                  f"（台阶边或吸附原子）不超过 {near} nm、至少 {points} 个点、"
                  f"偏压范围覆盖 {v_lo} V 到 {v_hi} V，谱中途没有突变（针尖不稳造成的跳变）",
                  f"cite at least {_count(n, 'dI/dV spectrum', 'dI/dV spectra')} in the evidence field, "
                  f"spread over at least {_count(positions, 'different position', 'different positions')} "
                  f"(positions are binned on a 0.5 nm grid); each spectrum is within {near} nm of "
                  f"the nearest scatterer (step edge or adatom), has at least "
                  f"{_count(points, 'point', 'points')} and a bias range covering {v_lo} V to {v_hi} V, and "
                  f"shows no sudden jump along the sweep (a jump caused by an unstable tip)")
    if kind == "sts_at_corral_centre":
        n, points = int(rule.get("min_n", 1)), int(rule.get("min_points", 0))
        centre, v_lo, v_hi = _num(rule.get("max_centre_dist_nm", 1.0)), _num(rule.get("v_lo_max")), \
            _num(rule.get("v_hi_min"))
        late = _t(lang, f"，并且在{since}之后采集" if after else "", f"acquired after {since}" if after else "")
        spectra = _and([f"taken within {centre} nm of the corral centre",
                        f"having at least {_count(points, 'point', 'points')}",
                        f"covering a bias range of {v_lo} V to {v_hi} V", "showing no sudden jump along the sweep"]
                       + ([late] if after else []))
        return _t(lang,
                  f"evidence 中至少引用 {n} 条 dI/dV 谱：取谱处离围栏中心不超过 "
                  f"{centre} nm、至少 {points} 个点、"
                  f"偏压范围覆盖 {v_lo} V 到 {v_hi} V、中途没有突变{late}",
                  f"cite at least {_count(n, 'dI/dV spectrum', 'dI/dV spectra')} in the evidence field, "
                  f"{spectra}")
    if kind == "zspec_pair":
        a, b = rule.get("adatom") or {}, rule.get("background") or {}
        a_n, b_n = int(a.get("min_n", 1)), int(b.get("min_n", 1))
        lateral, span = _num(a.get("near_nm_max", 0.15)), _num(a.get("df_span_beyond_min_pm_min", 0))
        far = _num(b.get("near_nm_min", 2.0))
        return _t(lang,
                  f"evidence 中至少引用 {a_n} 条在吸附原子正上方（横向偏离不超过 "
                  f"{lateral} nm）测的 Δf(z) 曲线，越过 Δf 极小值后至少再逼近 "
                  f"{span} pm，中途没有跳变、没有接触；以及至少 "
                  f"{b_n} 条在离吸附原子 {far} nm 以外的干净表面上测的曲线（没有接触）",
                  f"cite at least {_count(a_n, 'Δf(z) curve', 'Δf(z) curves')} in the evidence field, "
                  f"measured directly above the adatom (lateral offset at most {lateral} nm), continuing at "
                  f"least {span} pm closer after passing the Δf minimum, with no jump and no contact along "
                  f"the sweep; and at least {_count(b_n, 'curve', 'curves')} measured on the clean surface "
                  f"more than {far} nm from the adatom (no contact)")
    if kind == "didv_maps":
        n, energies = int(rule.get("min_n", 1)), int(rule.get("min_energies", 1))
        v_lo, v_hi, near = _num(rule.get("v_lo_max")), _num(rule.get("v_hi_min")), \
            _num(rule.get("near_nm", 10.0))
        terms = _frame_terms(rule, lang)
        return _t(lang,
                  f"evidence 中至少引用 {n} 张开着锁相（lockin）的恒流扫描图（dI/dV 图），"
                  f"偏压至少 {energies} 个不同的值（按 10 mV 区分），覆盖 {v_lo} V "
                  f"到 {v_hi} V；每张图的中心离最近的散射体（台阶边或吸附原子）不超过 "
                  f"{near} nm，图须{terms}",
                  f"cite at least {_count(n, 'constant-current scan map', 'constant-current scan maps')} in "
                  f"the evidence field, recorded with the lock-in (lockin) on (dI/dV maps), with at least "
                  f"{_count(energies, 'different bias value', 'different bias values')} (binned in 10 mV "
                  f"steps) covering {v_lo} V to {v_hi} V; the centre of each map is within {near} nm of "
                  f"the nearest scatterer (step edge or adatom), and the maps must {terms}")
    if kind == "drag_series":
        return _drag_series(c, lang, sc)
    if kind == "any_of":
        alts = [_evidence({"evidence": r}, lang) for r in rule.get("rules") or []]
        return _t(lang, "以下任一组即可——" + "；或者，".join(f"（{chr(0x61 + i)}）{a}" for i, a in enumerate(alts)),
                  "any one of the following alternatives will do — "
                  + "; or ".join(f"({chr(0x61 + i)}) {a}" for i, a in enumerate(alts)))
    if kind == "events":
        n = int(rule.get("min_n", 1))
        event, cause = str(rule.get("event")), rule.get("cause")
        if event == "adatom_hop" and cause == "folme":
            # ``folme`` is the cause the instrument gives the hops of a move_tip action (the only
            # action that drags: World.move_xy_path); imaging hops are ``scan``. The judge counts
            # events, one per atom per action, each of at least one hop: for n > 1 that is not n hops
            once = _t(lang, "（同一个原子在一次 move_tip 动作中的多次跳动只算一次）",
                      " (the hops of one atom during one move_tip action count as one)") if n != 1 else ""
            return _t(lang, f"（不需要引用）本局至少有 {n} 次原子跳动发生在 move_tip 动作中{once}",
                      f"(no citation needed) this episode has had at least {_count(n, 'atom hop', 'atom hops')} "
                      f"made during {'a move_tip action' if n == 1 else 'move_tip actions'}{once}")
        # any other event, with the cause the judge requires of it (a rule never drops its cause)
        by = _t(lang, f"（原因：{cause}）", f" with the cause {cause}") if cause else ""
        events = _count(n, f"{event} event", f"{event} events")
        return _t(lang, f"（不需要引用）本局至少发生过 {n} 次{event}{by}",
                  f"(no citation needed) this episode has had at least {events}{by}")
    return _t(lang, f"（证据类型 {kind}）", f"(evidence type {kind})")


def _bystander_lines(c: dict, sc, lang: str) -> list[str]:
    """Which atoms a ``bystander_max_shift_nm`` constraint looks at and how it measures them, as
    ``AdatomRegistry.any_target`` computes the quantity: every other atom, whatever its history, that
    lay within ``bystander_nm`` of the moved atom's start or of its goal site *at the start of the
    episode* (so an atom that has since moved into or out of that neighbourhood does not change who
    counts); an atom's displacement is from its own start position, and a lost atom or one on the tip
    counts as moved (``GONE_SHIFT_NM``)."""
    if c.get("field") != "bystander_max_shift_nm":
        return []
    shift, _, radius = _atom_numbers(sc)
    return [
        _t(lang,
           f"   - 计入的其他原子：本局开始时，离该原子本局开始时的位置不超过 {radius} nm、或离'该位置平移 {shift} nm "
           f"后最近的那个晶格位点'不超过 {radius} nm 的每一个其他原子",
           f"   - The other atoms counted: every other atom that, at the start of the episode, was within "
           f"{radius} nm of that atom's position at the start of the episode, or within {radius} nm of 'the "
           f"lattice site nearest to the point obtained by shifting that position by {shift} nm'"),
        _t(lang,
           "   - 位移的算法：每个计入的原子，取它本局开始时的位置到本局结束时所在位置的距离；丢失的、"
           "或结束时还在针尖上的原子算移动；这个量取其中最大的（没有计入的原子则为 0）",
           "   - How displacement is measured: for each atom counted, the distance from its position at the "
           "start of the episode to where it is at the end of the episode; an atom that is lost, or is on "
           "the tip at the end, counts as moved; the quantity is the largest of these (0 if no atom is "
           "counted)")]


def _claim_lines(c: dict, sc, n: int, lang: str) -> list[str]:
    cid, unit = str(c["id"]), _unit(c)
    say = str(c.get("say" if lang == "zh" else "say_en") or "").strip()
    if c.get("kind") == "constraint":
        head = _t(lang, f"{n}. {cid}（不需要报告）：{say}", f"{n}. {cid} (no report needed): {say}")
    else:
        head = _t(lang, f"{n}. {cid}：{say}", f"{n}. {cid}: {say}") + \
            (_t(lang, f"，单位 {unit}", f", unit: {unit}") if unit and c.get("kind") != "position" else "")
    tolerance = _tolerance(c, sc, lang)
    lines = [head, _t(lang, f"   - 判对条件：{tolerance}", f"   - Judged correct when: {tolerance}")]
    other = c.get("distinct_from")
    if other:
        if c.get("kind") == "angle":
            sep = _num(other.get("min_sep_deg", other.get("min_sep", 0)))
            lines.append(_t(lang,
                            f"   - 与 {other.get('claim')} 两处的真实条纹方向须相差至少 "
                            f"{sep}°（同一个畴的两条臂不算两个畴）",
                            f"   - The true stripe directions at the two reported places (this result's and "
                            f"that of {other.get('claim')}) must differ by at least {sep}° (the two arms of "
                            f"one domain do not count as two domains)"))
        else:
            lines.append(_t(lang, f"   - 须与 {other.get('claim')} 对应不同的真实能级",
                            f"   - It must correspond to a different true level than {other.get('claim')} "
                            f"does"))
    if c.get("greater_than"):
        lines.append(_t(lang, f"   - 数值须大于 {c['greater_than']}",
                        f"   - The value must be greater than {c['greater_than']}"))
    if c.get("of_claim"):
        lines.append(_t(lang, f"   - 针对的是 {c['of_claim']} 所匹配到的那个原子",
                        f"   - It concerns the atom that {c['of_claim']} was matched to"))
        lines += _bystander_lines(c, sc, lang)
    if c.get("truth") == "drag_threshold":
        truth = _drag_truth(c, lang)
        lines.append(_t(lang, f"   - 真值：{truth}", f"   - True value: {truth}"))
    peer = (c.get("evidence") or {}).get("distinct_atom_from")
    if peer:
        lines.append(_t(lang, f"   - 计入的移动抓住的原子须不是 {peer} 所用的那个原子",
                        f"   - The atom the counted moves held must not be the one that {peer} used"))
    evidence = _evidence(c, lang, sc)
    lines.append(_t(lang, f"   - 证据：{evidence}", f"   - Evidence: {evidence}"))
    return lines


def _require_english(sc, *, task: bool) -> None:
    """The English words are fields of the scenario file, not something made up at run time: a
    scenario that lacks them has no English text, and the error says which fields."""
    missing = ["task_en"] if task and not str(getattr(sc, "task_en", "") or "").strip() else []
    missing += [f"say_en of claim {c['id']}" for c in sc.claims if not str(c.get("say_en") or "").strip()]
    if missing:
        raise ValueError(f"{sc.id}: no English text, missing {', '.join(missing)}")


def rule_card(sc, lang: str = "zh") -> str:
    """The rule card of a paper scenario, in Chinese (``"zh"``) or English (``"en"``), generated
    from its claims and budget. ``"en"`` needs a ``say_en`` on every claim."""
    _check_lang(lang)
    if lang == "en":
        _require_english(sc, task=False)
    lines = [_t(lang, f"【评分规则 v{RULES_VERSION}】以下规则由场景文件自动生成，判分只按这些规则进行。",
                f"[Scoring rules v{RULES_VERSION}] The rules below are generated automatically from the "
                f"scenario file; scoring follows only these rules."),
             "", _t(lang, "要报告的结果（每条用一次 ReportResult；可以重复报告，以最后一次为准；没有报告的结果算未复现）：",
                    "Results to report (report each with one ReportResult call; a result may be reported "
                    "repeatedly and the last report counts; a result that is not reported counts as not "
                    "reproduced):")]
    n = 0
    for c in sc.claims:
        n += 1
        lines += _claim_lines(c, sc, n, lang)
    moves = _cites_moves(sc)
    lines += ["",
              _t(lang,
                 "引用数据：每条报告用 evidence 列出你用到的扫描帧（.sxm 文件名）和谱（.dat 文件名），也可以写 frame:N、"
                 "sts:N、zspec:N" + ("；移动轨迹用 move_tip 返回的 .dat 文件名或 move:N。" if moves else "。")
                 + "判分只在你引用的数据里检查证据要求，没有引用的数据不算。",
                 "Citing data: in each report, list in the evidence field the scan frames (.sxm file names) "
                 "and spectra (.dat file names) you used; you may also write frame:N, sts:N or zspec:N"
                 + ("; a move trace is named by the .dat file name move_tip returned, or move:N. " if moves else ". ")
                 + "Scoring checks the evidence requirements only within the data you cite; data you do not "
                 "cite does not count."),
              _t(lang,
                 "位置与角度：位置用扫描坐标（nm），x 向右、y 向上，与扫描框的中心坐标是同一套坐标；位置类结果还要用 "
                 "frame 写明坐标取自哪一张帧，判分按这张帧逐行记录的漂移把坐标换算到样品上，你不需要自己修正漂移。"
                 "角度在同一坐标系里从 +x 轴逆时针量；图像数组的行号向下增大，在像素里量角度时要把符号取反。",
                 "Positions and angles: positions are in scan coordinates (nm), with x to the right and y "
                 "upward, the same coordinates as the centre coordinates of the scan window; for results that "
                 "carry a position (x_nm, y_nm) you must also name in the frame field the frame the "
                 "coordinates were read from, and scoring converts the coordinates to the sample using the "
                 "drift recorded row by row in that frame, so you do not need to correct the drift yourself. "
                 "Angles are measured in the same coordinate system, counter-clockwise from the +x axis; the "
                 "row index of an image array increases downward, so when you measure an angle in pixels you "
                 "must flip its sign.")]
    budget = sc.budget or {}
    hours = budget.get("sim_hours")
    if hours:
        percent = _num(round(100 * (OVERRUN_TOLERANCE - 1), 6))
        txt = _t(lang,
                 f"仪器时间：本题预算 {_num(hours)} 小时仪器时间（按仿真时间计，模型思考时仪器暂停）。用量超过预算的 "
                 f"{percent}% 以上，本次复现不算成功。",
                 f"Instrument time: this task's budget is {_num(hours)} "
                 f"{'hour' if float(hours) == 1 else 'hours'} of instrument time (counted in simulated "
                 f"time; the instrument is paused while the model thinks). If the instrument time used "
                 f"exceeds the budget by more than {percent}%, this reproduction does not count as successful.")
        # the file's ``wire_cmds`` is a limit of the controller TCP path; the benchmark instrument
        # counts no commands, so the card does not state it (it would be a rule nothing applies)
        lines.append(txt)
    return "\n".join(lines)


def task_text(sc, lang: str = "zh") -> str:
    """What the agent is told, in ``lang``: the scenario's task (``task``, or ``task_en``), then
    (paper scenarios) its rule card. The English text needs ``task_en`` and a ``say_en`` per claim."""
    _check_lang(lang)
    if lang == "en":
        _require_english(sc, task=True)
    task = str((sc.task if lang == "zh" else sc.task_en) or "").strip()
    if not sc.paper:
        return task
    return f"{task}\n\n{rule_card(sc, lang)}"


def apply_budget_overrun(v, consumed_sim_s: float | None, max_sim_s: float | None):
    """An episode that used more instrument time than the scenario allows did not do the
    task as posed: success → False, partial capped. One action can overshoot the budget (a
    long scan or wait runs to completion), so the rule is applied to the whole episode when it
    is judged. Judged on the sim truth otherwise; the overrun is recorded in the verdict details."""
    from .truth_criteria import Verdict

    if not max_sim_s or consumed_sim_s is None or consumed_sim_s <= max_sim_s * OVERRUN_TOLERANCE:
        return v
    details = dict(v.details)
    details["budget_overrun"] = {"sim_s": float(consumed_sim_s), "max_sim_s": float(max_sim_s),
                                 "factor": float(consumed_sim_s) / float(max_sim_s)}
    return Verdict(False, min(v.partial, OVERRUN_PARTIAL_CAP), details)


def has_budget_sentence(task: str) -> bool:
    """A budget stated by hand in a task (the rule card states it from the file instead)."""
    return bool(_BUDGET_SENTENCE.search(str(task or "")))
