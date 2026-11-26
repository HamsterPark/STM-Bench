"""The evidence field's required mark and the amber note after a report with no files (screen trial 24)."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from stmbench.gui import shots  # noqa: E402
from tests.gui_helpers import app, fonts, open_window, run_button  # noqa: E402, F401

P1, P4 = "P1_barth1990_au111", "P4_atom_positioning_cu111"

def starred(rep, key: str) -> bool:
    return rep.labels[key].text().endswith("*</span>")


# ── the evidence field ──
def test_the_evidence_label_is_marked_for_the_claims_whose_card_line_asks_to_cite(open_window):
    win = open_window("en", scenario=P1)
    win.sidebar.show_tab("report")
    rep = win.sidebar.report
    run_button(win, "report.claim.stripe_period_nm")                 # frame: "cite at least one scan frame"
    assert rep.required_fields() == {"value", "evidence"} and starred(rep, "evidence") and starred(rep, "value")
    assert rep.fields["evidence"].text() == ""                       # still empty: no prefill, and Submit is not blocked
    run_button(win, "report.claim.stripe_orientation_deg")           # frame_covers: the frame field carries it
    assert "evidence" not in rep.required_fields() and not starred(rep, "evidence")
    run_button(win, "report.claim.stripe_period_nm")
    assert starred(rep, "evidence")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_marked_evidence_label_keeps_its_line_with_the_viewed_file_button(fonts, open_window, lang):
    for scenario in ("P1_barth1990_au111", "P2_standing_waves_cu111", "P3_corral_cu111", "P3_corral_repair_cu111", "P4_atom_positioning_cu111",
                     "P5_sader_jarvis_qplus", "P7_pull_threshold_cu111"):
        win = open_window(lang, scenario=scenario, name=f"{scenario}{lang}")
        win.sidebar.show_tab("report")
        rep = win.sidebar.report
        for cid, c in rep._claims.items():
            run_button(win, f"report.claim.{cid}")
            label, button = rep.labels["evidence"], rep.viewed_button
            assert starred(rep, "evidence") == bool(c["cite"]), (scenario, cid)
            top = label.mapTo(rep, label.rect().topLeft()).y()
            assert top <= button.mapTo(rep, button.rect().center()).y() <= top + label.height(), (scenario, cid)   # one line
            assert label.height() < 2 * label.fontMetrics().height(), (scenario, cid)
            assert label.sizeHint().width() <= label.width() + 1, (scenario, cid)


def test_a_report_of_a_citing_claim_with_no_files_gets_an_amber_note_and_others_do_not(open_window):
    win = open_window("en", scenario=P1)
    win.sidebar.show_tab("report")
    rep = win.sidebar.report
    note = "no evidence files named"
    run_button(win, "report.claim.stripe_period_nm")
    shots.fill(win, {"report.value": "4.5"})
    run_button(win, "report.submit")
    shots.settle(win)
    tones = [t for t, _ in win.recent.entries()]
    assert win.recent.entries()[0] == ("warn", note) and win.recent.entries()[1][0] == "ok"      # beside the reply
    assert note in rep.reply.text() and "recorded" in rep.reply.text()
    run_button(win, "report.claim.stripe_orientation_deg")           # does not cite: no note
    shots.fill(win, {"report.value": "30", "report.x_nm": "1", "report.y_nm": "2", "report.frame": "sim001.sxm"})
    n = len(win.recent.entries())
    run_button(win, "report.submit")
    shots.settle(win)
    assert win.recent.entries()[0][0] == "ok" and "recorded" in rep.reply.text() and note not in rep.reply.text()
    run_button(win, "report.claim.stripe_period_nm")                 # cites, with a file named: no note
    shots.fill(win, {"report.value": "4.6", "report.evidence": "sim001.sxm"})
    run_button(win, "report.submit")
    shots.settle(win)
    assert win.recent.entries()[0][0] == "ok" and note not in rep.reply.text()
    assert tones[0] == "warn"


def test_a_refused_report_gets_no_note(open_window):
    win = open_window("en", scenario=P1)
    win.sidebar.show_tab("report")
    run_button(win, "report.claim.stripe_period_nm")                 # no value: the instrument refuses
    run_button(win, "report.submit")
    shots.settle(win)
    assert all(t != "warn" for t, _ in win.recent.entries())
    assert win.worker.held


