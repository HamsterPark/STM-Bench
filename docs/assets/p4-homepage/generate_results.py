"""Render the P4 homepage scorecard from the published trial aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BACKGROUND = "#FCFBF7"
NAVY = "#182C3D"
SLATE = "#576777"
TEAL = "#14796B"
AMBER = "#956015"
PALE = "#EDEFEF"

EARLIER_NAMES = ("Luna", "Terra", "Sol", "Astra")
CURSOR_IDS = (
    "claude-opus-5-thinking-high",
    "inherit",
    "gpt-5.6-sol-medium",
    "cursor-grok-4.6-high-fast",
    "composer-2.5-fast",
)
# run order, not score order: one episode each, so the order must not read as a ranking
OPENCODE_IDS = (
    "opencode/cat-jam-free",
    "opencode/longcat-2.5-preview-free",
    "opencode/space-bunny-free",
    "opencode/mimo-v2.6-flash-free",
    "opencode/step-5-preview-free",
    "opencode/deepseek-v4.1-flash-free",
    "opencode/muse-spark-1.3-contributor-free",
    "opencode/nemotron-3.5-lightning-free",
)
ROW_PITCH = 50


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _check_context(data: dict, expected: tuple[str, str, int, str]) -> None:
    actual = (data["scenario"], data["difficulty"], data["seed"], data["mode"])
    if actual != expected:
        raise ValueError(f"Trial context differs from the P4 easy scorecard: {actual}")


def _row(entry: dict, label: str) -> dict:
    if entry["claims_total"] != 2:
        raise ValueError(f"Expected two claims for {label}")
    target = entry["target_position_verified"]
    neighbours = entry["neighbour_preservation_verified"]
    if not isinstance(target, bool) or not isinstance(neighbours, bool):
        raise ValueError(f"Claim flags must be booleans for {label}")
    score = entry["claims_verified"]
    if score != int(target) + int(neighbours):
        raise ValueError(f"Claim flags and score disagree for {label}")
    return {"label": label, "score": score, "target": target, "neighbours": neighbours}


def _load_rows(earlier_path: Path, cursor_path: Path, jev_path: Path,
               opencode_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    earlier = _read(earlier_path)
    cursor = _read(cursor_path)
    jev = _read(jev_path)
    opencode = _read(opencode_path)
    expected = ("P4_atom_positioning_cu111", "easy", 0, "H")
    for data in (earlier, cursor, jev, opencode):
        _check_context(data, expected)

    earlier_by_name = {entry["model"]: entry for entry in earlier["models"]}
    if set(earlier_by_name) != set(EARLIER_NAMES):
        raise ValueError("The earlier scorecard must contain the four replayed models")
    earlier_rows = []
    for name in EARLIER_NAMES:
        entry = earlier_by_name[name]
        if entry["episodes"] != 1:
            raise ValueError(f"Expected one earlier episode for {name}")
        earlier_rows.append(_row(entry, f"{name} · {entry['model_id']}"))

    cursor_by_id = {entry["model_id"]: entry for entry in cursor["models"]}
    if set(cursor_by_id) != set(CURSOR_IDS):
        raise ValueError("The later scorecard must contain the five selected Cursor IDs")
    later_rows = [_row(cursor_by_id[model_id], model_id) for model_id in CURSOR_IDS]

    runs = jev["runs"]
    if len(runs) != 5 or len({run["run_id"] for run in runs}) != 5:
        raise ValueError("Jev must describe five distinct development run records")
    if any(run["claims_verified"] != 0 or run["claims_total"] != 2 for run in runs):
        raise ValueError("Jev summary is labelled 0/2 in each of five runs")
    later_rows.append(
        {"label": "Jev Choice · five development runs", "score": 0,
         "target": False, "neighbours": False, "each": True}
    )

    if [entry["model_id"] for entry in opencode["models"]] != list(OPENCODE_IDS):
        raise ValueError("The free-tier batch must list its eight models in run order")
    if any(entry["episodes"] != 1 for entry in opencode["models"]):
        raise ValueError("Expected one episode for each free-tier model")
    if len({entry["run_id"] for entry in opencode["models"]}) != len(OPENCODE_IDS):
        raise ValueError("The free-tier batch must describe distinct run records")
    opencode_rows = [_row(entry, entry["model_id"].removeprefix("opencode/"))
                     for entry in opencode["models"]]
    return earlier_rows, later_rows, opencode_rows


def render(results_path: Path, output_dir: Path, *, cursor_path: Path | None = None,
           jev_path: Path | None = None, opencode_path: Path | None = None) -> None:
    """Draw all reported outcomes while keeping the different run sets explicit."""
    cursor_path = cursor_path or results_path.with_name("cursor-results.json")
    jev_path = jev_path or results_path.with_name("jev-results.json")
    opencode_path = opencode_path or results_path.with_name("opencode-results.json")
    earlier_rows, later_rows, opencode_rows = _load_rows(
        results_path, cursor_path, jev_path, opencode_path)

    # the first two groups keep their earlier positions; the third follows the second
    third_header = 409 + len(later_rows) * ROW_PITCH + 23
    notes_y = third_header + 24 + len(opencode_rows) * ROW_PITCH + 41
    height = notes_y + 3 * 29 + 42

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "stmbench-p4-homepage",
    })
    figure = plt.figure(figsize=(11, height / 100), dpi=100, facecolor=BACKGROUND)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set(xlim=(0, 1100), ylim=(height, 0))
    axes.axis("off")

    def label(x: float, y: float, value: str, **kwargs: object) -> None:
        kwargs.setdefault("color", NAVY)
        axes.text(x, y, value, va="center", **kwargs)

    label(28, 39, "Can an AI move one atom?", fontsize=23, weight="bold")
    label(28, 78, "P4 easy · verified task checks · seed 0 · mode H", fontsize=12)
    for x, heading in ((697, "ATOM AT TARGET"), (852, "NEIGHBOURS"), (1035, "SCORE")):
        label(x, 113, heading, fontsize=10, weight="bold", color=SLATE,
              ha="right" if heading == "SCORE" else "center")

    def draw_group(heading: str, rows: list[dict], header_y: int, first_y: int) -> None:
        label(28, header_y, heading, fontsize=12, weight="bold", color=SLATE)
        for index, row in enumerate(rows):
            y = first_y + index * ROW_PITCH
            fill = "#F8F6F0" if row.get("each") else "#F7F8F7"
            axes.add_patch(FancyBboxPatch(
                (24, y), 1052, 43, boxstyle="round,pad=0,rounding_size=8",
                facecolor=fill, edgecolor="#DFE4E0", linewidth=0.8,
            ))
            label(40, y + 21.5, row["label"], fontsize=13.5,
                  weight="bold" if row.get("each") else "normal")
            for x, verified in ((635, row["target"]), (790, row["neighbours"])):
                axes.add_patch(FancyBboxPatch(
                    (x, y + 6), 125, 31, boxstyle="round,pad=0,rounding_size=6",
                    facecolor=TEAL if verified else PALE,
                    edgecolor=TEAL if verified else "#D9DFDC", linewidth=0.8,
                ))
                label(x + 62.5, y + 21.5, "VERIFIED" if verified else "—",
                      fontsize=10.5, weight="bold", ha="center",
                      color="white" if verified else SLATE)
            score = row["score"]
            score_color = TEAL if score == 2 else (AMBER if score == 1 else SLATE)
            label(1038, y + 21.5, f"{score}/2" + (" each" if row.get("each") else ""),
                  fontsize=16 if row.get("each") else 19, weight="bold",
                  ha="right", color=score_color)

    draw_group("EARLIER REPLAYED EPISODES · 20–21 SEP 2026", earlier_rows, 139, 162)
    draw_group("LATER RESULTS · 21 SEP 2026 · FIVE CURSOR EPISODES + JEV CHOICE",
               later_rows, 385, 409)
    draw_group("15 OCT 2026 · EIGHT FREE-TIER MODELS · AGENT HARNESS + OPERATOR CONSOLE",
               opencode_rows, third_header, third_header + 24)
    for index, note in enumerate((
        "Two checks: atom at target and neighbours preserved. An unverified check does not prove damage.",
        "Jev: 0/2 in each of five development runs; its choice driver changed between runs.",
        "15 Oct: rows in run order; a different interface, and MAST changed during that batch.",
        "Single-seed development results, not success rates or a mode-A leaderboard.",
    )):
        label(28, notes_y + index * 29, note, fontsize=10.5, color=SLATE)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "p4-results"
    figure.savefig(stem.with_suffix(".png"), dpi=100, facecolor=BACKGROUND)
    figure.savefig(stem.with_suffix(".svg"), facecolor=BACKGROUND,
                   metadata={"Date": None, "Creator": "STM-Bench"})
    plt.close(figure)

    # Keep the SVG understandable without its surrounding README text.
    svg_path = stem.with_suffix(".svg")
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root.set("role", "img")
    root.set("aria-labelledby", "p4-results-title p4-results-description")
    title = ET.Element(f"{{{namespace}}}title", id="p4-results-title")
    title.text = "P4 easy: eighteen scorecard rows and two verified task checks"
    description = ET.Element(f"{{{namespace}}}desc", id="p4-results-description")
    scores = "; ".join(
        f"{row['label']}: {row['score']} of 2 verified"
        for row in earlier_rows + later_rows + opencode_rows
    )
    description.text = (
        f"{scores}. Jev's result occurred in each of five development runs with driver changes. "
        "The eight free-tier models of 15 October 2026 are listed in run order; they used an "
        "agent harness with an operator console, and MAST changed during that batch. "
        "The two checks are atom at target and neighbours preserved. An unverified "
        "check does not imply that a neighbour moved. All are P4 easy, seed 0, mode H "
        "development observations, not success rates or mode-A leaderboard results."
    )
    root.insert(0, title)
    root.insert(1, description)
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--cursor-results", type=Path,
                        default=Path(__file__).with_name("cursor-results.json"))
    parser.add_argument("--jev-results", type=Path,
                        default=Path(__file__).with_name("jev-results.json"))
    parser.add_argument("--opencode-results", type=Path,
                        default=Path(__file__).with_name("opencode-results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    render(args.results, args.output_dir, cursor_path=args.cursor_results,
           jev_path=args.jev_results, opencode_path=args.opencode_results)
