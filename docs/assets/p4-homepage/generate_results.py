"""Render the compact P4 homepage scorecard from published trial results."""

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
AMBER = "#956015"
TEAL = "#14796B"


def render(results_path: Path, output_dir: Path) -> None:
    """Draw verified checks without treating missing evidence as physical harm."""
    data = json.loads(results_path.read_text(encoding="utf-8-sig"))
    names = ["Luna", "Terra", "Sol", "Astra"]
    by_name = {entry["model"]: entry for entry in data["models"]}
    entries = [by_name[name] for name in names]
    if data["difficulty"] != "easy":
        raise ValueError("This showcase is labelled for the easy setting.")
    if not data["mast_support"].startswith("enabled"):
        raise ValueError("This showcase is labelled with MAST enabled.")
    for entry in entries:
        if entry["episodes"] != 1 or entry["claims_total"] != 2:
            raise ValueError("This layout describes one trial and two checks.")
        verified = sum(
            [entry["target_position_verified"], entry["neighbour_preservation_verified"]]
        )
        if verified != entry["claims_verified"]:
            raise ValueError("The recorded check flags and score disagree.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "stmbench-p4-homepage",
        }
    )
    figure = plt.figure(figsize=(12, 3.6), dpi=100, facecolor=BACKGROUND)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set(xlim=(0, 1200), ylim=(360, 0))
    axes.axis("off")

    def label(x: float, y: float, text: str, **kwargs: object) -> None:
        axes.text(x, y, text, color=NAVY, va="center", **kwargs)

    label(30, 39, "Can an AI move one atom?", fontsize=22, weight="bold")
    label(30, 76, "Verified task checks", fontsize=12, weight="bold")
    label(236, 76, "|  Easy setting  |  MAST enabled", fontsize=12)

    card_width = 273
    for index, entry in enumerate(entries):
        x = 30 + index * 289
        score = entry["claims_verified"]
        complete = score == 2
        accent = TEAL if complete else AMBER
        fill = "#E8F4EE" if complete else "#F6F1E6"
        edge = "#A8CFC0" if complete else "#E4DAC5"
        axes.add_patch(
            FancyBboxPatch(
                (x, 108),
                card_width,
                175,
                boxstyle="round,pad=0,rounding_size=14",
                facecolor=fill,
                edgecolor=edge,
                linewidth=0.8,
            )
        )
        label(x + 20, 140, entry["model"], fontsize=16, weight="bold")
        axes.text(
            x + 20, 203, f"{score}/2", fontsize=43, weight="bold", va="center", color=accent
        )
        axes.text(
            x + 20,
            259,
            "Both verified" if complete else ("Not verified" if score == 0 else "Partly verified"),
            fontsize=12,
            va="center",
            color=accent,
        )

    label(30, 312, "Checks: atom at target + neighbours preserved.", fontsize=12)
    axes.text(
        30,
        338,
        "One trial per model. These are task scores, not success rates.",
        fontsize=10.5,
        color=SLATE,
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "p4-results"
    figure.savefig(stem.with_suffix(".png"), dpi=100, facecolor=BACKGROUND)
    figure.savefig(
        stem.with_suffix(".svg"),
        facecolor=BACKGROUND,
        metadata={"Date": None, "Creator": "STM-Bench"},
    )
    plt.close(figure)

    # Keep the graphic readable to assistive technology when it is opened directly.
    svg_path = stem.with_suffix(".svg")
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root.set("role", "img")
    root.set("aria-labelledby", "p4-results-title p4-results-description")
    title = ET.Element(f"{{{namespace}}}title", id="p4-results-title")
    title.text = "P4 demo: verified task checks by model"
    description = ET.Element(f"{{{namespace}}}desc", id="p4-results-description")
    scores = "; ".join(
        f"{entry['model']}: {entry['claims_verified']} of {entry['claims_total']} checks verified"
        for entry in entries
    )
    description.text = (
        f"{scores}. The two checks are atom at target and neighbours preserved. "
        "Easy setting, MAST enabled, one trial per model. Not verified means the "
        "required evidence did not establish the check; it does not assert that "
        "neighbours moved. These are task scores, not success rates."
    )
    root.insert(0, title)
    root.insert(1, description)
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    render(args.results, args.output_dir)
