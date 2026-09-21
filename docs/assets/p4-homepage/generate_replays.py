"""Render concise GIF replays from the published, derived atom-position records.

Dots are enlarged for readability. Their centres use recorded coordinates without
interpolation; arrows show requested moves, not measured tip trajectories. Display
durations are editorial and are not proportional to instrument time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SIZE = (800, 620)
BACKGROUND = "#FCFBF7"
NAVY = "#182C3D"
SLATE = "#576777"
FIELD = "#152738"
MUTED = "#94A7B4"
TEAL = "#67DEC7"
AMBER = "#F0BF69"
WHITE = "#F4F8FA"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / filename), size)


def frame(trial: dict, stage: dict) -> Image.Image:
    """Draw one state. Each moved dot is placed at an actual recorded position."""
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    success = trial["task_completed"]
    accent = TEAL if success else AMBER
    draw.text((28, 22), trial["model"], fill=NAVY, font=font(38, True))
    badge = "VERIFIED SUCCESS" if success else "UNSUCCESSFUL ATTEMPT"
    badge_font = font(19, True)
    badge_width = draw.textlength(badge, font=badge_font) + 32
    bx = 772 - badge_width
    draw.rounded_rectangle((bx, 24, 772, 65), radius=18,
                           fill="#E8F4EE" if success else "#F6F1E6")
    draw.text((bx + 16, 33), badge, fill="#14796B" if success else "#956015",
              font=badge_font)
    draw.text((28, 75), "Condensed replay of recorded atom positions", fill=SLATE,
              font=font(21))

    draw.rounded_rectangle((28, 112, 772, 465), radius=20, fill=FIELD)
    xmin, xmax, ymin, ymax = trial["bounds_nm"]
    scale = min(688 / (xmax - xmin), 286 / (ymax - ymin))

    def xy(x: float, y: float) -> tuple[float, float]:
        return 400 + (x - (xmin + xmax) / 2) * scale, 292 - (y - (ymin + ymax) / 2) * scale

    def circle(point: tuple, r: float, **kwargs) -> None:
        x, y = point
        draw.ellipse((x - r, y - r, x + r, y + r), **kwargs)

    def arrow(start: list, end: list) -> None:
        x0, y0 = xy(*start)
        x1, y1 = xy(*end)
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1:
            return
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        # An annotation of the requested path; it is never presented as atom motion.
        for distance in range(18, max(19, int(length) - 18), 14):
            stop = min(distance + 7, length - 18)
            draw.line((x0 + distance * ux, y0 + distance * uy,
                       x0 + stop * ux, y0 + stop * uy), fill=accent, width=2)
        ex, ey = x1 - 19 * ux, y1 - 19 * uy
        draw.line((ex - 8 * ux + 5 * uy, ey - 8 * uy - 5 * ux,
                   ex, ey, ex - 8 * ux - 5 * uy, ey - 8 * uy + 5 * ux),
                  fill=accent, width=2)

    atoms = {a["id"]: dict(a) for a in trial["initial_atoms"]}
    count = stage["moves"]
    for move in trial["moves"][:count]:
        atoms[move["id"]].update(move)

    selected = trial["selected_atom_id"]
    attempt = trial["attempts"][stage.get("attempt", 0)]
    goal = trial["target"]["site"] if success else attempt["requested_end_nm"]
    if stage.get("arrow"):
        arrow(attempt["requested_start_nm"], attempt["requested_end_nm"])

    # Retain the exact positions, including revisits, in the visible trail.
    if success:
        for move in trial["moves"][:count]:
            circle(xy(move["x"], move["y"]), 2, fill="#438A81")
        origin = xy(*trial["target"]["start"])
        circle(origin, 7, outline="#668394", width=1)
        draw.text((origin[0], origin[1] + 24), "Start", fill=MUTED,
                  font=font(20), anchor="mt")
    else:
        origin = xy(*attempt["requested_start_nm"])
        draw.line((origin[0] - 7, origin[1] - 7, origin[0] + 7, origin[1] + 7),
                  fill=AMBER, width=3)
        draw.line((origin[0] - 7, origin[1] + 7, origin[0] + 7, origin[1] - 7),
                  fill=AMBER, width=3)
        draw.text((origin[0] - 12, origin[1] + 25), "Empty start", fill=AMBER,
                  font=font(20), anchor="rt")

    destination = xy(*goal)
    circle(destination, 17, outline=TEAL if stage.get("final") and success else WHITE,
           width=2)
    label = "Target" if success else "Requested target"
    draw.text((destination[0] + 25, destination[1] + 21), label,
              fill=WHITE, font=font(20), anchor="lt")
    for atom in atoms.values():
        if atom["status"] != "on_surface":
            continue
        pos = xy(atom["x"], atom["y"])
        colour = TEAL if atom["id"] == selected else MUTED
        circle(pos, 13 if atom["id"] == selected else 10, fill="#254758")
        circle(pos, 8 if atom["id"] == selected else 6, fill=colour)
        circle((pos[0] - 1, pos[1] - 1), 2, fill=WHITE)

    if success:
        draw.text((53, 132), "Selected atom", font=font(18), fill=TEAL)
        draw.text((550, 132), "Other atoms stay put", font=font(18), fill=MUTED)
    else:
        draw.text((53, 132), "Cross = requested start", font=font(18), fill=AMBER)
        draw.text((550, 132), "Dots = actual atoms", font=font(18), fill=MUTED)

    number = stage["phase"]
    phase_count = 6 if success else 5
    for i in range(phase_count):
        width = (744 - 8 * (phase_count - 1)) / phase_count
        x = 28 + i * (width + 8)
        draw.rounded_rectangle((x, 482, x + width, 488), radius=3,
                               fill=("#14796B" if success else "#956015")
                               if i < number else "#E3E5E5")
    draw.text((28, 502), stage["title"], fill=NAVY, font=font(31, True))
    draw.text((28, 547), stage["subtitle"], fill=SLATE, font=font(23))
    draw.text((28, 588), "Positions to scale; dots enlarged. Time condensed.",
              fill=SLATE, font=font(17))
    return image


def storyboard(trial: dict) -> list[dict]:
    stages = []

    def add(phase, title, subtitle, moves, duration, **kwargs):
        stages.append(dict(phase=phase, title=title, subtitle=subtitle,
                           moves=moves, duration=duration, **kwargs))

    if trial["model"] == "Astra":
        add(1, "Find an atom", "Choose one isolated atom to move.", 0, 2000)
        stories = [
            (2, "First move falls short", "Check the image: more work is needed."),
            (3, "Adjust and try again", "Move closer, then check again."),
            (4, "Still short of the target", "A small correction changes nothing."),
            (5, "Make a final adjustment", "Check the result once more."),
        ]
        count = 0
        for index, (attempt, story) in enumerate(zip(trial["attempts"], stories)):
            phase, title, subtitle = story
            add(phase, title, subtitle, count, 600, attempt=index, arrow=True)
            for move_index in attempt["move_indices"]:
                count = move_index + 1
                add(phase, title, subtitle, count, 170, attempt=index, arrow=True)
            add(phase, title, subtitle, count, 1600, attempt=index)
        add(6, "Verified: target reached", "Nearby atoms stayed in place. 2/2 checks.",
            count, 4000, final=True)
    else:
        add(1, "Look for an atom", "Actual atoms appear as dots in this replay.", 0, 2300)
        add(2, "Choose an empty spot", "No atom is at the requested starting point.", 0, 2600)
        add(3, "Send the move command", "The requested path crosses empty space.", 0, 3000, arrow=True)
        add(4, "Report success", "Luna reports that the move is complete.", 0, 2800, arrow=True)
        add(5, "Verified: no atom moved", "Neither task check passes. 0/2 checks.", 0, 4000, final=True)
    return stages


def render() -> None:
    source = json.loads((ROOT / "replay-data.json").read_text(encoding="utf-8"))
    scores = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    models = {m["model"]: m for m in scores["models"]}
    summary = []
    for trial in source["trials"]:
        model = models[trial["model"]]
        assert trial["episode_sha256"] == model["episode_sha256"]
        assert trial["verified_checks"] == model["claims_verified"]
        assert len(trial["moves"]) == model["recorded_atom_hops"]
        if trial["model"] == "Astra":
            assert [len(a["move_indices"]) for a in trial["attempts"]] == [3, 16, 0, 3]
            final = trial["moves"][-1]
            assert [final["x"], final["y"]] == trial["target"]["site"]
        else:
            assert not trial["moves"] and trial["selected_atom_id"] is None
        stages = storyboard(trial)
        frames = [frame(trial, stage) for stage in stages]
        stem = trial["model"].lower() + "-replay"
        # A single palette prevents colour flicker between successive states.
        palette = frames[-1].quantize(colors=192, method=Image.Quantize.MEDIANCUT)
        gif_frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        gif_frames[0].save(ROOT / (stem + ".gif"), save_all=True,
                           append_images=gif_frames[1:],
                           duration=[s["duration"] for s in stages],
                           loop=0, optimize=True, disposal=1)
        frames[-1].save(ROOT / (stem + ".png"))
        # A motion-free storyboard is also useful for accessibility and review.
        phase_ends = [max(i for i, s in enumerate(stages) if s["phase"] == p)
                      for p in range(1, stages[-1]["phase"] + 1)]
        sheet = Image.new("RGB", (1200, math.ceil(len(phase_ends) / 2) * 465), BACKGROUND)
        for cell, index in enumerate(phase_ends):
            tile = frames[index].resize((600, 465), Image.Resampling.LANCZOS)
            sheet.paste(tile, ((cell % 2) * 600, (cell // 2) * 465))
        sheet.save(ROOT / (stem + "-storyboard.png"))
        with Image.open(ROOT / (stem + ".gif")) as check:
            duration = 0
            for i in range(check.n_frames):
                check.seek(i)
                duration += check.info["duration"]
            assert duration == sum(s["duration"] for s in stages)
            assert check.size == SIZE and check.info["loop"] == 0
            summary.append({"model": trial["model"], "frames": check.n_frames,
                            "duration_ms": duration, "file": stem + ".gif"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    render()
