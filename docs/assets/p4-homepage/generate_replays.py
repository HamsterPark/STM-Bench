"""Render measurement-led replay summaries from immutable exported evidence.

Images are replay renderings of acquired scans. No noise, tip faults, drift or
atom motion is invented. Annotations and scene durations are editorial; command
feedback, image estimates and retrospective audit findings remain distinct.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 920
BG, NAVY, MUTED = "#FCFBF7", "#182C3D", "#576777"
TEAL, AMBER, CYAN = "#14796B", "#956015", "#5EF2DB"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / filename), size)


def lines(draw, text, face, width):
    result = []
    for part in text.split("\n"):
        line = ""
        for word in part.split():
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=face) > width and line:
                result.append(line)
                line = word
            else:
                line = candidate
        result.append(line)
    return result


def paragraph(draw, text, x, y, width, size=25, colour=NAVY, bold=False, max_lines=None):
    face = font(size, bold)
    wrapped = lines(draw, text, face, width)
    if max_lines is not None and len(wrapped) > max_lines:
        raise ValueError(f"Text does not fit: {text}")
    for line in wrapped:
        draw.text((x, y), line, fill=colour, font=face)
        y += size + 9
    return y


def stories(model):
    """Narration paraphrases recorded actions, not unobserved internal reasoning."""
    if model == "Astra":
        # frame, title, measurement, response, evidence, completed attempts
        rows = [
            (1, "Find a target in a measured image", "The automatic finder returns no candidates, even though bright spots are visible.", "Inspect the scan and zoom into a promising area.", "A tool's empty candidate list does not settle what is on the surface.", 0),
            (2, "Establish a reference before moving", "The closer scan resolves five spots. Astra identifies an isolated atom.", "Measure its position and record the nearby atoms before requesting a move.", "The final check must cover the neighbours as well as the selected atom.", 0),
            (3, "The first move falls well short", "A new scan shows only about 0.67 nm of the requested 4 nm move.", "Use the newly measured position to plan another attempt.", "Tool summary: moved; rescan not confirmed. The image reveals the shortfall.", 1),
            (3, "Change how the atom is moved", "The atom stopped long before the destination. Repeating the command alone is not enough evidence.", "Slow the motion fivefold, keep the same destination, then scan again.", "Second attempt: the speed changes from 0.5 to 0.1 nm/s.", 1),
            (4, "Closer, but still unfinished", "The next scan shows about 3.84 nm of movement. A small error remains.", "Try a short correction and acquire another image.", "Near the destination is not the same as a verified final position.", 2),
            (5, "A correction makes no progress", "Another scan puts the atom in the same place. The third attempt has stalled.", "Increase the interaction strength modestly while keeping the slower motion.", "Post-run record: this attempt produced zero atom hops, despite the usual tool summary.", 3),
            (6, "Adjust, then measure again", "The final scan places the atom at its destination after a stronger fourth attempt.", "Compare this scan with the reference, checking the atom and all three neighbours.", "The final adjustment included a backwards hop before reaching the target.", 4),
            (6, "Verify the whole result", "The atom has moved about 4.04 nm. Neighbours remain in place within the image uncertainty.", "Submit the measured results for independent verification.", "2/2 checks pass: target position and neighbour preservation, with measurement evidence.", 4),
        ]
    else:
        rows = [
            (1, "Search an uneven surface", "The survey contains small bright spots against a strong background.", "Take a smaller, more detailed scan to narrow the search.", "The model must find a usable atom before any manipulation can begin.", 0),
            (2, "Inspect a candidate more closely", "The second survey identifies a promising area for a close-up scan.", "Inspect that area at higher resolution before trying to move an atom.", "Post-run check: the chosen area really contains an isolated atom.", 0),
            (3, "A warning interrupts the plan", "The analysis tool flags a possible tip change partway through this scan.", "Terra reports suspected instability and calls a recovery routine, capped at three attempts.", "Post-run audit: no tip change occurred in this scan. The warning was a false alarm.", 0),
            (4, "Check the recovery baseline", "The routine acquires a baseline image before applying a pulse.", "The MAST routine begins its bounded sequence of conditioning pulses and checks.", "The baseline quality score was not retained. The three later scores are recorded.", 0),
            (5, "First recovery attempt", "After one pulse and a fresh scan, the tool reports zero quality.", "The recovery routine proceeds to its second allowed attempt.", "This is tool-reported quality, not a direct measurement of tip health.", 1),
            (6, "Second recovery attempt", "A second pulse and scan do not improve the reported score.", "The routine uses its final allowed recovery attempt.", "The model is now trying to restore confidence in a measurement before acting on it.", 2),
            (7, "Third recovery attempt", "The third pulse is followed by another scan. The score remains below the requested threshold.", "The routine returns without establishing successful recovery.", "All seven scans were saved. The quality metric could return zero on these images without proving a bad tip.", 3),
            (7, "Stop without claiming success", "After three recovery attempts, the measurement remains unverified.", "Terra stops. It reports no completed atom movement and submits no success claim.", "0/2 checks. Post-run audit exposes a false alarm and a limitation of the MAST quality metric.", 3),
        ]
    return [dict(frame=r[0], title=r[1], observation=r[2], response=r[3], evidence=r[4],
                 attempts=r[5], duration=8000 if i < len(rows)-1 else 10000)
            for i, r in enumerate(rows)]


def image_point(geom, position, box):
    x, y, size = box
    return (x + (position[0] - geom["cx_nm"]) / geom["w_nm"] * size + size / 2,
            y - (position[1] - geom["cy_nm"]) / geom["h_nm"] * size + size / 2)


def outlined_label(draw, xy, text, size=18, colour=CYAN):
    draw.text(xy, text, fill=colour, font=font(size, True), stroke_width=2, stroke_fill=NAVY)


def render_frame(trial, story, index, count):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    astra = trial["model"] == "Astra"
    accent = TEAL if astra else AMBER
    draw.text((32, 23), "STM-BENCH  /  TASK 4  /  EASY, ONE TRIAL  /  TIME CONDENSED", font=font(18, True), fill=MUTED)
    heading = "Astra: learn from each incomplete move" if astra else "Terra: a diagnostic warning derails the task"
    draw.text((32, 61), heading, font=font(37, True), fill=NAVY)
    sub = "6 scans  |  4 move attempts  |  54 instrument minutes" if astra else "7 scans  |  3 recovery pulses  |  stopped without claiming success"
    draw.text((32, 116), sub, font=font(23), fill=MUTED)
    draw.rounded_rectangle((32, 161, 1248, 209), radius=10, fill="#E8F4EE" if astra else "#F6F1E6")
    draw.text((48, 171), f"{index + 1:02d} / {count:02d}   {story['title']}", font=font(25, True), fill=accent)

    recorded = trial["frames"][story["frame"] - 1]
    raw = Image.open(ROOT / recorded["image"]).convert("RGB")
    draw.text((32, 231), f"RECORDED SCAN {story['frame']} / {len(trial['frames'])}", font=font(19, True), fill=MUTED)
    box = (32, 265, 416)
    image.paste(raw.resize((416, 416), Image.Resampling.NEAREST), box[:2])
    geom = recorded["geom"]
    if astra and story["frame"] > 1:
        pos = trial["image_estimates"]["positions_by_frame"][str(story["frame"])]
        px, py = image_point(geom, pos, box)
        draw.ellipse((px - 17, py - 17, px + 17, py + 17), outline=CYAN, width=2)
        outlined_label(draw, (px + 22, py - 9), "Selected atom", size=15)
    elif (astra and story["frame"] == 1) or (not astra and story["frame"] < 3):
        cx, cy, width = (-8, -13, 24) if astra else ((0, 0, 30) if story["frame"] == 1 else (-12.4, 11.1, 6))
        p1 = image_point(geom, (cx - width / 2, cy + width / 2), box)
        p2 = image_point(geom, (cx + width / 2, cy - width / 2), box)
        draw.rectangle((*p1, *p2), outline=CYAN, width=2)
        outlined_label(draw, (p1[0] + 4, p1[1] + 5), "Next close-up", size=15)
    if not astra and story["frame"] == 3:
        row = trial["diagnostic_context"]["audit_reproduced_change_row"]
        yy = 265 + row / 256 * 416
        for x in range(32, 448, 16):
            draw.line((x, yy, min(x + 8, 448), yy), fill="#FFD994", width=2)
        outlined_label(draw, (42, yy - 26), "Audit-reproduced warning", 16, "#FFD994")
    draw.text((32, 690), f"{geom['w_nm']:g} nm view | Replay rendering of saved data", fill=MUTED, font=font(17))

    def card(y, label, text, fill, label_colour):
        draw.rounded_rectangle((480, y, 1248, y + 164), radius=12, fill=fill)
        draw.text((500, y + 14), label, font=font(18, True), fill=label_colour)
        paragraph(draw, text, 500, y + 46, 718, size=25, max_lines=3)

    card(230, "OBSERVATION / TOOL FEEDBACK", story["observation"], "#F0F1EE", MUTED)
    card(406, "RESPONSE", story["response"], "#E8F4EE" if astra else "#F6F1E6", accent)
    card(582, "EVIDENCE / POST-RUN CONTEXT", story["evidence"], "#EEF0F3", MUTED)

    if astra:
        draw.text((32, 720), "SAME TARGET AREA, ENLARGED", font=font(16, True), fill=MUTED)
        if story["frame"] > 1:
            # Fixed sample coordinates prevent a changed scan centre from being
            # mistaken for physical movement or drift.
            left, right, low, high = -7.2, -0.6, -12.6, -10.4
            p1 = image_point(geom, (left, high), (0, 0, raw.width))
            p2 = image_point(geom, (right, low), (0, 0, raw.width))
            crop = raw.resize((416, 139), resample=Image.Resampling.NEAREST, box=(*p1, *p2))
            image.paste(crop, (32, 748))
            target = trial["image_estimates"]["requested_target_nm"]
            tx = 32 + (target[0] - left) / (right - left) * 416
            draw.line((tx, 748, tx, 887), fill=CYAN, width=2)
            outlined_label(draw, (tx + 6, 753), "Goal x", 15)
        else:
            paragraph(draw, "The reference close-up has not been acquired yet.", 32, 762, 408, 23, MUTED, max_lines=3)
        progress(draw, trial, story)
    else:
        draw.text((32, 726), "REPORTED QUALITY DURING RECOVERY", font=font(16, True), fill=MUTED)
        for n in range(4):
            x = 75 + n * 105
            value = trial["diagnostic_context"]["tool_reported_quality"][n]
            known = story["frame"] >= 4 + n and value is not None
            draw.text((x, 765), f"{value:g}" if known else "--", font=font(34, True), fill=accent if known else "#AFB7BE", anchor="mt")
            label = "Before" if n == 0 else f"Pulse {n}"
            draw.text((x, 812), label, font=font(16), fill=MUTED, anchor="mt")
        draw.text((32, 853), "Tool score; not ground truth about the tip.", font=font(17), fill=MUTED)
        paragraph(draw, "Post-run audit: false alarm. The quality metric could return zero without establishing a bad tip.", 500, 770, 718, 24, AMBER, max_lines=3)

    # Scene order only: this is not an instrument-time or wall-time scale.
    for n in range(count):
        x = 480 + n * 96
        draw.rounded_rectangle((x, 900, x + 86, 906), radius=3,
                               fill=accent if n <= index else "#DCE0E2")
    return image


def progress(draw, trial, story):
    draw.text((500, 761), "IMAGE-ESTIMATED MOVEMENT  |  REQUESTED: 4 nm", font=font(17, True), fill=MUTED)
    values = trial["image_estimates"]["displacement_by_attempt_nm"]
    for n, value in enumerate(values):
        x = 502 + 185 * n
        known = n < story["attempts"]
        colour = TEAL if n == 3 else AMBER
        draw.text((x, 788), f"{value:.2f}" if known else "--", font=font(30, True), fill=colour if known else "#AFB7BE")
        draw.text((x, 830), f"Attempt {n + 1}", font=font(17), fill=MUTED)
        draw.rounded_rectangle((x, 860, x + 152, 869), radius=4, fill="#DEE3E3")
        if known:
            draw.rounded_rectangle((x, 860, x + 152 * min(value / 4.04, 1), 869), radius=4, fill=colour)


def render():
    evidence = json.loads((ROOT / "replay-evidence.json").read_text(encoding="utf-8"))
    assert evidence["difficulty"] == "easy"
    manifest = []
    for trial in evidence["trials"]:
        for f in trial["frames"]:
            assert hashlib.sha256((ROOT / f["image"]).read_bytes()).hexdigest() == f["sha256"]
            assert f["drift_nm"] == [0, 0] and f["complete"]
        story = stories(trial["model"])
        frames = [render_frame(trial, s, i, len(story)) for i, s in enumerate(story)]
        palette_source = Image.new("RGB", (WIDTH, HEIGHT * len(frames)))
        for i, frame in enumerate(frames):
            palette_source.paste(frame, (0, HEIGHT * i))
        # Shared palette prevents artificial colour fluctuations between frames.
        palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
        stem = trial["model"].lower() + "-replay"
        quantized[0].save(ROOT / (stem + ".gif"), save_all=True,
                          append_images=quantized[1:], duration=[s["duration"] for s in story],
                          loop=0, disposal=1, optimize=True)
        frames[-1].save(ROOT / (stem + ".png"))
        sheet = Image.new("RGB", (WIDTH, HEIGHT * math.ceil(len(frames) / 2) // 2), BG)
        for i, frame in enumerate(frames):
            sheet.paste(frame.resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS),
                        ((i % 2) * WIDTH // 2, (i // 2) * HEIGHT // 2))
        sheet.save(ROOT / (stem + "-storyboard.png"))
        manifest.append({"model": trial["model"], "file": stem + ".gif",
                         "duration_ms": sum(s["duration"] for s in story),
                         "frames": len(frames), "scenes": story})
    (ROOT / "replay-scenes.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in m.items() if k != "scenes"} for m in manifest], indent=2))


if __name__ == "__main__":
    render()
