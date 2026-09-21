"""Render a compact Terra contrast from immutable recorded scan evidence.

The image uses one fixed sample-coordinate crop throughout. Only playback and
recovery-status indicators animate; the scan data are never warped or blended.
Recovery pulses belong to the single MAST routine requested by Terra, rather
than three independent model decisions. Audit labels are retrospective.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1280, 720
BG, NAVY, MUTED = "#FCFBF7", "#182C3D", "#576777"
TEAL, AMBER, CYAN = "#14796B", "#956015", "#5EF2DB"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
FRAME_MS = 200
SCAN_BOX = (44, 124, 420)


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def scenes() -> list[dict]:
    rows = [
        (3, 3000, 0, "warning", "A warning interrupts the plan",
         "Terra requests recovery before proceeding."),
        (4, 1000, 0, "baseline", "Check before changing the instrument",
         "MAST records a reference scan."),
        (5, 3000, 1, "recovery", "The check reports poor quality",
         "MAST tries recovery 1 of 3."),
        (6, 3000, 2, "recovery", "The warning remains unresolved",
         "MAST tries recovery 2 of 3."),
        (7, 3000, 3, "recovery", "The routine reaches its limit",
         "MAST completes recovery 3 of 3."),
        (7, 3000, 3, "stop", "Recovery is still unverified",
         "Terra stops without claiming success."),
    ]
    elapsed = 0
    result = []
    for index, (scan, duration, completed, stage, obstacle, response) in enumerate(rows):
        result.append({"index": index + 1, "scan": scan, "start_ms": elapsed,
                       "duration_ms": duration, "completed_recovery_checks": completed,
                       "stage": stage, "obstacle": obstacle, "response": response})
        elapsed += duration
    return result


def registered_crop(root: Path, frame: dict, reference_geom: dict) -> Image.Image:
    """Crop every saved image to the candidate scan's physical footprint."""
    geom = frame["geom"]
    assert geom["angle_deg"] == reference_geom["angle_deg"] == 0
    image = Image.open(root / frame["image"]).convert("RGB")
    cx, cy = reference_geom["cx_nm"], reference_geom["cy_nm"]
    width, height = reference_geom["w_nm"], reference_geom["h_nm"]
    left = (cx - width / 2 - geom["cx_nm"]) / geom["w_nm"] + 0.5
    top = (geom["cy_nm"] - cy - height / 2) / geom["h_nm"] + 0.5
    right, bottom = left + width / geom["w_nm"], top + height / geom["h_nm"]
    assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
    return image.resize((SCAN_BOX[2], SCAN_BOX[2]), Image.Resampling.NEAREST,
                        box=(left * image.width, top * image.height,
                             right * image.width, bottom * image.height))


def candidate_marker(image: Image.Image) -> tuple[float, float]:
    """Use the initial bright feature solely to place a fixed explanatory ring."""
    pixels = image.load()
    selected = [(x, y) for y in range(image.height) for x in range(image.width)
                if min(pixels[x, y]) > 225]
    assert selected, "The recorded candidate must be visible."
    return (sum(x for x, _ in selected) / len(selected),
            sum(y for _, y in selected) / len(selected))


def frame_image(trial: dict, story: dict, crops: dict, marker: tuple,
                elapsed_ms: int, total_ms: int, playing: bool = True) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((32, 20), "Terra · recovery before movement", font=font(36, True), fill=NAVY)
    draw.text((32, 72), "EASY / ONE TRIAL / RECORDED SCANS", font=font(18, True), fill=MUTED)
    draw.rounded_rectangle((510, 64, 964, 98), radius=9, fill="#E2F1EA")
    for x in (526, 539):
        draw.polygon(((x, 73), (x + 10, 81), (x, 89)), fill=TEAL)
    draw.text((565, 72), "TIME COMPRESSED / CUTS", font=font(18, True), fill=TEAL)

    # Playback animation is deliberately outside the scientific image.
    draw.rounded_rectangle((988, 21, 1248, 84), radius=14, fill=NAVY)
    if playing:
        pulse = 0.45 + 0.55 * (1 + math.sin(elapsed_ms / 240)) / 2
        r = round(4 + 2 * pulse)
        draw.ellipse((1009 - r, 41 - r, 1009 + r, 41 + r), fill=CYAN)
        draw.text((1025, 29), "PLAYING", font=font(17, True), fill="white")
    else:
        draw.text((1008, 29), "REPLAY", font=font(17, True), fill="white")
    draw.text((1009, 55), f"00:{elapsed_ms // 1000:02d} / 00:{total_ms // 1000:02d}",
              font=font(17), fill="#DCEBE7")

    draw.rounded_rectangle((32, 112, 1248, 556), radius=14, fill="#EDF0ED")
    draw.rounded_rectangle((32, 112, 476, 556), radius=14, fill=NAVY)
    image.paste(crops[story["scan"]], SCAN_BOX[:2])
    px, py = SCAN_BOX[0] + marker[0], SCAN_BOX[1] + marker[1]
    draw.ellipse((px - 33, py - 33, px + 33, py + 33), outline=CYAN, width=3)
    draw.text((60, 136), f"RECORDED SCAN {story['scan']} / 7", font=font(16, True),
              fill="white", stroke_fill=NAVY, stroke_width=2)
    draw.text((60, 508), "Same candidate · no movement", font=font(19, True),
              fill=CYAN, stroke_fill=NAVY, stroke_width=2)

    # The audit is visible during all phases, but kept apart from model actions.
    draw.rounded_rectangle((508, 132, 1220, 178), radius=10, fill="#F3E7D1")
    draw.text((528, 142), "Post-run audit: false alarm", font=font(23, True), fill=AMBER)
    draw.text((528, 201), "RECOVERY CHECKS", font=font(18, True), fill=MUTED)
    draw.text((528, 232), "MAST routine requested by Terra", font=font(23), fill=NAVY)

    complete = story["completed_recovery_checks"]
    for index in range(1, 4):
        x = 548 + (index - 1) * 224
        known = index <= complete
        active = story["stage"] == "recovery" and index == complete
        fill = "#F3E7D1" if known else "#E0E5E2"
        draw.rounded_rectangle((x - 20, 289, x + 171, 423), radius=15, fill=fill)
        draw.text((x, 301), f"Attempt {index}", font=font(20, True), fill=AMBER if known else MUTED)
        draw.text((x, 341), "0" if known else "—", font=font(43, True), fill=AMBER if known else MUTED)
        if active:
            angle = elapsed_ms / 5 % 360
            draw.arc((x + 112, 354, x + 145, 387), angle, angle + 230, fill=AMBER, width=4)
        elif known:
            draw.ellipse((x + 123, 363, x + 137, 377), fill=AMBER)

    draw.text((528, 439), "Tool-reported quality · baseline unknown", font=font(20), fill=MUTED)
    draw.text((528, 478), "No atom movement" if story["stage"] == "stop" else "Movement stays on hold",
              font=font(30, True), fill=NAVY)
    draw.text((528, 523), "The score does not establish a bad tip.", font=font(17), fill=MUTED)

    draw.rounded_rectangle((32, 578, 1248, 674), radius=14, fill="#F3E7D1")
    draw.text((56, 592), story["obstacle"], font=font(28, True), fill=AMBER)
    draw.text((56, 633), "→  " + story["response"], font=font(24), fill=NAVY)
    draw.rounded_rectangle((32, 702, 1248, 708), radius=3, fill="#DCE2DE")
    fraction = min((elapsed_ms + FRAME_MS) / total_ms, 1)
    draw.rounded_rectangle((32, 702, 32 + round(1216 * fraction), 708), radius=3, fill=AMBER)
    return image


def render(root: Path) -> dict:
    root = Path(root)
    evidence = json.loads((root / "replay-evidence.json").read_text(encoding="utf-8"))
    assert evidence["difficulty"] == "easy"
    trial = next(t for t in evidence["trials"] if t["model"] == "Terra")
    assert trial["attempts"] == []
    assert trial["diagnostic_context"]["tool_reported_quality"] == [None, 0, 0, 0]
    assert len(trial["tip_events"]) == 3
    assert all(e["cause"] == "pulse" and e["detail"]["outcome"] == "no_effect"
               for e in trial["tip_events"])
    selected_frames = {f["index"]: f for f in trial["frames"] if f["index"] >= 3}
    for recorded in selected_frames.values():
        assert hashlib.sha256((root / recorded["image"]).read_bytes()).hexdigest() == recorded["sha256"]
        assert recorded["complete"] and recorded["drift_nm"] == [0, 0]
    reference = selected_frames[3]["geom"]
    crops = {index: registered_crop(root, recorded, reference)
             for index, recorded in selected_frames.items()}
    marker = candidate_marker(crops[3])
    story = scenes()
    total_ms = sum(s["duration_ms"] for s in story)
    scene_frames = [frame_image(trial, s, crops, marker, s["start_ms"], total_ms)
                    for s in story]
    # A shared palette prevents quantization from inventing image fluctuations.
    palette_source = Image.new("RGB", (WIDTH, HEIGHT * len(scene_frames)))
    for index, scene_frame in enumerate(scene_frames):
        palette_source.paste(scene_frame, (0, HEIGHT * index))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for elapsed in range(0, total_ms, FRAME_MS):
        current = next(s for s in story if s["start_ms"] <= elapsed < s["start_ms"] + s["duration_ms"])
        frame = frame_image(trial, current, crops, marker, elapsed, total_ms)
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    frames[0].save(root / "terra-replay.gif", save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, disposal=1, optimize=True)
    frame_image(trial, story[-1], crops, marker, total_ms, total_ms,
                playing=False).save(root / "terra-replay.png")
    sheet = Image.new("RGB", (WIDTH, HEIGHT * 3 // 2), BG)
    for index, scene_frame in enumerate(scene_frames):
        sheet.paste(scene_frame.resize((WIDTH // 2, HEIGHT // 2), Image.Resampling.LANCZOS),
                    (index % 2 * WIDTH // 2, index // 2 * HEIGHT // 2))
    sheet.save(root / "terra-replay-storyboard.png")
    manifest = {
        "model": "Terra", "file": "terra-replay.gif", "size": [WIDTH, HEIGHT],
        "duration_ms": total_ms, "frames": len(frames), "frame_duration_ms": FRAME_MS,
        "episode_sha256": trial["episode_sha256"], "scenes": story,
        "annotation_scope": "Recorded scans; explanatory stationary candidate ring and animated playback/status UI. No atom trajectory or image interpolation.",
        "coordinates": {"registration": "Fixed candidate scan footprint for every image", "reference_geom": reference},
        "audit_scope": "False-alarm label is retrospective. Recovery checks are part of one MAST routine. Baseline quality is unknown; later scores are tool-reported, not measured tip health.",
        "playback_scope": "Persistent fast-forward symbol and TIME COMPRESSED / CUTS label indicate editorial time compression; no constant numerical speedup is asserted.",
    }
    (root / "terra-scenes.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with Image.open(root / "terra-replay.gif") as result:
        assert result.n_frames == len(frames)
        assert result.size == (WIDTH, HEIGHT)
        duration = 0
        for index in range(result.n_frames):
            result.seek(index)
            duration += result.info["duration"]
        assert duration == total_ms
    return manifest


if __name__ == "__main__":
    result = render(Path(__file__).resolve().parent)
    print(json.dumps({k: result[k] for k in ("model", "file", "duration_ms", "frames")}, indent=2))
