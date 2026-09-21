"""Render a trajectory-first replay over recorded, registered scan images.

The atom marker uses only logged positions. Scan images change at measurement
checkpoints; the marker is a retrospective overlay, not a live instrument feed.
Playback pulses are UI annotations and never change an atom's coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H = 1280, 720
FPS, DURATION = 10, 24.0
BG, NAVY, MUTED = "#FCFBF7", "#182C3D", "#576777"
TEAL, CYAN, AMBER = "#14796B", "#5EF2DB", "#FFD17B"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
VIEW = (-8.0, 1.5, -13.534375, -10.065625)
PLOT = (32, 112, 1216, 444)

# Editorial replay time; these intervals do not represent instrument speed.
PHASES = [
    (0.0, 3.2, 2, 1, "TRY 1", "A move is requested", "The atom moves only partway.", None),
    (3.2, 6.0, 3, 1, "CHECK", "Stopped short", "Astra scans again, then slows the next move 5x.", "SHORT MOVE"),
    (6.0, 10.0, 3, 2, "TRY 2", "A slower second attempt", "The route includes a brief backwards step.", None),
    (10.0, 12.5, 4, 2, "CHECK", "Close, but not there", "Astra measures the gap and tries a fine correction.", "NOT THERE YET"),
    (12.5, 15.5, 4, 3, "TRY 3", "The correction does nothing", "Astra checks with another scan.", "NO MOVEMENT"),
    (15.5, 18.0, 5, 3, "CHECK", "Still stuck", "Astra strengthens the pull and keeps moving slowly.", "NO MOVEMENT"),
    (18.0, 20.5, 5, 4, "TRY 4", "A stronger fourth attempt", "One step back, then onward to the target.", None),
    (20.5, 24.0, 6, 4, "VERIFY", "At the target", "Astra checks the nearby atoms: both task checks pass.", "VERIFIED"),
]


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def xy(position):
    left, right, low, high = VIEW
    x, y, width, height = PLOT
    return x + (position[0] - left) / (right - left) * width, y + (high - position[1]) / (high - low) * height


def crop_scan(frame, bounds, size):
    """Register scan renderings in sample coordinates; these trials have no drift."""
    raw = Image.open(ROOT / frame["image"]).convert("RGB")
    g = frame["geom"]
    assert g["angle_deg"] == 0 and frame["drift_nm"] == [0, 0]
    left, right, low, high = bounds
    box = ((left - g["cx_nm"]) / g["w_nm"] * raw.width + raw.width / 2,
           (g["cy_nm"] - high) / g["h_nm"] * raw.height + raw.height / 2,
           (right - g["cx_nm"]) / g["w_nm"] * raw.width + raw.width / 2,
           (g["cy_nm"] - low) / g["h_nm"] * raw.height + raw.height / 2)
    assert 0 <= box[0] < box[2] <= raw.width and 0 <= box[1] < box[3] <= raw.height
    return raw.resize(size, Image.Resampling.BILINEAR, box=box)


def load_evidence():
    evidence = json.loads((ROOT / "replay-evidence.json").read_text(encoding="utf-8"))
    positions = json.loads((ROOT / "replay-data.json").read_text(encoding="utf-8"))
    trial = next(t for t in evidence["trials"] if t["model"] == "Astra")
    route = next(t for t in positions["trials"] if t["model"] == "Astra")
    assert evidence["difficulty"] == "easy" and evidence["seed"] == 0
    assert trial["episode_sha256"] == route["episode_sha256"]
    assert trial["source_replay_sha256"] == route["source_replay_sha256"]
    assert len(route["moves"]) == 22
    assert [len(a["move_indices"]) for a in route["attempts"]] == [3, 16, 0, 3]
    assert not trial["tip_events"]
    for f in trial["frames"]:
        assert f["complete"] and f["drift_nm"] == [0, 0]
        assert hashlib.sha256((ROOT / f["image"]).read_bytes()).hexdigest() == f["sha256"]
    schedule = []
    # Preserve event order and relative timing inside each compressed move.
    for attempt, window in zip(route["attempts"], [(0.25, 1.45), (6.15, 9.35), None, (18.15, 19.45)]):
        indices = attempt["move_indices"]
        if not indices:
            assert window is None
            continue
        first = route["moves"][indices[0]]["sim"]
        last = route["moves"][indices[-1]]["sim"]
        for i in indices:
            move = route["moves"][i]
            display_time = window[0] + (move["sim"] - first) / (last - first) * (window[1] - window[0])
            schedule.append({"at_seconds": display_time, "move_index": i})
    scans = {f["index"]: crop_scan(f, VIEW, PLOT[2:]) for f in trial["frames"] if f["index"] >= 2}
    neighbour_bounds = (-13.6, -0.4, -20.0, -7.4)
    neighbour_image = crop_scan(trial["frames"][-1], neighbour_bounds, (211, 202))
    return dict(trial=trial, route=route, schedule=schedule, scans=scans,
                neighbour_bounds=neighbour_bounds, neighbour_image=neighbour_image)


def pill(draw, box, text, colour=CYAN, fill=NAVY, size=18):
    draw.rounded_rectangle(box, radius=9, fill=fill)
    draw.text((box[0] + 12, box[1] + 8), text, font=font(size, True), fill=colour)


def dashed_line(draw, start, end, colour="#B0C7D4"):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    for d in range(0, int(length), 22):
        e = min(d + 10, length)
        draw.line((start[0] + dx*d/length, start[1] + dy*d/length,
                   start[0] + dx*e/length, start[1] + dy*e/length), fill=colour, width=2)


def state_at(data, t):
    phase = next(p for p in PHASES if p[0] <= t < p[1])
    visible = [m for m in data["schedule"] if m["at_seconds"] <= t]
    count = len(visible)
    moves = data["route"]["moves"]
    position = data["route"]["target"]["start"] if not count else [moves[count-1]["x"], moves[count-1]["y"]]
    return phase, count, position


def render_frame(data, t, playing=True):
    phase, count, position = state_at(data, t)
    _, _, scan, attempt, badge, obstacle, response, stop_label = phase
    final = badge == "VERIFY"
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)
    draw.text((32, 19), "ASTRA  /  Move, check, adjust", font=font(34, True), fill=NAVY)
    draw.text((32, 67), "Recorded atom trajectory  /  Four attempts, six scans", font=font(21), fill=MUTED)
    # This is a playback status indicator, not a clickable control.
    status = "PLAYING" if playing else "KEY FRAME"
    pill(draw, (918, 24, 1248, 76), f"{status}  {int(t):02d}s / 24s", size=20)
    draw.polygon([(933, 91), (943, 85), (933, 79)], fill=TEAL)
    draw.text((951, 80), "AUTOPLAY / LOOPS" if playing else "RECORDED REPLAY", font=font(15, True), fill=TEAL)

    im.paste(data["scans"][scan], PLOT[:2])
    pill(draw, (48, 126, 206, 166), badge, size=18)
    pill(draw, (859, 126, 1232, 166), f"LAST MEASURED IMAGE: SCAN {scan}", colour="#DCE7EF", size=16)

    start = xy(data["route"]["target"]["start"])
    target = xy(data["route"]["target"]["site"])
    dashed_line(draw, start, target)
    # The target and route are explanatory post-run overlays; the agent never
    # received these true positions while performing the experiment.
    draw.ellipse((start[0]-20, start[1]-20, start[0]+20, start[1]+20), outline="#AAB9C8", width=2)
    draw.text((start[0], start[1]+35), "START", font=font(19, True), fill="#E1E8EF", anchor="mt", stroke_width=2, stroke_fill=NAVY)
    draw.ellipse((target[0]-25, target[1]-25, target[0]+25, target[1]+25), outline=CYAN if final else "#FFFFFF", width=3)
    draw.text((target[0], target[1]+42), "TARGET", font=font(19, True), fill=CYAN if final else "#FFFFFF", anchor="mt", stroke_width=2, stroke_fill=NAVY)

    points = [start] + [xy((m["x"], m["y"])) for m in data["route"]["moves"][:count]]
    if len(points) > 1:
        draw.line(points, fill=NAVY, width=10, joint="curve")
        draw.line(points, fill=CYAN, width=5, joint="curve")
        for point in points[:-1]:
            draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill=CYAN)
    px, py = xy(position)
    backstep = False
    if count:
        last_event_time = data["schedule"][count-1]["at_seconds"]
        backstep = count > 1 and points[-1][0] < points[-2][0] and t - last_event_time < 0.55
    colour = AMBER if backstep else CYAN
    radius = 21 + 4*math.sin(t * 2*math.pi*1.6)
    draw.ellipse((px-radius, py-radius, px+radius, py+radius), outline=colour, width=3)
    draw.ellipse((px-10, py-10, px+10, py+10), fill=colour, outline=NAVY, width=2)
    if backstep:
        pill(draw, (px-82, py-73, px+92, py-34), "BACKSTEP", colour=AMBER, size=17)
    elif stop_label:
        text_width = draw.textlength(stop_label, font=font(18, True))
        bx = max(218, min(px-text_width/2-12, 834-text_width))
        pill(draw, (bx, py-88, bx+text_width+24, py-45), stop_label, colour=CYAN if final else AMBER)

    if final:
        neighbour_check(im, draw, data)
    else:
        # Keep the route visually dominant, with only a compact attempt counter.
        draw.text((1224, 509), f"ATTEMPT {attempt} / 4", font=font(17, True), fill="#DFE7EF", anchor="rt", stroke_width=2, stroke_fill=NAVY)
    draw.text((49, 531), "Recorded path over saved scans", font=font(16), fill="#E1E8EF", stroke_width=2, stroke_fill=NAVY)

    draw.rounded_rectangle((32, 578, 1248, 668), radius=12, fill="#E8F4EE" if final else "#F0F1EE")
    draw.text((50, 590), obstacle, font=font(29, True), fill=TEAL if final else NAVY)
    if draw.textlength(response, font=font(22)) > 1176:
        raise ValueError("Response caption too long")
    draw.text((50, 632), "→ " + response, font=font(22), fill=MUTED)
    draw.text((32, 679), "Easy setting / single trial / time condensed / highlights are explanatory overlays", font=font(16), fill=MUTED)
    draw.rounded_rectangle((32, 705, 1248, 711), radius=3, fill="#DDE4E1")
    bar_end = 32 + 1216 * min((t + 1/FPS) / DURATION, 1)
    draw.rounded_rectangle((32, 705, bar_end, 711), radius=3, fill=TEAL)
    return im


def neighbour_check(im, draw, data):
    x, y = 1003, 235
    draw.rounded_rectangle((x-12, y-39, x+223, y+257), radius=12, fill=NAVY)
    draw.text((x, y-29), "NEIGHBOUR CHECK", font=font(17, True), fill=CYAN)
    im.paste(data["neighbour_image"], (x, y))
    left, right, low, high = data["neighbour_bounds"]
    for atom in data["route"]["initial_atoms"]:
        if atom["id"] == data["route"]["selected_atom_id"]:
            continue
        ax = x + (atom["x"]-left)/(right-left)*211
        ay = y + (high-atom["y"])/(high-low)*202
        draw.ellipse((ax-10, ay-10, ax+10, ay+10), outline=CYAN, width=2)
    draw.text((x, y+210), "3 neighbours unchanged", font=font(14), fill=CYAN)
    draw.text((x, y+232), "2 / 2 CHECKS VERIFIED", font=font(14, True), fill=CYAN)


def render_astra():
    data = load_evidence()
    preview_times = [0.0, 3.5, 7.0, 10.5, 13.5, 16.0, 18.5, 23.0]
    previews = [render_frame(data, t) for t in preview_times]
    # A shared palette built from representative scenes avoids artificial colour
    # changes. Generate one RGB frame at a time to limit memory use.
    palette_source = Image.new("RGB", (W, H*len(previews)))
    for i, im in enumerate(previews):
        palette_source.paste(im, (0, H*i))
    palette = palette_source.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
    quantized = [render_frame(data, i/FPS).quantize(palette=palette, dither=Image.Dither.NONE)
                 for i in range(int(DURATION*FPS))]
    quantized[0].save(ROOT / "astra-replay.gif", save_all=True, append_images=quantized[1:],
                      duration=int(1000/FPS), loop=0, disposal=1, optimize=True)
    render_frame(data, 23.9, playing=False).save(ROOT / "astra-replay.png")
    # Four full-width checkpoints are readable without playing the animation.
    sheet = Image.new("RGB", (W, H*4), BG)
    for i, t in enumerate([3.5, 10.5, 16.0, 23.9]):
        sheet.paste(render_frame(data, t, playing=False), (0, i*H))
    sheet.save(ROOT / "astra-replay-storyboard.png")
    metadata = {
        "schema_version": 2, "model": "Astra", "duration_seconds": DURATION, "fps": FPS,
        "view_bounds_nm": VIEW, "source_position_data": "replay-data.json",
        "source_scan_data": "replay-evidence.json", "movement_schedule": data["schedule"],
        "phases": [dict(start_s=p[0], end_s=p[1], last_scan=p[2], attempt=p[3],
                        badge=p[4], obstacle=p[5], response=p[6]) for p in PHASES],
        "overlay_note": "Positions are post-episode truth. The agent reacted to subsequent scans, not to hidden hops. No position interpolation, added drift or synthetic noise.",
    }
    (ROOT / "replay-scenes.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"model": "Astra", "frames": len(quantized), "duration_seconds": DURATION,
            "size_bytes": (ROOT / "astra-replay.gif").stat().st_size}


if __name__ == "__main__":
    print(json.dumps(render_astra(), indent=2))
    from render_terra import render
    result = render(ROOT)
    print(json.dumps({k: result[k] for k in ("model", "duration_ms", "frames")}, indent=2))
