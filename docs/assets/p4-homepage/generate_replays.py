"""Render a trajectory-first replay over recorded, registered scan images.

The atom marker uses only logged positions; the bright tip marker interpolates
command destinations. Scan images change at measurement checkpoints. Markers
are retrospective overlays, not a live instrument feed. Playback pulses are UI
annotations and never change an atom's coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
W, H = 1280, 720
FPS, DURATION = 20, 24.0
BG, NAVY, MUTED = "#FCFBF7", "#182C3D", "#576777"
TEAL, CYAN, AMBER = "#14796B", "#5EF2DB", "#FFD17B"
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
VIEW = (-8.0, 1.5, -14.143333333333333, -9.456666666666667)
PLOT = (32, 112, 900, 444)
OVERVIEW = (-13.8, 3.2, -21.5, -4.5)
MOVE_WINDOWS = [(0.0, 2.2), (5.4, 9.0), (12.0, 14.2), (17.2, 19.4)]

# Editorial replay time; these intervals do not represent instrument speed.
PHASES = [
    (0.0, 2.2, 2, 1, "MANIPULATE", "First attempt", "The tip travels on; the atom follows only partway.", None),
    (2.2, 3.2, 2, 1, "RESCAN", "Did the atom follow?", "Astra takes a new scan to find out.", None),
    (3.2, 5.4, 3, 1, "ADJUST", "Stopped short", "Astra slows the next move 5x.", "SHORT MOVE"),
    (5.4, 9.0, 3, 2, "MANIPULATE", "A slower second attempt", "The atom takes a step back, then follows the tip.", None),
    (9.0, 10.0, 3, 2, "RESCAN", "Check again", "Astra measures where the atom actually stopped.", None),
    (10.0, 12.0, 4, 2, "ADJUST", "Close, but not there", "Astra plans a fine correction.", "NOT THERE YET"),
    (12.0, 14.2, 4, 3, "MANIPULATE", "The tip moves; the atom stays", "Astra attempts a fine correction.", "NO MOVEMENT"),
    (14.2, 15.2, 4, 3, "RESCAN", "Did the correction work?", "Astra checks with another scan.", None),
    (15.2, 17.2, 5, 3, "ADJUST", "Still stuck", "Astra strengthens the pull and keeps moving slowly.", "NO MOVEMENT"),
    (17.2, 19.4, 5, 4, "MANIPULATE", "A stronger fourth attempt", "One step back, then onward to the target.", None),
    (19.4, 20.4, 5, 4, "RESCAN", "One last check", "Astra scans the moved atom and its neighbours.", None),
    (20.4, 24.0, 6, 4, "VERIFY", "At the target", "Astra checks the nearby atoms: both task checks pass.", "VERIFIED"),
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
    tip_motion = json.loads((ROOT / "tip-motion.json").read_text(encoding="utf-8"))
    assert evidence["difficulty"] == "easy" and evidence["seed"] == 0
    assert trial["episode_sha256"] == route["episode_sha256"]
    assert trial["source_replay_sha256"] == route["source_replay_sha256"]
    assert tip_motion["sources"]["episode_sha256"] == route["episode_sha256"]
    assert len(route["moves"]) == 22
    assert [len(a["move_indices"]) for a in route["attempts"]] == [3, 16, 0, 3]
    assert not trial["tip_events"]
    for f in trial["frames"]:
        assert f["complete"] and f["drift_nm"] == [0, 0]
        assert hashlib.sha256((ROOT / f["image"]).read_bytes()).hexdigest() == f["sha256"]
    schedule = []
    # Atom events keep their recorded times; approximate command time is mapped
    # from each tool's wall/sim anchor. Both use the same editorial time window.
    for attempt, tip, window in zip(route["attempts"], tip_motion["attempts"], MOVE_WINDOWS):
        indices = attempt["move_indices"]
        first, last = tip["command_start_sim"], tip["command_end_sim"]
        assert tip["verification_scan"]["duration_sim_seconds"] == 512
        for i in indices:
            move = route["moves"][i]
            assert first <= move["sim"] <= last
            display_time = window[0] + (move["sim"] - first) / (last - first) * (window[1] - window[0])
            schedule.append({"at_seconds": display_time, "move_index": i})
    scans = {f["index"]: crop_scan(f, VIEW, PLOT[2:]) for f in trial["frames"] if f["index"] >= 2}
    overviews = {f["index"]: crop_scan(f, OVERVIEW, (264, 264)) for f in trial["frames"] if f["index"] >= 2}
    return dict(trial=trial, route=route, schedule=schedule, scans=scans,
                overviews=overviews, tip_motion=tip_motion)


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


def tip_at(data, t, attempt):
    """Interpolate command destinations for illustration, never atom coordinates."""
    tip = data["tip_motion"]["attempts"][attempt-1]
    begin, end = MOVE_WINDOWS[attempt-1]
    fraction = min(1.0, max(0.0, (t-begin)/(end-begin)))
    sim = tip["command_start_sim"] + fraction*(tip["command_end_sim"]-tip["command_start_sim"])
    waypoints = tip["waypoints"]
    for a, b in zip(waypoints, waypoints[1:]):
        if a["sim"] <= sim <= b["sim"]:
            q = (sim-a["sim"])/(b["sim"]-a["sim"])
            return [a[k]+q*(b[k]-a[k]) for k in ("x_nm", "y_nm")]
    last = waypoints[-1] if sim >= waypoints[-1]["sim"] else waypoints[0]
    return [last["x_nm"], last["y_nm"]]


def render_frame(data, t, playing=True):
    phase, count, position = state_at(data, t)
    _, _, scan, attempt, badge, obstacle, response, stop_label = phase
    final = badge == "VERIFY"
    im = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(im)
    draw.text((32, 19), "ASTRA  /  Move, check, adjust", font=font(34, True), fill=NAVY)
    draw.text((32, 67), "White light: tip command  /  Teal trail: atom movement", font=font(20), fill=MUTED)
    # This is a playback status indicator, not a clickable control.
    status = "PLAYING" if playing else "KEY FRAME"
    pill(draw, (918, 24, 1248, 76), f"{status}  {int(t):02d}s / 24s", size=20)
    draw.polygon([(933, 91), (943, 85), (933, 79)], fill=TEAL)
    draw.text((951, 80), "LOOPS / EDITED TIME" if playing else "RECORDED REPLAY", font=font(15, True), fill=TEAL)

    im.paste(data["scans"][scan], PLOT[:2])
    pill(draw, (48, 126, 325, 166), f"{attempt} / 4  {badge}", size=18)
    if badge == "RESCAN":
        speed_label = ">> 512x FAST-FORWARD"
    elif badge == "MANIPULATE":
        speed_label = ">> SPED UP / EDITED" if attempt <= 2 else "SLOW MOTION / DETAIL"
    else:
        speed_label = "PAUSE TO READ / CUT"
    pill(draw, (582, 126, 916, 166), speed_label, colour=AMBER, size=17)

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
    draw.ellipse((px-16, py-16, px+16, py+16), outline=colour, width=3)
    draw.ellipse((px-8, py-8, px+8, py+8), fill=colour, outline=NAVY, width=2)
    tip_position = None
    if badge == "MANIPULATE":
        tip_position = tip_at(data, t, attempt)
        tip_light(im, xy(tip_position), t)
        draw = ImageDraw.Draw(im)
    if backstep:
        pill(draw, (px-82, py-73, px+92, py-34), "BACKSTEP", colour=AMBER, size=17)
    elif stop_label:
        text_width = draw.textlength(stop_label, font=font(18, True))
        bx = max(218, min(px-text_width/2-12, 834-text_width))
        pill(draw, (bx, py-88, bx+text_width+24, py-45), stop_label, colour=CYAN if final else AMBER)

    overview_panel(im, draw, data, scan, points, position, final, tip_position)
    draw.text((49, 531), f"LAST MEASURED IMAGE: SCAN {scan}", font=font(16), fill="#E1E8EF", stroke_width=2, stroke_fill=NAVY)
    if badge == "RESCAN":
        progress = (t-phase[0])/(phase[1]-phase[0])
        draw.rounded_rectangle((49, 469, 565, 510), radius=8, fill=NAVY)
        draw.text((62, 477), "RESCAN  8 min 32 s → 1 s", font=font(20, True), fill=AMBER)
        draw.rectangle((49, 515, 565, 521), fill="#31485A")
        draw.rectangle((49, 515, 49+516*progress, 521), fill=AMBER)

    draw.rounded_rectangle((32, 578, 1248, 668), radius=12, fill="#E8F4EE" if final else "#F0F1EE")
    draw.text((50, 590), obstacle, font=font(29, True), fill=TEAL if final else NAVY)
    if draw.textlength(response, font=font(22)) > 1176:
        raise ValueError("Response caption too long")
    draw.text((50, 632), "→ " + response, font=font(22), fill=MUTED)
    draw.text((32, 679), "Easy / one trial / edited timing / tip commands and atom records shown as post-run overlays", font=font(16), fill=MUTED)
    draw.rounded_rectangle((32, 705, 1248, 711), radius=3, fill="#DDE4E1")
    bar_end = 32 + 1216 * min((t + 1/FPS) / DURATION, 1)
    draw.rounded_rectangle((32, 705, bar_end, 711), radius=3, fill=TEAL)
    return im


def overview_panel(im, draw, data, scan, points, position, final, tip_position):
    """Keep the same neighbourhood and the main-view footprint visible in every frame."""
    x, y = 970, 181
    draw.rounded_rectangle((956, 112, 1248, 556), radius=12, fill=NAVY)
    draw.text((x, 128), "OVERVIEW", font=font(22, True), fill="#FFFFFF")
    draw.text((x, 156), f"Last measured scan: {scan}", font=font(16), fill="#DCE7EF")
    im.paste(data["overviews"][scan], (x, y))
    left, right, low, high = OVERVIEW
    def point(pos):
        return x + (pos[0]-left)/(right-left)*264, y + (high-pos[1])/(high-low)*264
    tl, br = point((VIEW[0], VIEW[3])), point((VIEW[1], VIEW[2]))
    draw.rectangle((*tl, *br), outline="#FFFFFF", width=2)
    draw.text((tl[0]+3, tl[1]-15), "MAIN VIEW", font=font(11, True), fill="#FFFFFF", stroke_width=1, stroke_fill=NAVY)
    for atom in data["route"]["initial_atoms"]:
        if atom["id"] == data["route"]["selected_atom_id"]:
            continue
        ax, ay = point((atom["x"], atom["y"]))
        draw.ellipse((ax-9, ay-9, ax+9, ay+9), outline=CYAN if final else "#B7C6D2", width=2)
    count = len(points)-1
    route_points = [point(data["route"]["target"]["start"])] + [point((m["x"], m["y"])) for m in data["route"]["moves"][:count]]
    if count:
        draw.line(route_points, fill=CYAN, width=3)
    ax, ay = point(position)
    draw.ellipse((ax-5, ay-5, ax+5, ay+5), fill=CYAN, outline=NAVY, width=1)
    gx, gy = point(data["route"]["target"]["site"])
    draw.ellipse((gx-9, gy-9, gx+9, gy+9), outline=CYAN, width=2)
    if tip_position is not None:
        tx, ty = point(tip_position)
        draw.ellipse((tx-6, ty-6, tx+6, ty+6), fill="#FFFFFF", outline=NAVY, width=2)
    draw.text((x, 464), "3 neighbours unchanged" if final else "Keep neighbours in place", font=font(16), fill=CYAN if final else "#DCE7EF")
    draw.text((x, 492), "2 / 2 CHECKS VERIFIED" if final else "Same area throughout", font=font(16, True), fill=CYAN if final else "#FFFFFF")
    draw.text((x, 524), "White box = close-up", font=font(14), fill="#DCE7EF")


def tip_light(im, position, t):
    """An explanatory bright cursor, not an atom or independent tip measurement."""
    px, py = position
    glow = Image.new("RGBA", (96, 96))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((20, 20, 76, 76), fill=(255, 209, 80, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(10))
    im.paste(glow, (round(px)-48, round(py)-48), glow)
    draw = ImageDraw.Draw(im)
    radius = 26 + 3*math.sin(t*math.pi*3)
    draw.ellipse((px-radius, py-radius, px+radius, py+radius), outline=NAVY, width=8)
    draw.ellipse((px-radius, py-radius, px+radius, py+radius), outline="#FFE5A1", width=3)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.line((px+dx*13, py+dy*13, px+dx*32, py+dy*32), fill=NAVY, width=7)
        draw.line((px+dx*13, py+dy*13, px+dx*32, py+dy*32), fill="#FFFFFF", width=3)
    draw.ellipse((px-13, py-13, px+13, py+13), fill=NAVY)
    draw.ellipse((px-9, py-9, px+9, py+9), fill="#FFFFFF", outline="#FFDA76", width=2)
    draw.text((px, py+77), "TIP", font=font(18, True), fill="#FFFFFF", anchor="mt", stroke_width=2, stroke_fill=NAVY)


def render_astra():
    data = load_evidence()
    preview_times = [0.55, 2.7, 4.0, 6.0, 9.5, 11.0, 13.0, 14.7, 16.0, 18.5, 19.8, 23.0]
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
    # Four full-width views retain the tip/atom distinction without animation.
    sheet = Image.new("RGB", (W, H*4), BG)
    for i, t in enumerate([1.75, 7.5, 13.0, 23.9]):
        sheet.paste(render_frame(data, t, playing=False), (0, i*H))
    sheet.save(ROOT / "astra-replay-storyboard.png")
    metadata = {
        "schema_version": 3, "model": "Astra", "duration_seconds": DURATION, "fps": FPS,
        "view_bounds_nm": VIEW, "source_position_data": "replay-data.json",
        "overview_bounds_nm": OVERVIEW, "overview_always_visible": True,
        "source_tip_commands": "tip-motion.json", "manipulation_windows_seconds": MOVE_WINDOWS,
        "playback_timing": "Manipulation command spans use edited speed; small corrections are slowed for visibility. Each 512-second verification scan is condensed to exactly one second and labelled 512x. Reading pauses and cuts have no physical timescale.",
        "source_scan_data": "replay-evidence.json", "movement_schedule": data["schedule"],
        "phases": [dict(start_s=p[0], end_s=p[1], last_scan=p[2], attempt=p[3],
                        badge=p[4], obstacle=p[5], response=p[6]) for p in PHASES],
        "overlay_note": "Teal atom positions are post-episode truth, with no interpolation. White tip light interpolates successful command destinations, not measured tip positions. The agent reacted to subsequent scans, not hidden hops. No added drift or synthetic noise.",
    }
    (ROOT / "replay-scenes.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"model": "Astra", "frames": len(quantized), "duration_seconds": DURATION,
            "size_bytes": (ROOT / "astra-replay.gif").stat().st_size}


if __name__ == "__main__":
    print(json.dumps(render_astra(), indent=2))
    from render_terra import render
    result = render(ROOT)
    print(json.dumps({k: result[k] for k in ("model", "duration_ms", "frames")}, indent=2))
