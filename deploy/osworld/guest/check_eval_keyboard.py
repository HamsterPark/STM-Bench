#!/usr/bin/env python3
"""The keyboard part of the evaluation guest's check: what the model's own keys can do, step by step.

Runs inside the guest as the agent's account (``user``) on the desktop (DISPLAY=:0), after ``stm-start`` has opened the
console, with the system ``python3`` (pyautogui is the one the OSWorld control plane already uses). Called by
``check_eval_guest.sh``; it can also be run alone:

    DISPLAY=:0 python3 check_eval_keyboard.py --shots /tmp/stm-eval-check

Steps, each with a screenshot ``<NN>_<name>.png`` in the shots directory and one output line ``PASS|FAIL <name> ...``:

  1. the console is the active window and is full screen;
  2. Ctrl+Alt+T opens a terminal, which becomes the active window;
  3. the terminal runs ``python3 -c "import numpy, scipy"`` (a marker file proves the command ran and succeeded);
  4. Alt+Tab returns to the console (the terminal stays open behind it);
  5. Alt+Tab again returns to the terminal, which is closed with ``exit``, and the console is active again.

The script never reads the task list, a seed or the private directory. Exit status: 0 if every step passed, 1 otherwise.
Where a tool the check needs is absent (pyautogui, wmctrl, xprop) the step fails with that reason: an unavailable check
is not a pass.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONSOLE_CLASS = "stmbench-gui"           # WM_CLASS the console sets (the delivered image's wmctrl -lx shows it)
TERMINAL_CLASSES = ("gnome-terminal", "xterm", "terminator", "konsole")
MARKER = Path("/tmp/stm_eval_kb.txt")
TERMINAL_COMMAND = f'python3 -c "import numpy, scipy" && echo ok > {MARKER}'

results: list[tuple[bool, str]] = []


def say(ok: bool, name: str, detail: str = "") -> bool:
    results.append((ok, name))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return ok


def sh(cmd: list[str], timeout: float = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{cmd[0]}: {type(exc).__name__}") from None
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}")
    return r.stdout


def windows() -> list[tuple[int, str]]:
    """(window id, WM_CLASS) of every managed window (wmctrl -lx)."""
    found = []
    for line in sh(["wmctrl", "-lx"]).splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 3:
            found.append((int(parts[0], 16), parts[2].lower()))
    return found


def active_window() -> int:
    m = re.search(r"0x[0-9a-fA-F]+", sh(["xprop", "-root", "_NET_ACTIVE_WINDOW"]))
    if not m:
        raise RuntimeError("no active window reported")
    return int(m.group(0), 16)


def active_class() -> str:
    wid = active_window()
    for w, cls in windows():
        if w == wid:
            return cls
    return "?"


def is_console(cls: str) -> bool:
    return CONSOLE_CLASS in cls


def is_terminal(cls: str) -> bool:
    return any(t in cls for t in TERMINAL_CLASSES)


def wait_active(pred, seconds: float = 12.0) -> str:
    """Poll until the active window's class satisfies ``pred``; returns the last class seen."""
    end, cls = time.monotonic() + seconds, "?"
    while time.monotonic() < end:
        try:
            cls = active_class()
        except RuntimeError as exc:
            cls = f"? ({exc})"
        if pred(cls):
            return cls
        time.sleep(0.4)
    return cls


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shots", type=Path, default=Path("/tmp/stm-eval-check"))
    a = ap.parse_args()
    a.shots.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DISPLAY", ":0")
    try:
        import pyautogui
    except Exception as exc:                    # noqa: BLE001  (a missing module or a display that refuses)
        say(False, "keyboard: pyautogui can be imported", type(exc).__name__)
        return 1
    pyautogui.FAILSAFE = False
    counter = [0]

    def shot(name: str) -> None:
        counter[0] += 1
        path = a.shots / f"{counter[0]:02d}_{name}.png"
        try:
            pyautogui.screenshot(str(path))
        except Exception as exc:                # noqa: BLE001
            say(False, f"screenshot {path.name}", type(exc).__name__)

    MARKER.unlink(missing_ok=True)
    time.sleep(1.0)
    shot("console")
    # 1. the console is active and full screen
    try:
        cls = wait_active(is_console, 8)
        fs = "_NET_WM_STATE_FULLSCREEN" in sh(["xprop", "-id", hex(active_window()), "_NET_WM_STATE"])
        say(is_console(cls) and fs, "keyboard: the console is the active window and full screen", f"class {cls}, fullscreen {fs}")
    except RuntimeError as exc:
        say(False, "keyboard: the console is the active window and full screen", str(exc))

    # 2. Ctrl+Alt+T opens a terminal that takes the focus
    before = terminal_ids()
    pyautogui.hotkey("ctrl", "alt", "t")
    cls = wait_active(is_terminal, 15)
    time.sleep(1.0)
    shot("terminal_opened")
    new = terminal_ids() - before
    say(is_terminal(cls) and bool(new), "keyboard: Ctrl+Alt+T opens a terminal in front of the full-screen console", f"active class {cls}")

    # 3. the terminal runs python3 with numpy and scipy
    pyautogui.write(TERMINAL_COMMAND, interval=0.02)
    pyautogui.press("enter")
    end = time.monotonic() + 20
    while time.monotonic() < end and not MARKER.exists():
        time.sleep(0.5)
    time.sleep(0.5)
    shot("terminal_python")
    ok = MARKER.exists() and MARKER.read_text().strip() == "ok"
    say(ok, 'keyboard: the terminal runs python3 -c "import numpy, scipy"', "" if ok else "the marker file was not written")
    MARKER.unlink(missing_ok=True)

    # 4. Alt+Tab returns to the console
    pyautogui.hotkey("alt", "tab")
    cls = wait_active(is_console, 10)
    time.sleep(1.0)
    shot("console_again")
    say(is_console(cls), "keyboard: Alt+Tab returns to the console", f"active class {cls}")

    # 5. Alt+Tab goes back to the terminal; closing it leaves the console in front
    pyautogui.hotkey("alt", "tab")
    cls = wait_active(is_terminal, 10)
    back_in_terminal = is_terminal(cls)
    if back_in_terminal:
        pyautogui.write("exit", interval=0.02)
        pyautogui.press("enter")
    cls2 = wait_active(is_console, 12)
    time.sleep(1.0)
    shot("terminal_closed")
    say(back_in_terminal and is_console(cls2), "keyboard: Alt+Tab reaches the terminal again and closing it leaves the console in front",
        f"classes {cls} then {cls2}")
    return 0 if all(ok for ok, _ in results) else 1


def terminal_ids() -> set[int]:
    """Ids of the terminal windows now open (empty when the window list cannot be read: the step then fails)."""
    try:
        return {w for w, c in windows() if is_terminal(c)}
    except RuntimeError:
        return set()


if __name__ == "__main__":
    sys.exit(main())
