"""The console's launcher and ``stm-start`` of the virtual machine (``deploy/osworld/guest/``).

What is pinned here: a task id is parsed strictly and an unknown or malformed one never reaches the console; the seed
travels from ``tasks.json`` to the console's standard input and appears in no argument, no environment variable, no
error text and no file the agent's account can read; the FIFO protocol (a request, the ready signal, the same id
again, another id, a console that dies before its window is shown and a retry); ``stm-start``'s exit statuses; and,
with Qt, the real sealed console started through the launcher (offscreen). The service account, systemd and the
desktop session are not exercised here (``deploy/osworld/guest/README`` and the workflow list what they need).

POSIX only (a FIFO, ``bash``, process groups).
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "posix", reason="a FIFO and process groups: POSIX only")

ROOT = Path(__file__).resolve().parents[1]
GUEST = ROOT / "deploy" / "osworld" / "guest"
P4 = "P4_atom_positioning_cu111"
SEED = 913275
TASK = "stm_p4_atom_07"


def _load():
    spec = importlib.util.spec_from_file_location("stm_launcher", GUEST / "stm_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stm_launcher"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load()

#: a stand-in for the console: reads the spec line, records what it was given, shows its "window" after a delay
FAKE = r'''
import json, os, sys, time
argv = sys.argv[1:]
ready = argv[argv.index("--ready-file") + 1]
seen = os.environ.get("FAKE_SEEN")
line = sys.stdin.readline()
if seen:
    with open(seen, "w") as fh:
        json.dump({"argv": sys.argv, "env": dict(os.environ), "has_spec": bool(line.strip())}, fh)
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "die":
    sys.exit(3)
time.sleep(float(os.environ.get("FAKE_DELAY", "0.1")))
if mode != "never":
    open(ready, "w").write("ready\n")
time.sleep(60)
'''


@pytest.fixture
def box(tmp_path):
    run, private, export = tmp_path / "run", tmp_path / "private", tmp_path / "export"
    private.mkdir(mode=0o700)
    export.mkdir()
    fake = tmp_path / "fake_console.py"
    fake.write_text(FAKE, encoding="utf-8")
    (private / "tasks.json").write_text(json.dumps({TASK: {"scenario": P4, "seed": SEED},
                                                    "stm_p4_atom_08": {"scenario": P4, "seed": SEED + 1}}))
    cfg = L.Config(run_dir=run, private_dir=private, export_dir=export, home=tmp_path / "home",
                   console_cmd=[sys.executable, str(fake)], extra_env={"FAKE_SEEN": str(tmp_path / "seen.json")})
    return cfg


@pytest.fixture
def running(box):
    """A launcher serving in a thread; yields it with its config."""
    launcher = L.Launcher(box)
    launcher.setup()
    stop = threading.Event()
    th = threading.Thread(target=launcher.serve, args=(stop, 0.02), daemon=True)
    th.start()
    yield launcher
    stop.set()
    th.join(5)
    launcher.close()


def write_request(cfg, text: bytes) -> None:
    fd = os.open(cfg.fifo, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, text)
    finally:
        os.close(fd)


def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ── parsing ──
@pytest.mark.parametrize("raw", [b"stm_p4_atom_01", b"stm_p4_atom_01\n", b"  stm_p4_atom_01 \r\n", b"a", b"a" * 48])
def test_a_task_id_is_lower_case_letters_digits_and_underscores(raw):
    assert L.parse_request(raw) == raw.strip().decode()


@pytest.mark.parametrize("raw", [b"", b"\n", b"   ", b"Stm_P4", b"1abc", b"_abc", b"a b", b"a-b", b"a/b", b"../x",
                                 b"a" * 49, b"a\x00b", "é".encode(), b"stm_p4;ls", b"$(id)", b"x" * 1000])
def test_anything_else_is_refused_without_repeating_it(raw):
    with pytest.raises(L.RequestError) as exc:
        L.parse_request(raw)
    echoed = raw.strip().decode("ascii", "ignore")
    assert not echoed or echoed not in str(exc.value)


def test_load_task_gives_scenario_and_seed_and_refuses_the_rest(tmp_path):
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps({"a_1": {"scenario": P4, "seed": SEED}, "b_1": {"scenario": P4, "seed": True},
                             "c_1": {"scenario": P4}, "d_1": {"scenario": P4, "seed": -1},
                             "e_1": {"scenario": P4, "seed": SEED, "extra": 1}, "f_1": [SEED]}))
    assert L.load_task(f, "a_1") == (P4, SEED)
    for bad in ("b_1", "c_1", "d_1", "e_1", "f_1", "zzz"):
        with pytest.raises(L.RequestError) as exc:
            L.load_task(f, bad)
        assert str(SEED) not in str(exc.value)
    f.write_text("not json")
    with pytest.raises(L.RequestError):
        L.load_task(f, "a_1")
    with pytest.raises(L.RequestError):
        L.load_task(tmp_path / "missing.json", "a_1")


# ── the FIFO protocol ──
def test_setup_makes_a_fifo_the_agent_account_can_write_and_nothing_else_writable(box):
    launcher = L.Launcher(box)
    launcher.setup()
    try:
        mode = stat.S_IMODE(os.stat(box.fifo).st_mode)
        assert stat.S_ISFIFO(os.stat(box.fifo).st_mode) and mode == 0o622
        assert stat.S_IMODE(os.stat(box.run_dir).st_mode) == 0o755
        assert stat.S_IMODE(os.stat(box.run_dir / "qt").st_mode) == 0o700
    finally:
        launcher.close()


def test_a_request_starts_the_console_with_the_seed_on_stdin_only(running, tmp_path, monkeypatch):
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)    # the service sets it; xcb is the default otherwise
    cfg = running.cfg
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.ready.exists)
    assert cfg.task.read_text().strip() == TASK
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["has_spec"] is True
    assert str(SEED) not in json.dumps(seen["argv"]) and str(SEED) not in json.dumps(seen["env"])
    assert "--sealed" in seen["argv"] and seen["argv"][seen["argv"].index("--lang") + 1] == "en"
    assert seen["argv"][seen["argv"].index("--ready-file") + 1] == str(cfg.ready)
    assert seen["env"]["HOME"] == str(cfg.home) and seen["env"]["QT_QPA_PLATFORM"] == "xcb"
    assert not cfg.error.exists()
    # nothing the agent's account can read names the seed
    for p in cfg.run_dir.rglob("*"):
        if p.is_file():
            assert str(SEED) not in p.read_text(errors="replace")


def test_the_launcher_does_not_pass_its_own_secrets_on(running, tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_PRIVATE_THING", "leak")
    write_request(running.cfg, (TASK + "\n").encode())
    assert wait_for(running.cfg.ready.exists)
    assert "SOME_PRIVATE_THING" not in json.loads((tmp_path / "seen.json").read_text())["env"]


@pytest.mark.parametrize("raw", [b"nope_1\n", b"Bad Id\n", b"\n", b"a" * 300 + b"\n", b"x;y\n"])
def test_an_unknown_or_malformed_id_starts_nothing_and_says_so_without_the_seed(running, raw, tmp_path):
    cfg = running.cfg
    write_request(cfg, raw)
    assert wait_for(cfg.error.exists)
    assert not cfg.ready.exists() and not cfg.task.exists() and running.child is None
    assert not (tmp_path / "seen.json").exists()
    assert str(SEED) not in cfg.error.read_text()
    # and a good request afterwards still works
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.ready.exists)


def test_the_same_id_again_changes_nothing_and_another_id_is_refused(running):
    cfg = running.cfg
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.ready.exists)
    pid = running.child.pid
    write_request(cfg, (TASK + "\n").encode())
    time.sleep(0.3)
    assert running.child is not None and running.child.pid == pid and not cfg.error.exists()
    write_request(cfg, b"stm_p4_atom_08\n")
    assert wait_for(cfg.error.exists)
    assert "already started" in cfg.error.read_text()
    assert running.child.pid == pid and cfg.task.read_text().strip() == TASK


def test_a_console_that_dies_before_its_window_is_cleaned_up_and_the_request_can_be_repeated(running, tmp_path):
    cfg = running.cfg
    cfg.extra_env["FAKE_MODE"] = "die"
    (cfg.export_dir / "half.json").write_text("{}")
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.error.exists)
    assert "before its window" in cfg.error.read_text() and str(SEED) not in cfg.error.read_text()
    assert not cfg.task.exists() and not list(cfg.export_dir.iterdir())
    cfg.extra_env["FAKE_MODE"] = "ok"
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.ready.exists)
    assert cfg.task.read_text().strip() == TASK


def test_after_the_window_was_shown_the_export_is_never_touched_and_no_second_run_starts(running):
    cfg = running.cfg
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.ready.exists)
    (cfg.export_dir / "actions.jsonl").write_text("{}\n")
    os.killpg(running.child.pid, 9)                       # the console is killed from outside
    assert wait_for(cfg.error.exists)
    assert (cfg.export_dir / "actions.jsonl").exists() and "ended" in cfg.error.read_text()
    cfg.error.unlink()
    write_request(cfg, (TASK + "\n").encode())
    assert wait_for(cfg.error.exists) and "already started" in cfg.error.read_text()
    assert (cfg.export_dir / "actions.jsonl").exists() and running.child is None


def test_a_restarted_service_does_not_start_a_second_episode(box):
    (box.private_dir / "run").mkdir()
    (box.private_dir / "run" / "ledger.jsonl").write_text("{}\n")
    launcher = L.Launcher(box)
    launcher.setup()
    try:
        assert launcher.feed((TASK + "\n").encode()) == ["refused"]
        assert launcher.child is None and "already started" in box.error.read_text()
    finally:
        launcher.close()


def test_a_request_without_a_newline_waits_and_an_overlong_one_is_dropped(box):
    launcher = L.Launcher(box)
    launcher.setup()
    try:
        assert launcher.feed(b"stm_p4") == []
        assert launcher.feed(b"_atom_07\n") == ["started"]
        launcher.stop_console()
        launcher.child = None
        launcher.task_id = None
        assert launcher.feed(b"x" * 600) == []
        assert launcher.feed(b"yy\n") == ["refused"]      # the rest of the long line
        assert "too long" in box.error.read_text()
    finally:
        launcher.close()


# ── stm-start ──
def run_start(cfg, *args, wait="5", extra_env=None):
    env = {**os.environ, "STM_BENCH_RUN_DIR": str(cfg.run_dir), "STM_START_WAIT_S": wait, "STM_START_NO_XHOST": "1",
           **(extra_env or {})}
    return subprocess.run(["bash", str(GUEST / "stm-start"), *args], capture_output=True, text=True, env=env, timeout=60)


def test_stm_start_waits_for_the_console_and_exits_0(running):
    r = run_start(running.cfg, TASK)
    assert r.returncode == 0 and "up" in r.stdout and str(SEED) not in r.stdout + r.stderr
    again = run_start(running.cfg, TASK)                  # a second call: already up
    assert again.returncode == 0
    assert run_start(running.cfg, "stm_p4_atom_08").returncode == 1


@pytest.mark.parametrize("args", [(), ("a", "b"), ("Bad",), ("1x",), ("a;b",), ("",)])
def test_stm_start_checks_its_usage(running, args):
    r = run_start(running.cfg, *args)
    assert r.returncode == 2 and "usage" in r.stderr
    assert not running.cfg.ready.exists() and running.child is None


def test_stm_start_prints_the_reason_of_a_refusal_and_never_the_seed(running):
    r = run_start(running.cfg, "nope_1")
    assert r.returncode == 1 and "unknown task id" in r.stderr and str(SEED) not in r.stdout + r.stderr


def test_stm_start_says_when_no_launcher_runs(box):
    t0 = time.monotonic()
    r = run_start(box, TASK, wait="2")
    assert r.returncode == 1 and "not running" in r.stderr and time.monotonic() - t0 < 10


def test_stm_start_waits_for_a_launcher_that_is_still_starting(box):
    """Right after boot the service is active a moment before its FIFO exists (a race seen in CI)."""
    launcher = L.Launcher(box)
    stop = threading.Event()

    def late():
        time.sleep(1.0)
        launcher.setup()
        launcher.serve(stop, 0.02)
    th = threading.Thread(target=late, daemon=True)
    th.start()
    try:
        r = run_start(box, TASK, wait="20")
        assert r.returncode == 0 and "up" in r.stdout, r.stderr
    finally:
        stop.set()
        th.join(5)
        launcher.close()


def test_stm_start_gives_up_after_its_wait(running):
    running.cfg.extra_env["FAKE_MODE"] = "never"
    t0 = time.monotonic()
    r = run_start(running.cfg, TASK, wait="2")
    assert r.returncode == 3 and "did not come up" in r.stderr and time.monotonic() - t0 < 10


def test_an_old_error_does_not_fail_a_new_request(running):
    cfg = running.cfg
    assert run_start(cfg, "nope_1").returncode == 1
    assert cfg.error.exists()
    assert run_start(cfg, TASK).returncode == 0


# ── the installed files ──
def test_the_unit_and_the_autostart_entry_agree_with_the_launcher_defaults():
    import configparser
    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.optionxform = str
    unit.read(GUEST / "stm-bench.service")
    svc = unit["Service"]
    assert svc["User"] == "stm" and svc["RuntimeDirectory"] == "stm-bench" and svc["UMask"] == "0022"
    assert svc["ExecStart"] == "/opt/stm-bench/venv/bin/python /opt/stm-bench/guest/stm_launcher.py"
    assert unit["Install"]["WantedBy"] == "multi-user.target"
    assert L.Config().run_dir == Path("/run/stm-bench") and L.Config().private_dir == Path("/var/lib/stm-bench/private")
    assert L.Config().export_dir == Path("/var/lib/stm-bench/export")
    desk = configparser.ConfigParser(interpolation=None)
    desk.read(GUEST / "stm-bench-xhost.desktop")
    assert desk["Desktop Entry"]["Exec"] == "xhost +SI:localuser:stm"


def test_the_shell_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(GUEST / "stm-start")], capture_output=True).returncode == 0


def test_the_installer_scripts_are_valid_bash_and_the_package_list_is_plain():
    import re
    for name in ("install.sh", "stm-bench-reset", "stm-lockdown", "pins.sh", "stm-start"):
        assert subprocess.run(["bash", "-n", str(GUEST / name)], capture_output=True).returncode == 0, name
    pkgs = [ln.strip() for ln in (GUEST / "apt-packages.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert len(pkgs) == len(set(pkgs)) and all(re.fullmatch(r"[a-z0-9][a-z0-9+.-]+", p) for p in pkgs)
    assert "libxcb-cursor0" in pkgs and "x11-xserver-utils" in pkgs      # Qt >= 6.5's xcb plugin, and xhost


def test_the_pins_are_exact_and_the_defaults_are_the_cautious_ones():
    out = subprocess.run(["bash", "-c", '. "$1"; printf "%s\n" "$PY_SHA256" "$PY_VERSION" "$PYSIDE_PIN" "$STM_COMMIT" '
                          '"$STM_USER" "$AGENT_USER" "$PY_URL"', "_", str(GUEST / "pins.sh")],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    sha, version, pyside, commit, stm, agent, url = out[:7]
    assert len(sha) == 64 and set(sha) <= set("0123456789abcdef") and version.startswith("3.13.")
    assert url.startswith("https://") and version in url and "x86_64-unknown-linux-gnu-install_only" in url
    assert all("==" in part for part in pyside.split()) and "PySide6_Essentials" in pyside
    assert set(commit) == {"0"}                          # the delivery commit is set by whoever delivers
    assert (stm, agent) == ("stm", "user")
    reqs = (ROOT / "deploy" / "osworld" / "grader" / "requirements.txt").read_text()
    assert all(f"{n}==" in reqs for n in ("numpy", "scipy", "PyYAML"))


def test_the_workflow_runs_only_when_started_by_hand():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "osworld-guest.yml").read_text(encoding="utf-8"))
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}


def _lockdown(tmp_path, fake_body, **extra):
    """Run stm-lockdown as this user with a fake visudo (STM_VISUDO) over temporary directories."""
    fake = tmp_path / "fake_visudo"
    fake.write_text("#!/bin/bash\n" + fake_body, encoding="utf-8")
    fake.chmod(0o755)
    env = {**os.environ, "STM_SUDOERS_DIR": str(tmp_path / "sudoers.d"), "STM_POLKIT_RULES_DIR": str(tmp_path / "rules.d"),
           "STM_POLKIT_PKLA_DIR": str(tmp_path / "pkla"), "STM_AGENT_USER": "nobody", "STM_VISUDO": str(fake),
           "STM_LOCKDOWN_ALLOW_NONROOT": "1", **extra}
    return subprocess.run(["bash", str(GUEST / "stm-lockdown")], capture_output=True, text=True, env=env)


#: a visudo that passes a single file (-cf) and, for the whole configuration (-c), fails as the case wants
_FAKE_PASS = 'if [ "$1" = -cf ]; then exit 0; fi\n'


def test_stm_lockdown_undoes_its_dropin_when_it_is_what_breaks_sudoers(tmp_path):
    r = _lockdown(tmp_path, _FAKE_PASS + f'[ -e {tmp_path}/sudoers.d/zz-stm-lockdown ] && exit 1\nexit 0\n')
    assert r.returncode == 1 and "drop-in removed" in r.stderr
    assert not (tmp_path / "sudoers.d" / "zz-stm-lockdown").exists()


def test_stm_lockdown_keeps_its_dropin_when_sudoers_was_already_broken(tmp_path):
    r = _lockdown(tmp_path, _FAKE_PASS + "echo '/etc/sudoers.d/runner: bad permissions' >&2\nexit 1\n")
    assert r.returncode == 0, r.stderr
    assert "already has problems" in r.stderr
    assert (tmp_path / "sudoers.d" / "zz-stm-lockdown").read_text() == "nobody ALL=(ALL:ALL) !ALL\n"
    assert (tmp_path / "rules.d" / "00-stm-lockdown.rules").exists()


def test_stm_lockdown_refuses_a_dropin_that_does_not_parse(tmp_path):
    r = _lockdown(tmp_path, "exit 1\n")
    assert r.returncode == 1 and "does not parse" in r.stderr and not (tmp_path / "sudoers.d").exists()


def test_stm_lockdown_with_a_sound_sudoers_says_nothing_and_keeps_the_dropin(tmp_path):
    r = _lockdown(tmp_path, "exit 0\n")
    assert r.returncode == 0 and "already has problems" not in r.stderr
    assert (tmp_path / "sudoers.d" / "zz-stm-lockdown").exists()


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="the refusal is for everyone but root")
def test_stm_lockdown_refuses_to_run_without_root():
    r = subprocess.run(["bash", str(GUEST / "stm-lockdown")], capture_output=True, text=True)
    assert r.returncode == 1 and "must run as root" in r.stderr


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="writes sudoers and polkit files as root (run in the container test)")
def test_stm_lockdown_writes_its_files_and_is_idempotent(tmp_path):
    env = {**os.environ, "STM_SUDOERS_DIR": str(tmp_path / "sudoers.d"), "STM_POLKIT_RULES_DIR": str(tmp_path / "rules.d"),
           "STM_POLKIT_PKLA_DIR": str(tmp_path / "pkla"), "STM_AGENT_USER": "nobody"}
    for _ in range(2):
        r = subprocess.run(["bash", str(GUEST / "stm-lockdown")], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
    assert (tmp_path / "sudoers.d" / "zz-stm-lockdown").read_text() == "nobody ALL=(ALL:ALL) !ALL\n"
    assert stat.S_IMODE((tmp_path / "sudoers.d" / "zz-stm-lockdown").stat().st_mode) == 0o440
    assert 'subject.user == "nobody"' in (tmp_path / "rules.d" / "00-stm-lockdown.rules").read_text()
    assert "Identity=unix-user:nobody" in (tmp_path / "pkla" / "00-stm-lockdown.pkla").read_text()
    bad = subprocess.run(["bash", str(GUEST / "stm-lockdown")], capture_output=True, text=True,
                         env={**env, "STM_AGENT_USER": "bad name"})
    assert bad.returncode == 1


# ── the real console, offscreen ──
def test_the_launcher_starts_the_real_sealed_console_and_the_export_appears(tmp_path):
    pytest.importorskip("PySide6")
    run, private, export = tmp_path / "run", tmp_path / "private", tmp_path / "export"
    private.mkdir(mode=0o700)
    export.mkdir()
    (private / "tasks.json").write_text(json.dumps({TASK: {"scenario": P4, "seed": SEED}}))
    cfg = L.Config(run_dir=run, private_dir=private, export_dir=export, home=tmp_path / "home",
                   console_cmd=[sys.executable, "-m", "stmbench.cli", "gui"],
                   extra_env={"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)})
    launcher = L.Launcher(cfg)
    launcher.setup()
    stop = threading.Event()
    th = threading.Thread(target=launcher.serve, args=(stop, 0.05), daemon=True)
    th.start()
    try:
        r = run_start(cfg, TASK, wait="150")
        assert r.returncode == 0, r.stderr + (cfg.console_log.read_text() if cfg.console_log.exists() else "")
        assert launcher.child is not None and launcher.child.poll() is None
        files = {p.name for p in export.iterdir()}
        assert {"session.json", "actions.jsonl", "digests.jsonl"} <= files
        for p in export.iterdir():
            assert stat.S_IMODE(p.stat().st_mode) == 0o644, p.name
        assert stat.S_IMODE(export.stat().st_mode) == 0o755
        assert stat.S_IMODE((private / "run").stat().st_mode) == 0o700
        for p in [*export.rglob("*"), *run.rglob("*"), cfg.console_log]:
            if p.is_file():
                assert str(SEED) not in p.read_text(errors="replace"), p.name
        assert str(SEED) not in r.stdout + r.stderr
    finally:
        stop.set()
        th.join(10)
        launcher.close()
