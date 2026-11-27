#!/usr/bin/env python3
"""The console's launcher in the virtual machine: a service that runs as the account ``stm``.

The agent's account (``user``) must be able to open the instrument without ever handling the
episode's seed or being able to read it. This service owns the seed: it reads ``tasks.json``
(``{task_id: {"scenario": ..., "seed": ...}}``, a private file of ``stm``) and starts the sealed
console on a task id that ``user`` writes into a FIFO.

Protocol (all paths are arguments; the defaults are the ones of ``deploy/osworld/README.md``)::

    <run dir>/start    FIFO, mode 0622: ``user`` writes one line, the task id (``stm-start`` does it)
    <run dir>/ready    the console writes it ("ready") once its window is on the screen
    <run dir>/task     the id of the task being run (written after the console started)
    <run dir>/error    one line saying why the last request failed; never a seed

One console per boot: the first valid request starts it, the same id again changes nothing, another
id is refused. A console that dies before its window was shown (display not up yet, a missing
library) is cleaned up (its private run directory, its half-written export) and the request can be
repeated; nothing the agent could have seen is lost then. After the window was shown, nothing is
ever cleaned: the export is the evidence.

The seed goes from ``tasks.json`` to the console's standard input as one JSON line. It is not in any
argument or environment variable, and no message of this module (log line, error file) repeats a
line it read, because a request or a task entry may hold it. Standard library only (POSIX).
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

#: a task id: lower case, digits and underscores, starting with a letter (the dataset's ids look like this)
TASK_ID = re.compile(r"[a-z][a-z0-9_]{0,47}")
MAX_REQUEST_BYTES = 256
#: the console's option that names the directory it writes plain-text (CSV) copies of its images, spectra and move
#: traces to. The one place that spells it: the console side (branch bench/console-manip) had not fixed the name
#: when this was written, so change it here only.
DATA_DIR_OPTION = "--data-dir"
#: environment variables the console may inherit from this service (everything else is dropped)
PASS_ENV = ("DISPLAY", "XAUTHORITY", "QT_QPA_PLATFORM", "QT_QPA_FONTDIR", "LIBGL_ALWAYS_SOFTWARE", "LANG",
            "LC_ALL", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")


class RequestError(Exception):
    """A request that is not (or not a known) task id; the text is safe to show to the agent's account."""


def parse_request(raw: bytes) -> str:
    """The task id in one request line; ``RequestError`` (never repeating the line) when it is not one."""
    if len(raw) > MAX_REQUEST_BYTES:
        raise RequestError("the request is too long to be a task id")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise RequestError("a task id is plain ASCII") from None
    text = text.strip()
    if not TASK_ID.fullmatch(text):
        raise RequestError("not a task id (lower case letters, digits and underscores, at most 48)")
    return text


def load_task(tasks_json: Path, task_id: str) -> tuple[str, int]:
    """``(scenario, seed)`` of ``task_id``. Errors say what is wrong, never what the file holds."""
    try:
        tasks = json.loads(Path(tasks_json).read_text(encoding="utf-8"))
    except OSError:
        raise RequestError("the task list cannot be read") from None
    except ValueError:
        raise RequestError("the task list is not JSON") from None
    if not isinstance(tasks, dict):
        raise RequestError("the task list is malformed")
    entry = tasks.get(task_id)
    if entry is None:
        raise RequestError("unknown task id")
    if not isinstance(entry, dict) or set(entry) != {"scenario", "seed"}:
        raise RequestError("the task entry is malformed")
    scenario, seed = entry["scenario"], entry["seed"]
    if not isinstance(scenario, str) or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RequestError("the task entry is malformed")
    return scenario, seed


def _write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _empty(directory: Path) -> None:
    """Remove what is inside ``directory`` (not the directory: it keeps its owner and mode)."""
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


@dataclass
class Config:
    run_dir: Path = Path("/run/stm-bench")
    private_dir: Path = Path("/var/lib/stm-bench/private")
    export_dir: Path = Path("/var/lib/stm-bench/export")
    home: Path = Path("/var/lib/stm-bench/home")
    #: where the console writes its plain-text copy of the data (owned by stm, 0755, files 0644: the agent's account
    #: reads it). It is evidence like the export: emptied before a task starts, never after the agent has seen a run.
    #: None: no copy, and the console is not given the option.
    data_dir: Path | None = Path("/srv/stm-data")
    #: how the console is started (the interpreter that runs this file, so the venv's)
    console_cmd: list[str] = field(default_factory=lambda: [sys.executable, "-m", "stmbench.cli", "gui"])
    #: environment given to the console on top of PASS_ENV
    extra_env: dict[str, str] = field(default_factory=dict)

    @property
    def tasks_json(self) -> Path:
        return self.private_dir / "tasks.json"

    @property
    def episode_dir(self) -> Path:
        """The console's private run directory; it must not exist (or be empty) when the console starts."""
        return self.private_dir / "run"

    @property
    def fifo(self) -> Path:
        return self.run_dir / "start"

    @property
    def ready(self) -> Path:
        return self.run_dir / "ready"

    @property
    def task(self) -> Path:
        return self.run_dir / "task"

    @property
    def error(self) -> Path:
        return self.run_dir / "error"

    @property
    def console_log(self) -> Path:
        return self.private_dir / "console.log"


class Launcher:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.child: subprocess.Popen | None = None
        self.task_id: str | None = None
        self.shown = False                       # the console has shown its window (the ready file exists)
        self._buffer = b""
        self._discarding = False
        self._fifo_fd: int | None = None
        # an episode directory that is not empty belongs to a console of an earlier life of this service
        self.locked = cfg.episode_dir.is_dir() and any(cfg.episode_dir.iterdir())

    # ── set-up ──
    def setup(self) -> None:
        cfg = self.cfg
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(cfg.run_dir, 0o755)
        for p in (cfg.ready, cfg.task, cfg.error):
            p.unlink(missing_ok=True)
        cfg.fifo.unlink(missing_ok=True)
        os.mkfifo(cfg.fifo, 0o622)
        os.chmod(cfg.fifo, 0o622)                # mkfifo obeys the umask: the agent's account must be able to write
        # O_RDWR keeps one reader open for ever, so that a writer never blocks or sees a broken pipe
        self._fifo_fd = os.open(cfg.fifo, os.O_RDWR | os.O_NONBLOCK)
        cfg.home.mkdir(parents=True, exist_ok=True)
        try:
            self._ensure_data_dir()
        except OSError:
            # /srv is root's: install.sh makes the directory. Said again, with the reason, when a task is requested.
            print("stm-bench launcher: the data directory is missing and cannot be created", flush=True)
        qt_dir = cfg.run_dir / "qt"
        qt_dir.mkdir(exist_ok=True)
        os.chmod(qt_dir, 0o700)

    def _ensure_data_dir(self) -> None:
        """Make the data directory if it is missing (the account `stm` may do so only where its parent allows)."""
        d = self.cfg.data_dir
        if d is None:
            return
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o755)                       # the agent's account must be able to list and read it

    def close(self) -> None:
        if self._fifo_fd is not None:
            os.close(self._fifo_fd)
            self._fifo_fd = None
        self.stop_console()

    # ── requests ──
    def feed(self, data: bytes) -> list[str]:
        """Bytes read from the FIFO; handles every complete line and returns the outcome of each
        (``"started"``, ``"same"`` or ``"refused"``)."""
        outcomes: list[str] = []
        self._buffer += data
        while True:
            nl = self._buffer.find(b"\n")
            if nl < 0:
                break
            line, self._buffer = self._buffer[:nl], self._buffer[nl + 1:]
            if self._discarding:
                self._discarding = False
                self._fail("the request is too long to be a task id")
                outcomes.append("refused")
                continue
            outcomes.append(self.handle(line))
        if len(self._buffer) > MAX_REQUEST_BYTES:
            self._buffer = b""
            self._discarding = True
        return outcomes

    def handle(self, line: bytes) -> str:
        try:
            task_id = parse_request(line)
            return self._start(task_id)
        except RequestError as exc:
            self._fail(str(exc))
            return "refused"

    def _start(self, task_id: str) -> str:
        cfg = self.cfg
        self._reap()
        if self.child is not None:
            if task_id == self.task_id:
                return "same"
            raise RequestError("a task was already started in this session")
        if self.locked:
            raise RequestError("a task was already started in this session")
        scenario, seed = load_task(cfg.tasks_json, task_id)
        cfg.error.unlink(missing_ok=True)
        cfg.ready.unlink(missing_ok=True)
        # a failed earlier attempt left nothing the agent has seen: start from empty directories
        try:
            self._ensure_data_dir()
        except OSError:
            raise RequestError("the data directory cannot be created") from None
        _empty(cfg.export_dir)
        if cfg.data_dir is not None:
            _empty(cfg.data_dir)
        if cfg.episode_dir.exists():
            shutil.rmtree(cfg.episode_dir, ignore_errors=True)
        argv = [*cfg.console_cmd, "--sealed", "--private-dir", str(cfg.episode_dir), "--export-dir",
                str(cfg.export_dir), "--lang", "en", "--ready-file", str(cfg.ready)]
        if cfg.data_dir is not None:
            argv += [DATA_DIR_OPTION, str(cfg.data_dir)]
        env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
        env.update({"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(cfg.home),
                    "XDG_RUNTIME_DIR": str(cfg.run_dir / "qt"), "NO_AT_BRIDGE": "1", "QT_ACCESSIBILITY": "0",
                    "PYTHONUNBUFFERED": "1", **cfg.extra_env})
        env.setdefault("QT_QPA_PLATFORM", "xcb")
        env.setdefault("DISPLAY", ":0")
        log = os.open(cfg.console_log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            self.child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=log, env=env,
                                          cwd=str(cfg.home), close_fds=True, start_new_session=True)
        except OSError:
            self.child = None
            raise RequestError("the console could not be started") from None
        finally:
            os.close(log)
        try:
            # the only channel of the seed; the console reads one line and the pipe is closed
            self.child.stdin.write((json.dumps({"scenario": scenario, "seed": seed}) + "\n").encode("utf-8"))
            self.child.stdin.close()
        except OSError:
            pass                                   # the console died at once: the next poll reports it
        self.task_id = task_id
        self.shown = False
        _write_atomic(cfg.task, task_id + "\n")
        return "started"

    def _fail(self, why: str) -> None:
        _write_atomic(self.cfg.error, why + "\n")

    # ── the console ──
    def _reap(self) -> None:
        """Notice a console that has ended; clean up after one that never showed its window."""
        child = self.child
        if child is None:
            return
        if not self.shown and self.cfg.ready.exists():
            self.shown = True
        code = child.poll()
        if code is None:
            return
        self.child = None
        if not self.shown and self.cfg.ready.exists():
            self.shown = True
        if self.shown:
            self.locked = True                     # the agent saw it: the export stays as it is, no second run
            self._fail("the console has ended")
        else:
            self.cfg.task.unlink(missing_ok=True)
            self.task_id = None
            _empty(self.cfg.export_dir)
            if self.cfg.data_dir is not None:
                _empty(self.cfg.data_dir)
            shutil.rmtree(self.cfg.episode_dir, ignore_errors=True)
            self._fail(f"the console ended before its window was shown (exit {code}); see the console log of "
                       "the service account")

    def stop_console(self) -> None:
        child, self.child = self.child, None
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # ── the loop ──
    def serve(self, stop: threading.Event, tick_s: float = 0.2) -> None:
        assert self._fifo_fd is not None, "call setup() first"
        while not stop.is_set():
            readable, _, _ = select.select([self._fifo_fd], [], [], tick_s)
            if readable:
                try:
                    data = os.read(self._fifo_fd, 4096)
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EINTR):
                        raise
                    data = b""
                if data:
                    self.feed(data)
            self._reap()


def _config_from_args(argv: list[str] | None) -> Config:
    d = Config()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", type=Path, default=d.run_dir)
    ap.add_argument("--private-dir", type=Path, default=d.private_dir)
    ap.add_argument("--export-dir", type=Path, default=d.export_dir)
    ap.add_argument("--home", type=Path, default=d.home)
    ap.add_argument("--data-dir", type=Path, default=d.data_dir,
                    help="where the console writes the plain-text copy of its data (default: %(default)s)")
    ap.add_argument("--no-data-dir", action="store_true", help="write no plain-text copy (the option is not passed on)")
    ap.add_argument("--console-cmd", default=None,
                    help="the console's command up to its option list (default: this interpreter -m stmbench.cli gui)")
    a = ap.parse_args(argv)
    cfg = Config(a.run_dir, a.private_dir, a.export_dir, a.home, None if a.no_data_dir else a.data_dir)
    if a.console_cmd:
        import shlex
        cfg.console_cmd = shlex.split(a.console_cmd)
    return cfg


def main(argv: list[str] | None = None) -> int:
    os.umask(0o022)                                 # the export is read by the agent's account: files 0644, directories 0755
    launcher = Launcher(_config_from_args(argv))
    launcher.setup()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    print("stm-bench launcher: waiting for a task id on", launcher.cfg.fifo, flush=True)
    try:
        launcher.serve(stop)
    finally:
        launcher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
