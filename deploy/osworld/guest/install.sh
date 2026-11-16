#!/usr/bin/env bash
# Install the STM-Bench console into an Ubuntu 22.04 guest (the OSWorld base image, or a container for testing).
# Run as root (or as the agent's account: it re-runs itself with sudo). Needs outbound network (apt, GitHub, PyPI).
#
#   STM_TASKS_JSON=<private tasks.json> bash install.sh
#
# Inputs (environment; defaults and pins are in pins.sh next to this file):
#   STM_NO_TASKS=1   bake no task list (a seed in an image file is readable by anyone who holds the image; the list then
#                    has to reach the private directory by another route, see README)
#   STM_TASKS_JSON   required unless STM_NO_TASKS=1: {task_id: {"scenario": ..., "seed": ...}}; copied to the service account's private
#                    directory (0600). Not printed. Delete your copy afterwards.
#   STM_SRC_TARBALL  `git archive` of the commit to install instead of fetching STM_COMMIT from STM_REPO_URL
#   STM_PY_TARBALL   a local copy of the pinned Python tarball (still checked against PY_SHA256)
#   STM_NO_SYSTEMD=1 do not enable or start the service (a container without systemd)
#   STM_REVOKE_SUDO=1 remove the agent's account from the sudo group at the end
#   APT_OPTS         extra apt-get options (a rootless container needs -o APT::Sandbox::User=root)
#   STM_SKIP_APT_UPDATE=1  do not run apt-get update (offline test with the package lists and archives in place)
#   STM_NO_CHOWN=1   do not chown (a container whose user namespace maps a single uid; the files stay root's)
#   PIP_NO_INDEX=1 PIP_FIND_LINKS=<dir>   pip's own variables: install from a directory of wheels (offline test)
#
# Result (the shared contract of deploy/osworld/README.md): code and venv in /opt/stm-bench (root-owned, world-readable),
# the service account `stm` (no password login), /var/lib/stm-bench/{private 0700, export 0755, home 0700} owned by stm,
# the service /etc/systemd/system/stm-bench.service, /usr/local/bin/stm-start, /usr/local/sbin/stm-bench-reset.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

if [ "$(id -u)" -ne 0 ]; then
  exec sudo -E bash "$0" "$@"
fi
# shellcheck disable=SC1091
. "$HERE/pins.sh"
[ -z "${STM_INSTALL_ENV:-}" ] || . "$STM_INSTALL_ENV"
export DEBIAN_FRONTEND=noninteractive
say() { echo "[stm-install $(date +%H:%M:%S)] $*"; }

if [ "${STM_NO_TASKS:-0}" != 1 ]; then
  : "${STM_TASKS_JSON:?STM_TASKS_JSON must name the private task list (or set STM_NO_TASKS=1)}"
  [ -s "$STM_TASKS_JSON" ] || { echo "no task list at the given path"; exit 1; }
fi
if [ -z "${STM_SRC_TARBALL:-}" ]; then
  case "$STM_COMMIT" in *[!0-9a-f]*|"") echo "STM_COMMIT is not a hex commit"; exit 1;; esac
  if [ "${STM_COMMIT//0/}" = "" ]; then echo "STM_COMMIT is still the placeholder (pins.sh)"; exit 1; fi
fi

# ── packages ──
say "apt: Qt's xcb platform libraries, fonts, xhost, curl, git (apt-packages.txt)"
# shellcheck disable=SC2086
APT="apt-get -o DPkg::Lock::Timeout=180 ${APT_OPTS:-}"
if [ "${STM_SKIP_APT_UPDATE:-0}" != 1 ]; then
  for i in 1 2 3 4 5; do $APT update && break || sleep 10; done
fi
mapfile -t PACKAGES < <(grep -vE '^\s*(#|$)' "$HERE/apt-packages.txt")
$APT install --no-install-recommends -y "${PACKAGES[@]}"

# ── accounts and directories ──
# chown is skipped with STM_NO_CHOWN=1: a container whose user namespace maps one uid cannot give files to another.
own() { [ "${STM_NO_CHOWN:-0}" = 1 ] || chown "$@"; }
if ! id "$STM_USER" >/dev/null 2>&1; then
  say "account $STM_USER (system account, no password login, no shell)"
  useradd --system --user-group --home-dir /var/lib/stm-bench/home --no-create-home --shell /usr/sbin/nologin "$STM_USER"
fi
passwd -l "$STM_USER" >/dev/null 2>&1 || true
install -d -m 0755 /opt/stm-bench /opt/stm-bench/guest /var/lib/stm-bench
install -d -m 0700 /var/lib/stm-bench/private /var/lib/stm-bench/home
install -d -m 0755 /var/lib/stm-bench/export
own "$STM_USER:$STM_USER" /var/lib/stm-bench/private /var/lib/stm-bench/home /var/lib/stm-bench/export

# ── Python 3.13 (pinned, checksum-verified) ──
say "python $PY_VERSION (python-build-standalone $PY_RELEASE)"
PYTAR=${STM_PY_TARBALL:-/tmp/stm-python.tar.gz}
if [ -z "${STM_PY_TARBALL:-}" ]; then
  curl --fail --silent --show-error --location --retry 5 --retry-delay 5 -o "$PYTAR" "$PY_URL"
fi
echo "$PY_SHA256  $PYTAR" | sha256sum -c - >/dev/null || { echo "the Python tarball does not match PY_SHA256"; exit 1; }
rm -rf /opt/stm-bench/python
mkdir -p /opt/stm-bench/python
tar -xzf "$PYTAR" --no-same-owner -C /opt/stm-bench/python --strip-components=1
[ -n "${STM_PY_TARBALL:-}" ] || rm -f "$PYTAR"
GOT=$(/opt/stm-bench/python/bin/python3 -c 'import platform; print(platform.python_version())')
[ "$GOT" = "$PY_VERSION" ] || { echo "python is $GOT, not $PY_VERSION"; exit 1; }

# ── the repository at the pinned commit ──
say "source tree"
rm -rf /opt/stm-bench/src
mkdir -p /opt/stm-bench/src
if [ -n "${STM_SRC_TARBALL:-}" ]; then
  tar -xf "$STM_SRC_TARBALL" --no-same-owner -C /opt/stm-bench/src
  COMMIT_NOTE="tarball $(sha256sum "$STM_SRC_TARBALL" | cut -c1-16)"
  if [ -n "${STM_COMMIT:-}" ] && [ "${STM_COMMIT//0/}" != "" ]; then COMMIT_NOTE="$COMMIT_NOTE (commit $STM_COMMIT)"; fi
else
  T=$(mktemp -d)
  git -C "$T" init -q
  git -C "$T" fetch -q --depth 1 "$STM_REPO_URL" "$STM_COMMIT"
  HEAD=$(git -C "$T" rev-parse FETCH_HEAD)
  case "$HEAD" in "${STM_COMMIT}"*) ;; *) echo "fetched ${HEAD:0:12}, not ${STM_COMMIT:0:12}"; exit 1;; esac
  git -C "$T" archive "$HEAD" | tar -x --no-same-owner -C /opt/stm-bench/src
  rm -rf "$T"
  COMMIT_NOTE=$HEAD
fi
for f in pyproject.toml deploy/osworld/grader/requirements.txt deploy/osworld/guest/stm_launcher.py; do
  [ -f "/opt/stm-bench/src/$f" ] || { echo "the source tree has no $f"; exit 1; }
done

# ── the environment: the grader's pins, plus Qt ──
say "venv and pinned packages"
rm -rf /opt/stm-bench/venv
/opt/stm-bench/python/bin/python3 -m venv /opt/stm-bench/venv
PIP="/opt/stm-bench/venv/bin/python -m pip --disable-pip-version-check --no-cache-dir"
$PIP install -r /opt/stm-bench/src/deploy/osworld/grader/requirements.txt
# shellcheck disable=SC2086
$PIP install $PYSIDE_PIN
# --no-deps: the pins above are the dependencies. Built from a copy, so that /opt/stm-bench/src stays as delivered.
B=$(mktemp -d)
cp -r /opt/stm-bench/src/. "$B/"
$PIP install --no-deps "$B"
rm -rf "$B"
{ echo "# $COMMIT_NOTE"; /opt/stm-bench/venv/bin/python -m pip freeze | grep -v '^stm-bench'
  /opt/stm-bench/venv/bin/python -m pip show stm-bench | sed -n 's/^Version: /stm-bench==/p'; } > /opt/stm-bench/versions.txt
echo "$COMMIT_NOTE" > /opt/stm-bench/COMMIT
find /opt/stm-bench -type d -exec chmod u+rwx,go+rx {} +
find /opt/stm-bench -type f -exec chmod go+r {} +

# ── guest files ──
say "launcher, stm-start, unit, autostart entry"
install -m 0644 "$HERE/stm_launcher.py" /opt/stm-bench/guest/stm_launcher.py
install -m 0755 "$HERE/stm-start" /usr/local/bin/stm-start
install -m 0750 "$HERE/stm-bench-reset" /usr/local/sbin/stm-bench-reset
install -m 0644 "$HERE/stm-bench.service" /etc/systemd/system/stm-bench.service
install -d -m 0755 /etc/xdg/autostart
install -m 0644 "$HERE/stm-bench-xhost.desktop" /etc/xdg/autostart/stm-bench-xhost.desktop

# ── the private task list: shape checked, nothing printed ──
if [ "${STM_NO_TASKS:-0}" = 1 ]; then
  say "no task list baked (STM_NO_TASKS=1): the launcher refuses every task id until tasks.json is placed in the private directory"
else
say "task list"
install -m 0600 "$STM_TASKS_JSON" /var/lib/stm-bench/private/tasks.json
own "$STM_USER:$STM_USER" /var/lib/stm-bench/private/tasks.json
/opt/stm-bench/venv/bin/python - <<'PY'
import json, re, sys
from pathlib import Path
import stmbench.gui.session as s
tasks = json.loads(Path("/var/lib/stm-bench/private/tasks.json").read_text(encoding="utf-8"))
ok = isinstance(tasks, dict) and tasks
for tid, e in (tasks.items() if ok else []):
    ok = ok and re.fullmatch(r"[a-z][a-z0-9_]{0,47}", tid) and isinstance(e, dict) and set(e) == {"scenario", "seed"} \
        and isinstance(e["seed"], int) and not isinstance(e["seed"], bool) and e["seed"] >= 0 \
        and (s.SCENARIO_DIR / f"{e['scenario']}.yaml").is_file()
if not ok:
    sys.exit("the task list is not {task_id: {scenario, seed}} of known scenarios")
print(f"{len(tasks)} tasks")
PY
fi

# ── the service ──
if [ "${STM_NO_SYSTEMD:-0}" != 1 ]; then
  say "systemd: enable (and start, if the system is running)"
  systemctl daemon-reload
  systemctl enable stm-bench.service
  if [ "$(systemctl is-system-running 2>/dev/null || true)" != offline ]; then
    systemctl restart stm-bench.service || true
  fi
fi

if [ "${STM_REVOKE_SUDO:-0}" = 1 ]; then
  say "removing $AGENT_USER from the sudo group (STM_REVOKE_SUDO=1)"
  gpasswd -d "$AGENT_USER" sudo || true
  if grep -rlE "^\s*$AGENT_USER\s" /etc/sudoers.d 2>/dev/null; then
    echo "WARNING: $AGENT_USER has a sudoers.d entry of its own; remove it as well"
  fi
fi
say "done: $COMMIT_NOTE"
