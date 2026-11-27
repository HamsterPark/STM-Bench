#!/bin/bash
# The check of the evaluation guest, run inside the built virtual machine as the agent's account (`user`).
# It prints one line per check, `PASS <what>`, `FAIL <what> (<why>)` or `NOT-RUN <what> (<why>)`, and a last line
# `RESULT: <p> passed, <f> failed, <n> not run`. A check that could not be made is NOT-RUN, never a pass; the exit
# status is 0 only when every check passed (1: a failure, 2: nothing failed but something was not run).
#
#   check_eval_guest.sh [--stage] [--shots DIR] [--scan-click X,Y] [--csv-wait SECONDS] <task_id>
#
#   --stage            first do what a task's config does (needs the sudo password, $SUDO_PASSWORD, default "password"):
#                        sudo sh -c 'chgrp -R stm /opt/stm-bench && chmod -R o-rwx /opt/stm-bench'
#                        sudo /usr/local/sbin/stm-lockdown
#                        stm-start <task_id>; sleep 3
#                      Without it the guest must be in that state already (a task as the agent sees it).
#   --shots DIR        where the keyboard part writes its screenshots (default /tmp/stm-eval-check)
#   --scan-click X,Y   the screen position of the console's "Scan" button, in pixels: the check clicks it, waits for the
#                      scan to be saved and looks for the frame's <name>_z_forward.csv in /srv/stm-data. The position depends on the console's
#                      layout and is read once from a screenshot (deploy/eval/IMAGE.md). Without it that check is NOT-RUN.
#   --csv-wait S       how long to wait for the CSV after the click (default 240 s; a scan takes simulated minutes, the
#                      guest's wall time is shorter)
#
# What it covers, as the agent's account (the whole list is in deploy/osworld/README.md, "The machine's side"):
#   the console is up; system python3 imports numpy and scipy; /srv/stm-data is readable and not writable and holds a
#   CSV after a scan; /opt/stm-bench, /var/lib/stm-bench/private and tasks.json cannot be read; sudo is refused; and the
#   keyboard part (check_eval_keyboard.py: Ctrl+Alt+T, python3 in the terminal, Alt+Tab, with a screenshot per step).
# It never reads the task list or the private directory, and prints no file content from /srv/stm-data (names, sizes,
# modes and counts only). The task's world is not touched beyond the click on "Scan".
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=0
SHOTS=/tmp/stm-eval-check
SCAN_CLICK=""
CSV_WAIT=240
DATA=${STM_DATA_DIR:-/srv/stm-data}
RUN=${STM_BENCH_RUN_DIR:-/run/stm-bench}
usage() { sed -n '2,/^set -u/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE=1; shift;;
    --shots) SHOTS=$2; shift 2;;
    --scan-click) SCAN_CLICK=$2; shift 2;;
    --csv-wait) CSV_WAIT=$2; shift 2;;
    -h|--help) usage; exit 0;;
    -*) echo "unknown option $1" >&2; exit 64;;
    *) break;;
  esac
done
if [ $# -ne 1 ] || ! [[ $1 =~ ^[a-z][a-z0-9_]{0,47}$ ]]; then usage >&2; exit 64; fi
TASK=$1
if [ -n "$SCAN_CLICK" ] && ! [[ $SCAN_CLICK =~ ^[0-9]{1,4},[0-9]{1,4}$ ]]; then echo "--scan-click wants X,Y in pixels" >&2; exit 64; fi
mkdir -p "$SHOTS"
export DISPLAY=${DISPLAY:-:0}

PASSED=0; FAILED=0; NOTRUN=0
pass() { echo "PASS $1"; PASSED=$((PASSED + 1)); }
fail() { echo "FAIL $1${2:+  ($2)}"; FAILED=$((FAILED + 1)); }
notrun() { echo "NOT-RUN $1${2:+  ($2)}"; NOTRUN=$((NOTRUN + 1)); }
# check <what> <shell command that succeeds when the check passes>; the command's own output is not shown
check() {
  local what=$1; shift
  if bash -c "$*" >/dev/null 2>&1; then pass "$what"; else fail "$what"; fi
}

# What the image is, for comparing two builds of it (the CI route and a collaborator's own): INFO lines are facts, not checks.
CONSOLE_COMMIT=$(head -1 /opt/stm-bench/COMMIT 2>/dev/null || echo "(unreadable)")
# ── the state a task is in ──
if [ "$STAGE" = 1 ]; then
  PW=${SUDO_PASSWORD:-password}
  echo "$PW" | sudo -S -p '' sh -c 'chgrp -R stm /opt/stm-bench && chmod -R o-rwx /opt/stm-bench' || echo "(staging: closing /opt/stm-bench failed)"
  echo "$PW" | sudo -S -p '' /usr/local/sbin/stm-lockdown || echo "(staging: stm-lockdown failed)"
  /usr/local/bin/stm-start "$TASK" || echo "(staging: stm-start failed)"
  sleep 3
fi
check "the console is up for the task ($RUN/ready, $RUN/task)" "test -f $RUN/ready && [ \"\$(cat $RUN/task)\" = $TASK ]"
echo "INFO console build: $CONSOLE_COMMIT"
dpkg-query -W -f='INFO package ${Package} ${Version}\n' python3 python3-numpy python3-scipy 2>/dev/null || echo "INFO packages: dpkg-query unavailable"
stat -c 'INFO mode %n %U:%G %a' /srv/stm-data /opt/stm-bench /var/lib/stm-bench /var/lib/stm-bench/private /var/lib/stm-bench/export \
  /usr/local/bin/stm-start /usr/local/sbin/stm-lockdown /usr/local/sbin/stm-bench-reset /opt/stm-bench/guest/stm_launcher.py 2>&1 | sed 's/^stat: cannot.*/INFO mode (a path is not visible to this account)/'

# ── Python: the model's terminal ──
PY=$(command -v python3 || true)
if [ "$PY" = /usr/bin/python3 ]; then pass "python3 is the system one (/usr/bin/python3)"; else fail "python3 is the system one (/usr/bin/python3)" "found '${PY:-none}'"; fi
if VERS=$(/usr/bin/python3 -c 'import numpy, scipy, sys; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)' 2>&1); then
  pass "python3 imports numpy and scipy ($VERS)"
else
  fail "python3 imports numpy and scipy" "$(echo "$VERS" | tail -1 | cut -c1-120)"
fi

# ── the data copy ──
check "$DATA exists and can be listed and read" "test -d $DATA && test -r $DATA && test -x $DATA && ls $DATA"
if [ -d "$DATA" ]; then
  OWNER=$(stat -c '%U %a' "$DATA" 2>/dev/null)
  if [ "$OWNER" = "stm 755" ]; then pass "$DATA belongs to stm, mode 755"; else fail "$DATA belongs to stm, mode 755" "is '$OWNER'"; fi
fi
check "$DATA cannot be written by this account" "! touch $DATA/.probe && ! test -e $DATA/.probe"
check "nothing in $DATA is unreadable or writable by others" "! find $DATA -mindepth 1 \( ! -perm -u=r -o ! -perm -g=r -o ! -perm -o=r -o -perm -g=w -o -perm -o=w \) | grep -q ."

# a scan saved in the console leaves a CSV
if [ -z "$SCAN_CLICK" ]; then
  notrun "a scan saved in the console leaves a CSV in $DATA" "give --scan-click X,Y, the position of the Scan button"
else
  BEFORE=$(find "$DATA" -name '*_z_forward.csv' 2>/dev/null | wc -l)
  if python3 - "$SCAN_CLICK" "$SHOTS" <<'PY' 2>/dev/null
import sys, time
import pyautogui
x, y = (int(v) for v in sys.argv[1].split(","))
pyautogui.FAILSAFE = False
pyautogui.screenshot(f"{sys.argv[2]}/00_before_scan.png")
pyautogui.click(x, y)
time.sleep(2)
pyautogui.screenshot(f"{sys.argv[2]}/00_after_scan_click.png")
PY
  then
    END=$((SECONDS + CSV_WAIT)); NOW=$BEFORE
    while [ "$SECONDS" -lt "$END" ]; do
      NOW=$(find "$DATA" -name '*_z_forward.csv' 2>/dev/null | wc -l)
      [ "$NOW" -gt "$BEFORE" ] && break
      sleep 2
    done
    python3 -c 'import pyautogui; pyautogui.screenshot("'"$SHOTS"'/00_after_scan.png")' >/dev/null 2>&1 || true
    if [ "$NOW" -gt "$BEFORE" ]; then
      pass "a scan saved in the console leaves its z-forward CSV in $DATA ($((NOW - BEFORE)) new)"
      NEWEST=$(find "$DATA" -name '*_z_forward.csv' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
      if [ "$(stat -c %a "$NEWEST")" = 644 ]; then pass "the CSV is mode 644 ($(basename "$NEWEST"), $(stat -c %s "$NEWEST") bytes, $(wc -l < "$NEWEST") lines)"; else fail "the CSV is mode 644" "$(stat -c %a "$NEWEST")"; fi
      # a cheap look at the console's header whitelist: no column or comment line may name a seed or the truth
      if grep -qiE 'seed|truth|hidden' "$NEWEST"; then fail "the CSV names neither a seed nor the truth"; else pass "the CSV names neither a seed nor the truth"; fi
    else
      fail "a scan saved in the console leaves a CSV in $DATA" "none within ${CSV_WAIT} s; see $SHOTS/00_after_scan.png"
    fi
  else
    fail "a scan saved in the console leaves a CSV in $DATA" "the click could not be made (pyautogui)"
  fi
fi

# ── what the agent's account must not reach ──
check "/opt/stm-bench cannot be listed or read" "test -d /opt/stm-bench && ! ls /opt/stm-bench && ! cat /opt/stm-bench/COMMIT && ! ls /opt/stm-bench/src"
check "the console's scenario files and code cannot be read" "! cat /opt/stm-bench/src/pyproject.toml && ! ls /opt/stm-bench/src/stmbench/trackB/scenarios && ! test -r /opt/stm-bench/venv/bin/python"
check "/var/lib/stm-bench/private cannot be listed" "test -d /var/lib/stm-bench/private && ! ls /var/lib/stm-bench/private"
check "tasks.json cannot be read" "! cat /var/lib/stm-bench/private/tasks.json && ! test -r /var/lib/stm-bench/private/tasks.json"
check "the episode's private run directory cannot be read" "! ls /var/lib/stm-bench/private/run && ! cat /var/lib/stm-bench/private/console.log"
check "the export is readable and not writable" "ls /var/lib/stm-bench/export && ! touch /var/lib/stm-bench/export/.probe"
check "sudo is refused (sudo -n true fails)" "command -v sudo && ! sudo -n true"
check "sudo -S with the base image's password is refused too" "command -v sudo && ! echo \"\${SUDO_PASSWORD:-password}\" | sudo -S -p '' true"
check "this account is in neither the sudo nor the admin group" "! id -nG | tr ' ' '\\n' | grep -qxE 'sudo|admin'"

# ── the keys ──
if python3 -c 'import pyautogui' >/dev/null 2>&1; then
  KB=$(python3 "$HERE/check_eval_keyboard.py" --shots "$SHOTS" 2>&1) || true
  echo "$KB"
  P=$(echo "$KB" | grep -c '^PASS ') || true
  F=$(echo "$KB" | grep -c '^FAIL ') || true
  PASSED=$((PASSED + P)); FAILED=$((FAILED + F))
  [ $((P + F)) -ge 5 ] || { fail "keyboard: all five steps ran" "$((P + F)) of 5 gave a line"; }
else
  notrun "keyboard: Ctrl+Alt+T, python3 in the terminal, Alt+Tab" "python3 cannot import pyautogui"
fi

echo "screenshots: $SHOTS ($(find "$SHOTS" -name '*.png' | wc -l) files)"
echo "RESULT: $PASSED passed, $FAILED failed, $NOTRUN not run"
[ "$FAILED" -eq 0 ] || exit 1
[ "$NOTRUN" -eq 0 ] || exit 2
exit 0
