#!/usr/bin/env bash
# Build the two images of the internal evaluation version on a Linux machine with KVM and Docker, then check the VM.
#
#   deploy/eval/build_image.sh --ref <freeze ref> --tasks <private tasks.json> --runner <runner fork checkout> \
#                              --out <empty output dir> [options]
#
# What it makes (in --out):
#   ubuntu_stm_eval.qcow2 (+ .sha256)           the VM image: OSWorld base + the console of the freeze commit + the task list
#   stmbench-grader_<commit12>.tar (+ .sha256)  `docker save` of the grading image built from the same commit
#   grader_image.tag                            the grading image's tag, stmbench-grader:<commit12>
#   manifest.txt                                commit, tarball and file checksums, the pins in force (no seed, no world value)
#   check/<task id>/                            the in-VM check's output and screenshots, one cold boot per scenario
# It never writes into the repository; --tasks must lie outside it.
#
# Inputs
#   --ref REF        a git ref (tag, branch, commit) of this repository: the frozen STM-Bench. Everything (the console, the
#                    guest scripts, their pins deploy/osworld/guest/pins.sh, the grader's Dockerfile) is taken from that
#                    commit's tree by `git archive`, never from the working tree.
#   --tasks FILE     the private {task_id: {"scenario": ..., "seed": ...}}; checked for shape, never printed, never copied
#                    anywhere but into the image (0600, readable by the service account only).
#   --runner DIR     a checkout of our OSWorld-Science fork on the branch with scripts/vm_prep/stm_eval/ and the
#                    ubuntu_stm_eval snapshot (the local branch `eval-guest-check` of the runner clone). It supplies the
#                    boot, provision and bake of the VM (`osci`, run with `uv`).
#   --out DIR        where the products go (created; must be empty); no whitespace in the path.
# Options
#   --vm-dir DIR     the runner's vm directory (default <runner>/vm); needs base/Ubuntu.qcow2, the pristine OSWorld image
#                    (scripts/vm_prep/base/download_base.sh of the runner puts it there). Images are built under images/.
#   --port N         control port of the build guest (default 5191)
#   --expect-tasks N refuse a task list that does not hold exactly N tasks (default 21; 0 = any)
#   --scan-click X,Y position of the console's Scan button for the in-VM check's CSV step (IMAGE.md: how to read it off a
#                    screenshot); without it that step is reported NOT-RUN and the check does not pass
#   --quick          check one task only after the build (default: one per scenario, a cold boot each)
#   --check-only     do the pre-flight (inputs and machine) and print every check's result, build nothing
#   --skip-grader / --skip-vm / --skip-check   leave that step out (a re-run after a failure)
# Environment: STM_TASKS_JSON, OSCI_RUNNER_DIR, OSCI_VM_DIR stand in for --tasks, --runner, --vm-dir; PYTHON (default python3).
#
# Exit status: 0 everything built and every check of the VM passed; 1 a step or check failed; 2 bad usage.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd -P)
PYTHON=${PYTHON:-python3}

REF=""; TASKS=${STM_TASKS_JSON:-}; RUNNER=${OSCI_RUNNER_DIR:-}; OUT=""; VMDIR=${OSCI_VM_DIR:-}
PORT=5191; EXPECT=21; SCAN_CLICK=""; QUICK=0; CHECK_ONLY=0; DO_GRADER=1; DO_VM=1; DO_CHECK=1
usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; }
die() { echo "build_image: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF=${2:?}; shift 2;;
    --tasks) TASKS=${2:?}; shift 2;;
    --runner) RUNNER=${2:?}; shift 2;;
    --out) OUT=${2:?}; shift 2;;
    --vm-dir) VMDIR=${2:?}; shift 2;;
    --port) PORT=${2:?}; shift 2;;
    --expect-tasks) EXPECT=${2:?}; shift 2;;
    --scan-click) SCAN_CLICK=${2:?}; shift 2;;
    --quick) QUICK=1; shift;;
    --check-only) CHECK_ONLY=1; shift;;
    --skip-grader) DO_GRADER=0; shift;;
    --skip-vm) DO_VM=0; shift;;
    --skip-check) DO_CHECK=0; shift;;
    -h|--help) usage; exit 0;;
    *) die "unknown argument: $1 (see --help)";;
  esac
done
[ -n "$REF" ] || die "--ref is required"
[ -n "$TASKS" ] || die "--tasks (or STM_TASKS_JSON) is required"
[ -n "$OUT" ] || die "--out is required"
[ "$DO_VM" = 0 ] && [ "$DO_GRADER" = 0 ] && [ "$CHECK_ONLY" = 0 ] && die "nothing to build"
case "$OUT" in *[[:space:]]*) die "--out must not contain whitespace (it is passed through a word-split variable)";; esac
case "$SCAN_CLICK" in ""|[0-9]*,[0-9]*) ;; *) die "--scan-click wants X,Y in pixels";; esac
case "$PORT$EXPECT" in *[!0-9]*) die "--port and --expect-tasks are numbers";; esac

say() { echo "[build_image $(date +%H:%M:%S)] $*"; }
BAD=0
good() { echo "  [ok]      $*"; }
bad() { echo "  [MISSING] $*"; BAD=$((BAD + 1)); }
abspath() { (cd "$(dirname "$1")" 2>/dev/null && printf '%s/%s\n' "$(pwd -P)" "$(basename "$1")"); }

# ── 1. the inputs: everything that can be known without building ──
say "pre-flight: inputs"
COMMIT=$(git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}") || die "$REF is not a commit of $REPO"
C12=${COMMIT:0:12}
good "freeze commit $C12 ($REF)"
need_in_tree() {
  if git -C "$REPO" cat-file -e "$COMMIT:$1" 2>/dev/null; then good "the commit has $1"; else bad "the commit has no $1"; fi
}
for f in pyproject.toml deploy/osworld/guest/install.sh deploy/osworld/guest/pins.sh deploy/osworld/guest/apt-packages.txt \
         deploy/osworld/guest/stm_launcher.py deploy/osworld/guest/stm-start deploy/osworld/guest/stm-lockdown \
         deploy/osworld/guest/stm-bench-reset deploy/osworld/guest/check_eval_guest.sh \
         deploy/osworld/guest/check_eval_keyboard.py deploy/osworld/grader/Dockerfile deploy/osworld/grader/requirements.txt; do
  need_in_tree "$f"
done
for pkg in python3-numpy python3-scipy; do
  if git -C "$REPO" show "$COMMIT:deploy/osworld/guest/apt-packages.txt" 2>/dev/null | grep -qx "$pkg"; then good "apt-packages.txt at the commit lists $pkg"
  else bad "apt-packages.txt at the commit does not list $pkg"; fi
done
if git -C "$REPO" show "$COMMIT:deploy/osworld/guest/stm_launcher.py" 2>/dev/null | grep -q 'DATA_DIR_OPTION'; then good "the launcher at the commit passes the console's data option"
else bad "the launcher at the commit has no data-directory support (a ref from before the evaluation guest work)"; fi

# the pins of that commit, read from its own pins.sh
PINS=$(mktemp); trap 'rm -f "$PINS"' EXIT
git -C "$REPO" show "$COMMIT:deploy/osworld/guest/pins.sh" > "$PINS" 2>/dev/null || true
PINVALS=$(bash -c 'set -a; . "$1"; printf "%s\n" "$PY_VERSION" "$PY_RELEASE" "$PY_SHA256" "$PYSIDE_PIN"' _ "$PINS" 2>/dev/null) || PINVALS=""
if [ -n "$PINVALS" ]; then
  { read -r PYV; read -r PYR; read -r PYSHA; read -r PYSIDE; } <<<"$PINVALS"
  if [[ $PYSHA =~ ^[0-9a-f]{64}$ ]] && [[ $PYSIDE == *==* ]] && [[ $PYV == 3.13.* ]]; then
    good "pins: Python $PYV (python-build-standalone $PYR, sha256 checked at install), $PYSIDE"
  else bad "the pins of the commit are not exact (PY_SHA256 / PYSIDE_PIN / PY_VERSION)"; fi
else bad "the pins of the commit cannot be read"; fi

# the task list: shape only; no seed is printed
if [ -s "$TASKS" ]; then
  TASKS=$(abspath "$TASKS")
  case "$TASKS" in "$REPO"/*) bad "the task list lies inside the repository ($REPO): move it out; it holds seeds";; *) good "the task list lies outside the repository";; esac
  if [ -n "$(find "$TASKS" -maxdepth 0 -perm /077 2>/dev/null)" ]; then echo "  [warn]    the task list is readable by others (chmod 600 it)"; fi
  SCEN=$(mktemp)
  git -C "$REPO" ls-tree --name-only "$COMMIT" stmbench/trackB/scenarios/ | sed 's|.*/||; s|\.yaml$||' > "$SCEN"
  if SUMMARY=$("$PYTHON" -I - "$TASKS" "$SCEN" "$EXPECT" <<'PY' 2>&1
import json, re, sys
tasks_path, scen_path, expect = sys.argv[1], sys.argv[2], int(sys.argv[3])
known = {ln.strip() for ln in open(scen_path, encoding="utf-8") if ln.strip()}
try:
    tasks = json.load(open(tasks_path, encoding="utf-8"))
except ValueError:
    sys.exit("the task list is not JSON")
if not isinstance(tasks, dict) or not tasks:
    sys.exit("the task list is not a non-empty {task_id: {scenario, seed}} object")
used = set()
for tid, e in tasks.items():
    ok = (re.fullmatch(r"[a-z][a-z0-9_]{0,47}", tid) and isinstance(e, dict) and set(e) == {"scenario", "seed"}
          and isinstance(e["seed"], int) and not isinstance(e["seed"], bool) and e["seed"] >= 0
          and isinstance(e["scenario"], str) and e["scenario"] in known)
    if not ok:
        sys.exit("a task entry is malformed or names a scenario the commit does not have (entry not shown)")
    used.add(e["scenario"])
if expect and len(tasks) != expect:
    sys.exit(f"{len(tasks)} tasks, expected {expect} (--expect-tasks)")
print(f"{len(tasks)} tasks over {len(used)} scenarios")
PY
  ); then good "the task list: $SUMMARY (seeds not shown)"; else bad "the task list: $SUMMARY"; fi
  rm -f "$SCEN"
else bad "no task list at --tasks"; fi

# the runner fork
if [ -n "$RUNNER" ] && [ -d "$RUNNER" ]; then
  RUNNER=$(cd "$RUNNER" && pwd -P)
  for f in scripts/vm_prep/stm_eval/build_image.sh scripts/vm_prep/stm_eval/check_image.py scripts/vm_prep/common/build_image.sh; do
    if [ -f "$RUNNER/$f" ]; then good "the runner has $f"; else bad "the runner has no $f (wrong branch? use eval-guest-check)"; fi
  done
  if grep -q '^  ubuntu_stm_eval:' "$RUNNER/configs/snapshots.yaml" 2>/dev/null; then good "the runner knows the snapshot ubuntu_stm_eval"; else bad "configs/snapshots.yaml of the runner has no ubuntu_stm_eval"; fi
  VMDIR=${VMDIR:-$RUNNER/vm}
else bad "--runner (or OSCI_RUNNER_DIR) is not a directory"; fi

# the output directory
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then bad "--out is not empty (refusing to overwrite products)"; else good "--out is free"; fi

# ── 2. the machine ──
say "pre-flight: machine"
for tool in git tar docker qemu-img uv sha256sum "$PYTHON"; do
  if command -v "$tool" >/dev/null 2>&1; then good "$tool"; else bad "$tool is not installed"; fi
done
if docker info >/dev/null 2>&1; then good "the Docker daemon answers"; else bad "the Docker daemon does not answer (is this user in the docker group?)"; fi
if [ -c /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]; then good "/dev/kvm is usable"; else bad "/dev/kvm is missing or not readable and writable by this user"; fi
for d in "$(dirname "$OUT")" "${VMDIR:-.}"; do
  if [ -d "$d" ]; then
    AV=$(df --output=avail -BG "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ "${AV:-0}" -ge 60 ]; then good "$d has ${AV} GB free"; else bad "$d has ${AV:-?} GB free, needs about 60 (base 24, working copy, built image 21)"; fi
  fi
done
if [ -n "$VMDIR" ] && [ -s "$VMDIR/base/Ubuntu.qcow2" ]; then
  if qemu-img info "$VMDIR/base/Ubuntu.qcow2" 2>/dev/null | grep -qi 'backing file'; then bad "the base image has a backing file"; else good "the pristine base image $VMDIR/base/Ubuntu.qcow2"; fi
else bad "no base image at ${VMDIR:-<vm dir>}/base/Ubuntu.qcow2 (run the runner's scripts/vm_prep/base/download_base.sh)"; fi
if docker image inspect happysixd/osworld-docker >/dev/null 2>&1; then good "the OSWorld QEMU container image is present"; else echo "  [note]    happysixd/osworld-docker is not pulled yet; the first VM start pulls it"; fi

if [ "$BAD" -gt 0 ]; then
  echo "pre-flight: $BAD check(s) failed"
  [ "$CHECK_ONLY" = 1 ] && exit 1
  exit 2
fi
say "pre-flight passed"
[ "$CHECK_ONLY" = 1 ] && exit 0

# ── 3. the source tarball, once, for both images ──
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd -P)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK" "$PINS"' EXIT
TARBALL=$WORK/stm-src.tar
git -C "$REPO" archive --format=tar -o "$TARBALL" "$COMMIT"
mkdir -p "$WORK/src"
tar -xf "$TARBALL" -C "$WORK/src"
TARSHA=$(sha256sum "$TARBALL" | cut -d' ' -f1)
say "source tarball of $C12: sha256 ${TARSHA:0:16}…"
{
  echo "stm-bench commit  $COMMIT"
  echo "source tarball    sha256 $TARSHA (git archive --format=tar of the commit)"
  echo "task list         sha256 $(sha256sum "$TASKS" | cut -d" " -f1) (the file as given; keep this manifest private)"
  echo "pins              Python $PYV (python-build-standalone $PYR), $PYSIDE"
  echo "built             $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -sr)"
} > "$OUT/manifest.txt"

# ── 4. the grading image ──
if [ "$DO_GRADER" = 1 ]; then
  TAG=stmbench-grader:$C12
  say "grading image $TAG (deploy/osworld/grader/Dockerfile of the commit)"
  docker build -f "$WORK/src/deploy/osworld/grader/Dockerfile" -t "$TAG" "$WORK/src"
  # the image's user is `grader` (uid 10001): read from its metadata and passwd, with --user 0 for the small runs, so that
  # the check also works on a rootless engine
  CFG_USER=$(docker image inspect -f '{{.Config.User}}' "$TAG")
  [ "$CFG_USER" = grader ] || { echo "the image's user is '$CFG_USER', not grader"; exit 1; }
  PW=$(docker run --rm --network none --user 0 --entrypoint getent "$TAG" passwd grader)
  case "$PW" in grader:x:10001:*) ;; *) echo "grader is not uid 10001 in the image: $PW"; exit 1;; esac
  docker run --rm --network none --user 0 "$TAG" grade --help >/dev/null
  GID=$(docker image inspect -f '{{.Id}}' "$TAG")
  docker save -o "$OUT/stmbench-grader_$C12.tar" "$TAG"
  (cd "$OUT" && sha256sum "stmbench-grader_$C12.tar" > "stmbench-grader_$C12.tar.sha256")
  echo "$TAG" > "$OUT/grader_image.tag"
  {
    echo "grader image      $TAG ($GID)"
    echo "grader tar        $(cut -d' ' -f1 "$OUT/stmbench-grader_$C12.tar.sha256") stmbench-grader_$C12.tar"
  } >> "$OUT/manifest.txt"
  say "grader image saved: $OUT/stmbench-grader_$C12.tar"
fi

# ── 5. the VM image ──
if [ "$DO_VM" = 1 ]; then
  say "VM image: the runner boots the OSWorld base, installs the console from the tarball, bakes"
  mkdir -p "$VMDIR/images"
  (
    cd "$RUNNER"
    export OSCI_VM_DIR="$VMDIR" STM_COMMIT="$COMMIT" STM_SRC_TARBALL="$TARBALL" STM_TASKS_JSON="$TASKS"
    export STM_SETUP_ARGS="--smoke --screenshot $OUT/smoke.png"
    scripts/vm_prep/stm_eval/build_image.sh "$PORT"
  )
  BUILT=$VMDIR/images/ubuntu_stm_eval.qcow2
  [ -s "$BUILT" ] || { echo "the runner did not produce $BUILT"; exit 1; }
  qemu-img info "$BUILT" | grep -qi 'backing file' && { echo "the built image has a backing file"; exit 1; }
  mv "$BUILT" "$OUT/ubuntu_stm_eval.qcow2"
  rm -f "$BUILT".md5 "${BUILT%.qcow2}.manifest.json" 2>/dev/null || true
  (cd "$OUT" && sha256sum ubuntu_stm_eval.qcow2 > ubuntu_stm_eval.qcow2.sha256)
  SIZE=$(stat -c %s "$OUT/ubuntu_stm_eval.qcow2")
  echo "VM image          $(cut -d' ' -f1 "$OUT/ubuntu_stm_eval.qcow2.sha256") ubuntu_stm_eval.qcow2 ($SIZE bytes)" >> "$OUT/manifest.txt"
  say "VM image: $OUT/ubuntu_stm_eval.qcow2"
fi

# ── 6. the in-VM check ──
RC=0
if [ "$DO_CHECK" = 1 ]; then
  [ -s "$OUT/ubuntu_stm_eval.qcow2" ] || { echo "no image to check at $OUT/ubuntu_stm_eval.qcow2"; exit 1; }
  say "in-VM check (a cold boot per scenario; screenshots in $OUT/check)"
  ARGS=(--image "$OUT/ubuntu_stm_eval.qcow2" --guest-dir "$WORK/src/deploy/osworld/guest" --out "$OUT/check" --port "$PORT")
  [ "$QUICK" = 1 ] && ARGS+=(--quick)
  [ -z "$SCAN_CLICK" ] || ARGS+=(--scan-click "$SCAN_CLICK")
  ( cd "$RUNNER" && STM_TASKS_JSON="$TASKS" OSCI_VM_DIR="$VMDIR" uv run python scripts/vm_prep/stm_eval/check_image.py "${ARGS[@]}" ) \
    2>&1 | tee "$OUT/check.log" || true
  if grep -q '^RESULT: ok' "$OUT/check.log"; then echo "VM check         passed (check.log)" >> "$OUT/manifest.txt"; else
    echo "VM check         NOT passed (check.log)" >> "$OUT/manifest.txt"; RC=1; fi
fi
say "done. Products in $OUT:"
ls -l "$OUT"
exit "$RC"
