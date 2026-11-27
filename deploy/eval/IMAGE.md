# Building the evaluation images on a KVM machine

The internal evaluation version needs two images, both made from one frozen STM-Bench commit by
[`build_image.sh`](build_image.sh) on **any Linux machine with KVM and Docker**. The main route is our fork's workflow (below), but nothing depends on GitHub Actions: this script runs alone, and
nothing is fetched from GitHub except by the tools the image itself needs (apt, the Python and Qt downloads, the
OSWorld base image, the QEMU container image).

| Product | What it is |
|---|---|
| `ubuntu_stm_eval.qcow2` | the VM the model works in: OSWorld Ubuntu 22.04, the console, the system Python with numpy and scipy, the task list (only the service account `stm` can read it) |
| `stmbench-grader_<commit12>.tar` | `docker save` of the grading image (`stmbench-grader:<commit12>`), built from `deploy/osworld/grader/Dockerfile` of the same commit; it replays an exported episode on the host |

**Whoever holds the qcow2 can read the seeds in it.** The model cannot (the console's service account owns them); a
person with the image file can, as with the delivered image. Keep the image and `tasks.json` private; never put either, or
a log that might name a seed, into a repository.

## Two routes to the same image

1. **Main route: our fork's workflow** `build-stm-eval` (OSWorld-Science fork, branch `eval-guest-check` of the runner
   clone; file `.github/workflows/build-stm-eval.yml`). A GitHub-hosted `ubuntu-24.04` runner has KVM and Docker; the workflow
   checks out the frozen commit of HamsterPark/STM-Bench (which must already be pushed there), writes the task list from a
   repository secret, downloads the base image and runs **this** `build_image.sh`, then uploads the products to a **private**
   Hugging Face repository given as an input (never the delivered dataset or its pull requests: the workflow and
   `upload_products.py` refuse it). No artifact stays on GitHub, no public release, no public dataset. To use it, once the
   freeze commit is fixed and the branch pushed to the fork (not done: nothing is pushed or triggered before the coordinator
   says so):

   ```bash
   gh secret set STM_EVAL_TASKS_JSON --repo HamsterPark/OSWorld-Science < <private tasks.json>
   gh secret set HF_EVAL_VM_TOKEN --repo HamsterPark/OSWorld-Science   # reads the token from the prompt
   gh workflow run build-stm-eval.yml --repo HamsterPark/OSWorld-Science --ref eval-guest-check \
       -f stm_commit=<40 hex> -f upload_repo=HamsterPark/stm-bench-eval-vm -f check_scope=all -f scan_click=<X,Y>
   ```

   The fork is public, so the workflow's logs are public: the script prints no seed, task list or world value, and the
   in-VM check prints only `PASS`/`FAIL`/`INFO` lines. Delete the secret when the build is done
   (`gh secret delete STM_EVAL_TASKS_JSON --repo HamsterPark/OSWorld-Science`).
2. **Backup route: this script on a collaborator's own KVM machine** (above).

Both run the same `build_image.sh` at the same commit, with the same pins, so the image should be the same item by item.
Apt is not pinned, so builds on different days can differ in package versions; the checklist shows where.

### Matching the two routes

Compare the `manifest.txt` and `check.log` of the two builds (keep both private). Each line must be equal except where
stated:

| What | Where to read it | Must match |
|---|---|---|
| STM-Bench commit | `manifest.txt` `stm-bench commit`; `check.log` `INFO console build:` | identical (the second also carries the tarball's hash) |
| source tarball | `manifest.txt` `source tarball sha256` | identical |
| Python build and Qt | `manifest.txt` `pins`; `/opt/stm-bench/versions.txt` in the image | identical |
| package versions of `python3`, `python3-numpy`, `python3-scipy` | `check.log` `INFO package` lines | identical; if they differ, apt moved between the builds: rebuild one of them the same day |
| task list | `manifest.txt` `task list sha256` | identical (same file) |
| file permissions | `check.log` `INFO mode` lines (`/srv/stm-data stm:stm 755`, `/opt/stm-bench` closed to `user`, private 700, export 755, `stm-lockdown` 750) | identical |
| snapshot name and port | `ubuntu_stm_eval`, 5191 (`configs/snapshots.yaml`) | identical |
| the check's verdict | every `PASS`/`FAIL`/`NOT-RUN` line of `check.log` | identical, all `PASS` |
| image and grading tar checksums | `*.sha256` | **not expected to match**: qcow2 files embed timestamps and the zero fill; compare the lines above instead |

## The data copy the check looks for

The console writes, per scan, one `<name>_<channel>_<direction>.csv` per frame channel (z and current, plus didv with the
lock-in on) and direction, e.g. `sim001_z_forward.csv`; per spectrum or move trace `<name>.csv` (names are the data page's
file names without the extension, spaces included); headers are `#` lines of whitelisted keys; each file is written after the
action succeeded, within about a second, and never overwritten. The check clicks Scan and waits for a `*_z_forward.csv`.
The console refuses a `--data-dir` that is, or lies inside, the run, private or export directory; `/srv/stm-data` is none
of them.

## What you need

- the machine: `/dev/kvm` readable and writable by your user, Docker (your user can run `docker`), `qemu-img`, `uv`,
  `git`, `sha256sum`, `python3`; about 60 GB free for the build (base image 24 GB, a working copy, the 21 GB product);
  outbound network. A rootless Docker works if it can use `/dev/kvm` and `/dev/net/tun`.
- this repository, with the freeze commit in it (`git rev-parse <ref>` works).
- our runner fork (OSWorld-Science), on the branch that holds `scripts/vm_prep/stm_eval/` and the `ubuntu_stm_eval`
  snapshot (the local branch `eval-guest-check` of the clone used for the delivery). It does the booting, provisioning
  and baking of the VM, through `osci`.
- the pristine OSWorld image at `<runner>/vm/base/Ubuntu.qcow2` (`scripts/vm_prep/base/download_base.sh` of the runner
  downloads it, about 24 GB; the build refuses an image with a backing file).
- the private task list, `tasks.json`, **outside the repository**: `{"<task id>": {"scenario": ..., "seed": ...}, ...}`,
  21 entries for the evaluation version. Mode 0600. It is read for its shape (and for the task ids, to pick which tasks the
  check boots); no seed is printed, logged or put on a command line.

## The three inputs, as the build takes them

1. **The freeze commit** (`--ref`): everything comes from `git archive` of that commit, never from the working tree. The
   build refuses a commit that lacks the evaluation guest's files (`check_eval_guest.sh`, the numpy and scipy packages, the
   launcher's data-directory support).
2. **The pins**: `deploy/osworld/guest/pins.sh` of that commit (Python build and its sha256, Qt version) and its
   `apt-packages.txt`. The pre-flight prints the Python and Qt pins in force. `install.sh` checks the Python tarball against
   the pinned sha256. Ubuntu's own packages are not pinned (apt drifts between builds); the manifest records the build date
   and `/opt/stm-bench/versions.txt` inside the image records the Python packages.
3. **The task list** (`--tasks`).

## Build

```bash
# once: the base image and the QEMU container
(cd <runner> && scripts/vm_prep/base/download_base.sh)
docker pull happysixd/osworld-docker

# what can be checked without building (also on a machine without KVM: the machine checks then say what is missing)
deploy/eval/build_image.sh --ref <freeze ref> --tasks <private tasks.json> --runner <runner> --out <new dir> --check-only

# the build: pre-flight, source tarball, grading image, VM image, in-VM check
deploy/eval/build_image.sh --ref <freeze ref> --tasks <private tasks.json> --runner <runner> --out <new dir> \
    [--scan-click X,Y] [--quick]
```

Steps, in order (`--skip-grader`, `--skip-vm` and `--skip-check` leave one out, for a re-run after a failure):

1. **Pre-flight** (every result printed as `[ok]` or `[MISSING]`; any `[MISSING]` stops the build with exit status 2).
2. **Source tarball**: `git archive --format=tar <commit>`, sha256 recorded in `manifest.txt`; both images are built from it.
3. **Grading image**: `docker build -f deploy/osworld/grader/Dockerfile`, then the checks of the delivered
   `build_grader.sh` (user `grader`, uid 10001, `grade --help` runs with `--network none`), `docker save`, sha256.
4. **VM image**: the runner boots the pristine base in the QEMU container on port 5191, uploads the tarball and the task
   list, runs the commit's `deploy/osworld/guest/install.sh` in the guest, checks what the agent's account can and cannot
   do, starts the console once through `stm-start` for a smoke test and resets it (`smoke.png` is the screenshot), overwrites
   the free space with zeros (so that the deleted private copy of the task list is not recoverable from the image), powers
   the guest down and flattens it into a self-contained qcow2.
5. **In-VM check**: for the first task of each scenario (`--quick`: one task), a **cold boot of the built image** and
   `check_eval_guest.sh --stage <task id>` run as the agent's account (see
   [the guest README](../osworld/README.md#the-evaluation-guest-a-data-copy-and-the-models-own-python)). Output and
   screenshots go to `<out>/check/<task id>/`; every `PASS`, `FAIL` and `NOT-RUN` line is printed. The build exits 0 only if
   all of them passed.

The check's CSV step needs the position of the console's **Scan** button on the 1920x1080 screen. Boot the image once
by hand (`osci vm start --snapshot ubuntu_stm_eval --image <qcow2> --rebuild`, noVNC on port 8197), run `stm-start <task id>`
in a terminal, take a screenshot, read the button's pixel position, and pass it as `--scan-click X,Y`. Without it that
step is reported `NOT-RUN`, and a `NOT-RUN` is not a pass.

## Outputs and their checksums

```
<out>/ubuntu_stm_eval.qcow2            + ubuntu_stm_eval.qcow2.sha256
<out>/stmbench-grader_<commit12>.tar   + stmbench-grader_<commit12>.tar.sha256
<out>/grader_image.tag                 stmbench-grader:<commit12>
<out>/manifest.txt                     commit, tarball sha256, pins, image checksums, whether the check passed
<out>/check.log, <out>/check/<task id>/{check.log,*.png}
<out>/smoke.png
```

`.sha256` files are `sha256sum` output (`<hash>  <file name>`): verify with `sha256sum -c <file>.sha256` in the folder.
The grading image is loaded on the grading host with `docker load -i stmbench-grader_<commit12>.tar`; its tag is the one
the task's private record names (`grader_image`).

The runner finds the VM image as `ubuntu_stm_eval.qcow2` in `<OSCI data dir>/stm/vm/` (or `<vm dir>/images/`), or is
pointed at it with `--image`. The tasks of the evaluation tree name the snapshot `ubuntu_stm_eval`.

## What was checked without a KVM machine, and what only one can confirm

Checked on a machine without KVM (2026-11-27): `bash -n` of every script; the pre-flight (`--check-only`) against a
scratch repository clone with a made-up task list; the launcher, its data directory, `stm-bench-reset` and the check
scripts' parsing and argument handling in `tests/test_osworld_guest.py` (Linux); the runner's snapshot registry in its
unit tests.

Only a KVM machine can confirm: that the build runs to the end (apt, the downloads, the control plane's uploads and the
detached bootstrap); that the zero fill leaves a smaller image; that the cold-started image boots, `stm-start` brings the
console up on each scenario; that Ctrl+Alt+T, Alt+Tab and the terminal behave as the keyboard check expects in GNOME with
the console full screen (the window class of the console, `stmbench-gui`, and of the terminal are read from `wmctrl`;
pyautogui must import in the base image's `python3`); that the console of the freeze commit writes the CSV copy to the
directory it is given and that its files are mode 0644; the position of the Scan button; and the checksums of the real
products.

Use the pre-flight output and `check.log` to say what ran; do not report a step that printed `NOT-RUN` as passed.

## The runner's domain label

The snapshot `ubuntu_stm_eval` carries the runner domain `stm`, which only decides where the runner looks for an image
(`data/stm/vm/`) and where its doctor message points. The internal evaluation tree is a **separate dataset root** and shares
nothing with the delivered tree (`stm_p4_atom_*`, `stm_p3_*`, their private files and graders) except that label; the image
file has its own name, so the two never collide. To give it its own domain name (say `stm_eval`), change `domain:` in the
`ubuntu_stm_eval` entry of `configs/snapshots.yaml`, the expected domain set in
`tests/unit/test_layout_and_images.py::test_snapshots_carry_a_domain`, the `ev.domain` assertion in
`tests/unit/test_tasks_and_configs.py`, and the data-root directory the tree is built under; the build scripts in
`scripts/vm_prep/stm_eval/` need no change.

## Where the products are kept: a private Hugging Face repository

The products go to a **private** Hugging Face repository under HamsterPark, suggested name
`HamsterPark/stm-bench-eval-vm` (the workflow's `upload_repo` input stays explicit and required; the name is confirmed when
the workflow is started). `upload_products.py` refuses anything under SciAILab (the delivery) and an existing repository that
is not private. The workflow's token is a **new secret, `HF_EVAL_VM_TOKEN`**, not the delivery's `HF_TOKEN`: a fine-grained
Hugging Face token with write access to this one repository only. The user creates the token, and creates the repository
as private before the first run (a token scoped to one repository may not be allowed to create new ones).

**Inviting collaborators is the user's step, after the upload:** in the repository's settings on Hugging Face, add each
collaborator by account. Anyone with read access can download the image and so read the seeds in it (see the top of this
file); invite only those who may.

### What Hugging Face says about private storage (read 2026-11-27, first choice)

Read from the pages themselves (a fetch of each page, quotes below are verbatim); none of the pages shows a date.

| Point | What the page says | Source |
|---|---|---|
| Free account, private storage | "100GB" (public: "Best-effort"; footnote: "Beyond the first few gigabytes, please use this resource responsibly ... If you need substantial storage space, you will need to upgrade to PRO, Team or Enterprise.") | https://huggingface.co/docs/hub/storage-limits |
| PRO | private "1TB + pay-as-you-go"; $9 a month | same page; https://huggingface.co/pricing |
| Team | private "1TB per seat + pay-as-you-go"; $20 per user per month | same |
| Enterprise | private "1TB per seat + pay-as-you-go"; $500 per user per month | same |
| Extra private storage | "base price of $18/TB/mo"; volume tiers $16 (50 TB and up), $14 (200 TB), $12 (500 TB) | https://huggingface.co/docs/hub/storage-limits |
| Per-repository size | "There is no per-repo size limit for models and datasets, but uploads count against your account's total storage quota"; for TB-sized repositories "contact us" | https://huggingface.co/docs/hub/storage-limits (same text on /repositories-recommendations) |
| Per-file size | recommended "<200GB"; "500GB is the hard limit for a single file size" | same |
| Counts | files per repository "<100k"; entries per folder "<10k"; commit size "<100 files" | same |
| Git LFS or Xet | "the Hub has adopted Xet ... chunk-level deduplication, smaller uploads, and faster downloads than Git LFS"; Git LFS "remains supported". `huggingface_hub` >= 0.32.0 installs `hf_xet`; git users need the Git Xet extension. Whether a *new* repository defaults to Xet is not stated | https://huggingface.co/docs/hub/storage-backends ; https://huggingface.co/docs/hub/xet/using-xet-storage |
| `upload_large_folder` | not mentioned on any page read; the pages say `upload_folder` and `hf upload` "auto-split large folders into multiple commits" and link to a "tips and tricks for large uploads" guide that was not read. No 300 GB or 1 TB repository threshold is stated | the same pages |

Not found or not readable: the free-tier storage on /pricing (that page gives no storage figures), and what happens to a
free account that exceeds 100 GB (the footnote above is all there is).

What this means for the products: one qcow2 of about 21 GB (the delivered image was 20.7 GB) is far below the per-file
limits, and with the grading image's tar (not measured; a Python 3.13 image with numpy, scipy and PyYAML should be of the
order of one gigabyte or less, which is an estimate) the repository holds about 22 GB, inside the free 100 GB of private
storage **if the quota is not already used by other private repositories of the same account**. It was not established
whether that quota belongs to the HamsterPark account or to an organisation of that name, or how it is shared with
repositories already there: look at the account's storage page (https://huggingface.co/storage) before the first upload.
A second image version counts again unless Xet deduplicates it (likely for similar files, not verified). Transfer
rates and bandwidth limits were not looked at. Collaborators with read access download with `hf download` or
`huggingface_hub` and need a Hugging Face account that was invited.

### Fallback: Google Drive through rclone (written, untested)

Used **only if Hugging Face's private allowance or cost does not work out**; Hugging Face stays the first choice. The
workflow's `destination` input chooses (`hf` by default, or `gdrive`); the Drive branch has never been run. The user has a
Google Drive of 5 TB.

How it works: the workflow runs `rclone copy` into one folder named at dispatch (`gdrive_folder`), then `rclone check
--one-way` (Drive keeps an md5 for each file), with the credential from `gh secret set` (commands in the workflow's header).
The folder is then shared by the user with the collaborators' Google accounts only (Drive's "Restricted" general access, a
standard sharing setting that was not looked up on a help page). Collaborators download with rclone or gdown.

**Which credential can be limited to one folder, and what each can and cannot limit** (read 2026-11-27 through a web search
tool's summaries of the pages, not by opening each page; no page showed a date):

| Credential | What it can limit | What it cannot do |
|---|---|---|
| Service account (rclone `service_account_credentials`), the folder shared to the account | reaches only what is shared with it, so one folder | **it has no storage of its own and cannot own files**: an upload into a folder of an ordinary account fails with a 403; Google says to upload into a **shared drive** (Workspace only: personal @gmail.com accounts cannot create one) or to use OAuth for a person. Not usable with a personal 5 TB Google One drive. https://developers.google.com/workspace/drive/api/guides/about-shareddrives , https://developers.google.com/workspace/drive/api/guides/handle-errors |
| OAuth refresh token of the Drive's owner, scope `drive.file` | "per-file access": only files the app created or that the user opened with it ("non-sensitive" scope); this is what the workflow uses, so the token reaches only what rclone itself creates, not the rest of the 5 TB. https://developers.google.com/workspace/drive/api/guides/api-specific-auth | it cannot be pointed at a folder that already exists (rclone creates its own folder); the scope `drive` (all files) is "restricted" and not limited to a folder. Own OAuth client id and secret are advised over rclone's shared one; **how long a refresh token lives for an app in Google's "testing" status was not looked up** |

**Limits and prompts** (what was found; where a figure is missing it says so):

- Upload: "each user can upload and copy 750 GB to Drive within 24 hours" (https://support.google.com/a/answer/172541,
  https://developers.google.com/drive/api/guides/limits; both are about Workspace accounts, the rule for a personal account was
  not read). 22 GB is far below it. Largest single file: 5 TB (not confirmed on a page read).
- Download quota of a shared file: the error "Too many users have viewed or downloaded this file recently" appears for
  large or widely shared files and, per the gdown maintainers, "may take up to 24 hours" to clear
  (https://github.com/wkentaro/gdown/issues/406). **Google's own page and the exact quota (bytes or downloads) were not found.**
  With a handful of collaborators it should not trigger; if it does, the workaround people report is to copy the file into the
  downloader's own Drive and download from there.
- "Google Drive can't scan this file for viruses": Drive shows it for large files before the download; **the size
  threshold was not found.** gdown handles the confirmation page itself by requesting the download URL with `confirm=t`
  (a user report says that without it a large file comes back as a 2.4 KB HTML page with status 200, https://github.com/wkentaro/gdown/issues/406
  thread). rclone downloads through the API; for files Google flags it has the flag
  `--drive-acknowledge-abuse`, and for a service account "it will need Manager permission (not Content Manager)" for the flag
  to work, else "Google will just ignore the flag" (https://rclone.org/drive/). Whether an ordinary 21 GB qcow2 is ever
  flagged, so that the flag is needed, is not known.
- gdown's own limit on folder downloads (a figure of 50 files is often quoted) was **not found in the sources read**; the
  products folder holds more than 50 files once the screenshots are counted, so tell collaborators to download the two image
  files and their `.sha256` by file link with gdown, or the whole folder with `rclone copy`.
- Collaborators check the download with `sha256sum -c` against the `.sha256` files, as for any route.

**Read from the pages themselves (2026-11-27, a second pass that fetched the pages; quotes near-verbatim)**, which settles
some of the points above:

- rclone (https://rclone.org/drive/, "source file drive.md last updated 2026-11-06"): with scope `drive.file` "rclone can
  read/view/modify only those files and folders it creates", and files uploaded by other means "will not be visible to
  rclone" (so the workflow's design, rclone creating its own folder, is the one that scope allows). `drive` is "full access to
  all files". A service account needs the folder shared to it (`service_account_file`, `root_folder_id`). The page says "it
  is only possible to upload 750 GiB of data to Google Drive a day (this is an undocumented limit)". It has no note on service
  accounts' missing storage quota (that comes from Google's developer pages above).
- Google Drive help (https://support.google.com/drive/answer/2423534): the download-quota error "occurs when a file receives
  high view or download volume over a short period" and "typically resets within 24 hours"; the exact quota is not given.
  Virus scan: "There's a 100MB scanning limit and some file types can't be scanned" (https://support.google.com/drive/answer/141702;
  the exact phrase "can't scan this file for viruses" is not on that page, so the 100 MB figure is the best available threshold).
  File size: up to 5 TB (https://support.google.com/drive/answer/37603). Restricted sharing: "Only people who have been given
  access can open the file" (https://support.google.com/drive/answer/2494822). Upload: 750 GB per user per 24 hours
  (Workspace page, last updated 2026-11-12). No official download limit was found; a figure of 10 GB a day appears only in
  forums.
- gdown (https://github.com/wkentaro/gdown): it "skips the virus-scan confirmation page so large downloads actually
  finish"; when Google throttles a much-downloaded file, `--cookies-from-browser` lets a signed-in account through; failed files
  of a folder download are reported at the end. **"Google Drive terminates connections after approximately 1 hour for large
  files"**: a 21 GB download on a slow line can be cut and has to be repeated (rclone retries and resumes by chunk; whether
  gdown resumes was not stated). The page gives no 50-file folder limit.
- `drive.file` in Google's own words: "Create new Drive files, or modify existing files, that you open with an app or that the
  user shares with an app", non-sensitive (https://developers.google.com/drive/api/guides/api-specific-auth; the rest of
  that table row was not retrieved).

Still not found: any service-account storage-quota statement on rclone's own page, Google's exact download quota, and the lifetime of an OAuth refresh token of an app in "testing" status. Not looked at: rclone's flags for bandwidth
and retries.

## Task ids

The internal task ids all carry the prefix `stm_eval_` (for example `stm_eval_p4_atom_01`, `stm_eval_p1_herringbone_02`), so
they never share a name with the delivered tasks (`stm_p4_atom_*`, `stm_p3_*`) and results summed by task id cannot mix the
two. They fit the launcher's id pattern (lower case letters, digits and underscores, starting with a letter, at most 48
characters; the longest planned one has 28) and `stm-start` and the check scripts use them unchanged; the runner's domain
logic works on the snapshot's `domain` label and on the tasks' directories, not on the id's spelling (the runner's `TaskSet` refuses a duplicate id across data roots, which the prefix rules out).
