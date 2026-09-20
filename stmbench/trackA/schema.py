"""Column declarations for the Track A manifests (DESIGN.md §5.1).

Every manifest column is declared here with a *role*:

* ``id``     — ``sample_id``;
* ``input``  — what a model may be shown (file references, header fields, context);
* ``label``  — the operator ground truth;
* ``group``  — the leakage group (instrument-day: consecutive frames share a day, a
  position and a tip, data guide §3.11), which also fixes the ``split``;
* ``strat``  — stratification for reporting (year, autosave regime, genre, …);
* ``split``  — ``train`` / ``val`` / ``test``, a deterministic function of the group;
* ``meta``   — bookkeeping that is NOT an input (it describes the frame's future or its
  outcome); a consumer that feeds a ``meta`` column to a model is leaking.

Each spec also lists ``leak_forbidden`` names — columns that must never be given the
``input`` role for that task (T1: anything about how the frame ended or what came next;
T4: the ``COMMENT`` field, 84 % non-empty and full of the parameters being predicted).
:func:`validate` checks a DataFrame against its spec (columns, dtypes, nulls, unique ids,
enumerations, split ⇔ group consistency).

Thresholds live here too so that manifests and their consumers read one number.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

ROLE_ID = "id"
ROLE_INPUT = "input"
ROLE_LABEL = "label"
ROLE_GROUP = "group"
ROLE_STRAT = "strat"
ROLE_SPLIT = "split"
ROLE_META = "meta"
ROLES = (ROLE_ID, ROLE_INPUT, ROLE_LABEL, ROLE_GROUP, ROLE_STRAT, ROLE_SPLIT, ROLE_META)
DTYPES = ("str", "int", "float", "bool")

# ── enumerations ─────────────────────────────────────────────────────────────
SPLITS = ("train", "val", "test")
SPLIT_FRACS = {"train": 0.70, "val": 0.10, "test": 0.20}
# restart-save autosave regime of the instrument-month (data guide §3.5): frames saved
# while it was OFF are born cleaner — every cross-era statistic stratifies on this.
AUTOSAVE_REGIMES = ("on", "off", "mixed", "unknown")
# coarse material family from COMMENT / file name (the DINO image genres of the guide
# exist only for PPT crops, not for raw frames — this is the raw-frame stand-in)
GENRES = ("clean_metal", "molecule_on_metal", "semiconductor", "layered", "other", "unknown")
ABORT_KINDS = ("complete", "glance", "deliberate")
NEXT_ACTIONS = ("stay", "relocate", "long_stop")
NEXT_ACTIONS_FINE = ("restart_same", "zoom", "pan", "relocate", "long_stop")
RUN_END_KINDS = ("relocate", "resize", "pause", "long_stop", "day_end")

# ── thresholds (data guide §3.5, §4, §6) ─────────────────────────────────────
GLANCE_MAX_ACQ_FRAC = 0.10       # "early" glance: < 10 % of the rows acquired
STAY_MAX_MOVE_FRAC = 0.10        # same position: centre moved < 10 % of the field of view
RELOCATE_MIN_MOVE_FRAC = 1.0     # relocation: centre moved ≥ one whole field of view
LONG_STOP_S = 1800.0             # > 30 min before the next frame = long stop
RUN_MAX_GAP_S = 300.0            # glance run: same position, ≤ 300 s apart, same size
RUN_SIZE_RTOL = 0.01
SAME_SIZE_RTOL = 0.10            # T3 fine action: "zoom" if the size changed by > 10 %
T1_HEAD_ROWS = 8                 # rows of the current frame a T1 model may see (glance_strips.csv)
T2_MAX_GAP_S = 1800.0
T2_MIN_ACQ_FRAC = 0.5            # the "large field" frame must be at least half scanned
T2_ZOOM_MAX_RATIO = 0.5          # DESIGN §5.1: next size < ½ current
CONTEXT_FRAMES = 5               # T2/T3 context: previous frames of the same day
DEDUP_DECIMALS_M = 12            # §3.6 key rounds size/position to 1 pm


@dataclass(frozen=True)
class Col:
    name: str
    dtype: str
    role: str
    doc: str
    nullable: bool = False
    choices: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"{self.name}: dtype {self.dtype!r} not in {DTYPES}")
        if self.role not in ROLES:
            raise ValueError(f"{self.name}: role {self.role!r} not in {ROLES}")


@dataclass(frozen=True)
class ManifestSpec:
    task: str
    title: str
    cols: tuple[Col, ...]
    leak_forbidden: frozenset[str]
    doc: str

    def __post_init__(self) -> None:
        names = [c.name for c in self.cols]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"{self.task}: duplicate columns {sorted(dup)}")
        leaked = [c.name for c in self.cols if c.role == ROLE_INPUT and c.name in self.leak_forbidden]
        if leaked:
            raise ValueError(f"{self.task}: leak-forbidden columns declared as input: {leaked}")
        if names.count("sample_id") != 1 or names.count("split") != 1 or names.count("group_key") != 1:
            raise ValueError(f"{self.task}: sample_id / group_key / split must be declared exactly once")

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cols]

    def by_role(self, role: str) -> list[str]:
        return [c.name for c in self.cols if c.role == role]

    @property
    def input_columns(self) -> list[str]:
        return self.by_role(ROLE_INPUT)

    @property
    def label_columns(self) -> list[str]:
        return self.by_role(ROLE_LABEL)

    def column(self, name: str) -> Col:
        for c in self.cols:
            if c.name == name:
                return c
        raise KeyError(name)


# ── split ────────────────────────────────────────────────────────────────────
def split_of(group_key: str) -> str:
    """Deterministic split of a group key (same key ⇒ same split, everywhere, forever)."""
    h = hashlib.sha1(str(group_key).encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / float(1 << 32)
    acc = 0.0
    for name in SPLITS:
        acc += SPLIT_FRACS[name]
        if u < acc:
            return name
    return SPLITS[-1]


# ── shared columns ───────────────────────────────────────────────────────────
def _common(*, material_role: str = ROLE_STRAT) -> tuple[Col, ...]:
    return (
        Col("sample_id", "str", ROLE_ID, "stable id, derived from the input path(s)"),
        Col("instrument", "str", ROLE_STRAT, "Rig_A / Rig_B (the sequence parquet's group)"),
        Col("group_key", "str", ROLE_GROUP, "instrument:YYYY-MM-DD — the leakage group"),
        Col("split", "str", ROLE_SPLIT, "train/val/test, split_of(group_key)", choices=SPLITS),
        Col("year", "int", ROLE_STRAT, "calendar year of the frame start"),
        Col("autosave_regime", "str", ROLE_STRAT,
            "restart-save autosave of the instrument-month (autosave_history_by_month.csv)",
            choices=AUTOSAVE_REGIMES),
        Col("material", "str", material_role, "substrate parsed from COMMENT / file name, 'unknown' if none"),
        Col("genre", "str", ROLE_STRAT, "coarse material family", choices=GENRES),
    )


def _frame_inputs() -> tuple[Col, ...]:
    return (
        Col("path", "str", ROLE_META,
            "absolute path of the .sxm as recorded in the index — bookkeeping only; consumers open rel_path"),
        Col("rel_path", "str", ROLE_INPUT,
            "this frame, relative to stmsim.paths.raw_mirror_root() ($STM_BENCH_RAW; meta.json: raw_root)"),
        Col("size_nm", "float", ROLE_INPUT, "scan range x (width), nm"),
        Col("height_nm", "float", ROLE_INPUT, "scan range y (height), nm — equals size_nm for square frames"),
        Col("nx", "int", ROLE_INPUT, "pixels per line"),
        Col("ny", "int", ROLE_INPUT, "lines"),
        Col("bias_v", "float", ROLE_INPUT, "bias from the header, V (70 frames of the index have none)",
            nullable=True),
    )


def _context() -> tuple[Col, ...]:
    return (
        Col("context_paths_json", "str", ROLE_INPUT,
            "JSON list of earlier frames (oldest first), relative to the raw mirror root like rel_path"),
        Col("context_dt_s_json", "str", ROLE_INPUT,
            "JSON list: seconds from each context frame's start to this frame's start"),
    )


def _next_meta() -> tuple[Col, ...]:
    return (
        Col("next_path", "str", ROLE_META, "the next frame of the day (never an input); null at day end",
            nullable=True),
        Col("dt_next_s", "float", ROLE_META, "seconds to the next frame's start; null at day end", nullable=True),
        Col("move_next_frac", "float", ROLE_META, "|Δcentre| to the next frame / this frame's size", nullable=True),
        Col("next_size_ratio", "float", ROLE_META, "next size / this size", nullable=True),
    )


NEXT_FRAME_LEAKS = frozenset({"next_path", "dt_next_s", "move_next_frac", "next_size_ratio",
                              "next_u", "next_v", "next_w_rel", "next_h_rel", "next_angle_delta_deg"})

# ── T1 continue vs stop ──────────────────────────────────────────────────────
T1 = ManifestSpec(
    task="T1",
    title="continue vs stop",
    doc=(
        "Sample = one frame at step k of a same-position run (frames ≤ RUN_MAX_GAP_S apart, "
        "centre within STAY_MAX_MOVE_FRAC, same size). Input = the earlier frames of the run "
        "(they have already ended, rendered in full) + the first T1_HEAD_ROWS rows of THIS "
        "frame (rendered head-only) + start-time gaps; frames with fewer acquired rows than "
        "that are not samples (the head itself would reveal the stop). Label = the operator "
        "stopped THIS frame before it completed (stopped = the frame shown head-only). Only "
        "instrument-months with restart-save autosave ON (otherwise glances were never "
        "written and the negatives are missing). Runs are kept whatever way they end — a "
        "run that ends in a relocation or long stop is where the last step is a 'stop' "
        "(DESIGN §5.1 leakage note). step_idx is an input on purpose: the step-index "
        "baseline is the mandatory negative control."
    ),
    cols=_common() + _frame_inputs() + _context() + (
        Col("head_rows", "int", ROLE_INPUT, "rows of this frame the model may look at (= T1_HEAD_ROWS)"),
        Col("step_idx", "int", ROLE_INPUT, "0-based position of this frame in its run"),
        Col("run_id", "str", ROLE_META, "group_key:rN — the run this sample belongs to"),
        Col("stopped", "bool", ROLE_LABEL, "operator stopped this frame before it completed"),
        Col("acq_frac", "float", ROLE_META, "fraction of rows acquired (reveals the label)"),
        Col("acq_rows", "int", ROLE_META, "rows acquired (reveals the label)"),
        Col("abort_kind", "str", ROLE_META, "complete / glance / deliberate", choices=ABORT_KINDS),
        Col("run_len", "int", ROLE_META, "frames in the run (reveals the future)"),
        Col("run_end_kind", "str", ROLE_META, "what broke the run after its last frame", choices=RUN_END_KINDS),
        Col("run_last_complete", "bool", ROLE_META, "the run's last frame was complete"),
    ) + _next_meta(),
    leak_forbidden=NEXT_FRAME_LEAKS | {"acq_frac", "acq_rows", "abort_kind", "run_len", "run_end_kind",
                                       "run_last_complete", "nan_rows", "partial_rows", "comment"},
)

# ── T2 where next ────────────────────────────────────────────────────────────
T2 = ManifestSpec(
    task="T2",
    title="where next (region selection)",
    doc=(
        "Sample = a frame (≥ T2_MIN_ACQ_FRAC scanned) whose next frame of the day, within "
        "T2_MAX_GAP_S, is smaller and lies fully inside it with a moved centre "
        "(move_next_frac > 0). Label = the next frame's box in THIS frame's coordinates: "
        "centre (next_u, next_v) with the frame spanning [-0.5, 0.5] along its own x/y axes "
        "after undoing this frame's scan angle, and size (next_w_rel, next_h_rel) as a "
        "fraction of this frame's size. zoom_in marks next size ≤ T2_ZOOM_MAX_RATIO."
    ),
    cols=_common() + _frame_inputs() + (
        Col("angle_deg", "float", ROLE_INPUT, "scan angle of this frame, deg"),
        Col("acq_frac", "float", ROLE_INPUT, "how much of this frame was scanned (visible anyway)"),
        Col("abort_kind", "str", ROLE_STRAT, "complete / glance / deliberate", choices=ABORT_KINDS),
    ) + _context() + (
        Col("next_u", "float", ROLE_LABEL, "next centre, this frame's x axis, in [-0.5, 0.5]"),
        Col("next_v", "float", ROLE_LABEL, "next centre, this frame's y axis, in [-0.5, 0.5]"),
        Col("next_w_rel", "float", ROLE_LABEL, "next width / this width"),
        Col("next_h_rel", "float", ROLE_LABEL, "next height / this height"),
        Col("next_angle_delta_deg", "float", ROLE_LABEL, "next angle − this angle, deg"),
        Col("zoom_in", "bool", ROLE_STRAT, "next size ≤ T2_ZOOM_MAX_RATIO × this size"),
    ) + _next_meta(),
    leak_forbidden=NEXT_FRAME_LEAKS | {"comment"},
)

# ── T3 stay vs relocate ──────────────────────────────────────────────────────
T3 = ManifestSpec(
    task="T3",
    title="stay vs relocate",
    doc=(
        "Sample = any frame (complete or aborted) with a following frame the same day. Label "
        "= what the operator did next, three classes (the primary label, scored as three-class "
        "accuracy + macro one-vs-rest AUROC): long_stop (> LONG_STOP_S before the next frame), "
        "relocate (centre moved ≥ RELOCATE_MIN_MOVE_FRAC fields), else stay. relocate is the "
        "binary fold (relocate or long_stop; score(..., t3_binary=True)). next_action_fine "
        "splits stay into restart_same (same place, same size), zoom (size changed) and pan "
        "(moved < one field)."
    ),
    cols=_common() + _frame_inputs() + (
        Col("acq_frac", "float", ROLE_INPUT, "how much of this frame was scanned (visible anyway)"),
        Col("abort_kind", "str", ROLE_STRAT, "complete / glance / deliberate", choices=ABORT_KINDS),
    ) + _context() + (
        Col("next_action", "str", ROLE_LABEL, "stay / relocate / long_stop", choices=NEXT_ACTIONS),
        Col("relocate", "bool", ROLE_LABEL, "next_action != stay"),
        Col("next_action_fine", "str", ROLE_LABEL, "restart_same / zoom / pan / relocate / long_stop",
            choices=NEXT_ACTIONS_FINE),
    ) + _next_meta(),
    leak_forbidden=NEXT_FRAME_LEAKS | {"comment"},
)

# ── T4 working-point prior ───────────────────────────────────────────────────
T4 = ManifestSpec(
    task="T4",
    title="working-point prior",
    doc=(
        "Sample = one distinct working point (instrument-day × material × size × bias × line "
        "time) used on a frame whose material is known. Input = material/adsorbate/genre, "
        "field size, pixels, controller, temperature. COMMENT is stripped (it holds the "
        "parameters). Labels = bias, line time, speed; setpoint_a only when the raw headers "
        "were read (the index's 'setpoint' column is the controller ON flag, not the setpoint)."
    ),
    cols=_common(material_role=ROLE_INPUT) + (
        Col("adsorbate", "str", ROLE_INPUT, "overlayer parsed from COMMENT, '' if none"),
        Col("size_nm", "float", ROLE_INPUT, "scan range x, nm"),
        Col("nx", "int", ROLE_INPUT, "pixels per line"),
        Col("controller", "str", ROLE_INPUT, "Z-controller name (log Current / Frequency …)"),
        Col("z_feedback", "str", ROLE_INPUT,
            "Z feedback on/off while scanning (off + 0 V = constant-height qPlus frames, 5 % of the corpus)",
            choices=("on", "off", "unknown")),
        Col("rec_temp_k", "float", ROLE_INPUT, "REC_TEMP from the header, K", nullable=True),
        Col("bias_v", "float", ROLE_LABEL, "bias the operator used, V"),
        Col("t_fwd_s", "float", ROLE_LABEL, "forward line time, s"),
        Col("speed_nm_s", "float", ROLE_LABEL, "size_nm / t_fwd_s"),
        Col("setpoint_a", "float", ROLE_LABEL, "current setpoint from the raw header, A (NaN unless read)",
            nullable=True),
        Col("first_path", "str", ROLE_META, "first frame of this working point"),
        Col("n_frames", "int", ROLE_META, "frames taken at this working point"),
        Col("n_complete", "int", ROLE_META, "of which complete"),
    ),
    leak_forbidden=frozenset({"comment", "path", "rel_path"}),
)

SPECS: dict[str, ManifestSpec] = {"T1": T1, "T2": T2, "T3": T3, "T4": T4}


# ── validation ───────────────────────────────────────────────────────────────
def _dtype_ok(series, dtype: str) -> bool:
    from pandas.api import types as pt
    if dtype == "int":
        return pt.is_integer_dtype(series)
    if dtype == "float":
        return pt.is_float_dtype(series) or pt.is_integer_dtype(series)
    if dtype == "bool":
        return pt.is_bool_dtype(series)
    return pt.is_string_dtype(series) or pt.is_object_dtype(series)


def validate(df, spec: ManifestSpec) -> list[str]:
    """Return the list of problems (empty ⇒ the DataFrame conforms to ``spec``)."""
    problems: list[str] = []
    have = list(df.columns)
    missing = [n for n in spec.names if n not in have]
    extra = [n for n in have if n not in spec.names]
    if missing:
        problems.append(f"missing columns: {missing}")
    if extra:
        problems.append(f"undeclared columns: {extra}")
    for c in spec.cols:
        if c.name not in have:
            continue
        s = df[c.name]
        if not _dtype_ok(s, c.dtype):
            problems.append(f"{c.name}: dtype {s.dtype} is not {c.dtype}")
        if not c.nullable and s.isna().any():
            problems.append(f"{c.name}: {int(s.isna().sum())} nulls in a non-nullable column")
        if c.choices is not None:
            bad = sorted(set(s.dropna().unique()) - set(c.choices))
            if bad:
                problems.append(f"{c.name}: values outside {c.choices}: {bad[:5]}")
    if "sample_id" in have and df["sample_id"].duplicated().any():
        problems.append(f"sample_id: {int(df['sample_id'].duplicated().sum())} duplicates")
    if "group_key" in have and "split" in have and len(df):
        n_splits = df.groupby("group_key")["split"].nunique()
        multi = n_splits[n_splits > 1]
        if len(multi):
            problems.append(f"split: {len(multi)} groups straddle splits, e.g. {multi.index[0]!r}")
        wrong = df[df["split"] != df["group_key"].map(split_of)]
        if len(wrong):
            problems.append(f"split: {len(wrong)} rows disagree with split_of(group_key)")
    return problems


def assert_valid(df, spec: ManifestSpec) -> None:
    problems = validate(df, spec)
    if problems:
        raise ValueError(f"{spec.task} manifest invalid:\n  " + "\n  ".join(problems))
