"""
Data layer for the personalised LSTM-autoencoder relapse-detection pipeline.

Owns every leakage-sensitive decision:
  * patient / cohort / fold discovery on disk,
  * the per-CV-iteration split (4 remission folds train, the held-out remission
    fold split temporally into val-normal / test-normal, the relapse timeline
    split temporally into a validation half and a test half),
  * per-patient imputation (median) and scaling (standardisation), fit on that
    patient's training remission bins only, then frozen and reused for val/test,
  * the impute -> scale -> window order, producing overlapping 2-hour windows
    (24 bins, stride 12 bins, 50 % overlap) that never span a day boundary or an
    internal time gap.

All data, remission and relapse alike, is ordered by real time before any split,
so no split is decided by fold index.

Imported by train.py, evaluate_unimodal.py and evaluate_gated_fusion.py. It holds
the constants (FEATURE_COLS, window geometry) that must be identical at train and
test time.

Expected on-disk layout (output of feature extraction):
    <features_dir>/<cohort>/<user_id>/<fold>/<week>/features.csv
with cohort in {"remission", "relapse"}, 5 remission folds and 2 relapse folds
per patient.
"""

from __future__ import annotations

import re
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("dataset")

# --- Constants binding train and test time ----------------------------------
REMISSION = "remission"
RELAPSE   = "relapse"

N_REMISSION_FOLDS = 5
N_RELAPSE_FOLDS   = 2

# 2-hour window, 1-hour overlap, on a 5-minute bin grid.
WINDOW_BINS = 24            # 2 h / 5 min
STRIDE_BINS = 12            # 1 h / 5 min  -> 50 % overlap
BIN_MINUTES = 5

# Time-of-day regime. Default night is 22:00 -> 07:00 (wraps midnight).
# Expressed as minute-of-day so the wrap is easy to test.
TOD_CHOICES = ("day", "night", "both")
NIGHT_START_MIN = 22 * 60   # 22:00
NIGHT_END_MIN   = 7 * 60    # 07:00

# Physiological and time-encoding columns. Identifiers, coverage and per-day
# diagnostics are deliberately excluded.
FEATURE_COLS: List[str] = [
    "acc_X", "acc_X_std", "acc_Y", "acc_Y_std", "acc_Z", "acc_Z_std",
    "acc_X_zcr", "acc_Y_zcr", "acc_Z_zcr",
    "gyr_X", "gyr_X_std", "gyr_Y", "gyr_Y_std", "gyr_Z", "gyr_Z_std",
    "gyr_X_zcr", "gyr_Y_zcr", "gyr_Z_zcr",
    "heartRate_mean", "heartRate_std",
    "rRInterval_mean", "rRInterval_rmssd", "rRInterval_sdnn",
    "rRInterval_lombscargle_hf", "rRInterval_lombscargle_lf",
    "rRInterval_lombscargle_lf_hf",
    "rRInterval_sd1", "rRInterval_sd2", "rRInterval_sd1_sd2",
    "sin_t", "cos_t", "sleep_main_duration_h",
    "sleep_main_onset_sin",
    "sleep_main_onset_cos",
    "sleep_fragmentation",
    "sleep_onset_std_min",
    "activity_total_acc",
]
# Columns that identify a single calendar day uniquely within a patient.
_DAY_KEY_COLS = ["cohort", "fold", "week", "day_offset"]


# --- Time-of-day filtering ---------------------------------------------------
def parse_clock(s) -> int:
    """
    Parse a clock string into minute-of-day. Accepts 'HH:MM', 'HH.MM', or 'HH'
    (e.g. '22:00', '22.00', '22', '7', '07:00'). The separator is HOURS:MINUTES,
    not a decimal, so '7.30' means 07:30.
    """
    s = str(s).strip()
    if ":" in s:
        h, m = s.split(":", 1)
    elif "." in s:
        h, m = s.split(".", 1)
    else:
        h, m = s, "0"
    return (int(h) % 24) * 60 + (int(m) % 60)


def _tod_mask(timestamps,
              regime: str,
              night_start_min: int,
              night_end_min: int) -> np.ndarray:
    """
    Boolean mask selecting the bins that belong to `regime`.
    Night wraps midnight whenever night_start_min > night_end_min
    (the default 22:00 -> 07:00 does).
    """
    t = pd.to_datetime(np.asarray(timestamps))
    mod = np.asarray(t.hour) * 60 + np.asarray(t.minute)   # minute-of-day
    if night_start_min <= night_end_min:
        is_night = (mod >= night_start_min) & (mod < night_end_min)
    else:                                                   # wraps midnight
        is_night = (mod >= night_start_min) | (mod < night_end_min)
    if regime == "night":
        return is_night
    if regime == "day":
        return ~is_night
    return np.ones(len(mod), dtype=bool)                    # "both"


def _filter_tod(df: pd.DataFrame,
                regime: str,
                night_start_min: int,
                night_end_min: int) -> pd.DataFrame:
    """
    Keep only bins whose Timestamp falls in `regime`. 'both' is a no-op, so
    train_tod='both' reproduces the original (un-split) pipeline exactly.

    The downstream windower groups by calendar day and rejects any window with
    an internal time gap (span == expected), so a night filter cleanly yields a
    late-evening chunk and an early-morning chunk per day with no window ever
    bridging the 07:00 -> 22:00 daytime gap.
    """
    if df.empty or regime == "both":
        return df
    mask = _tod_mask(df["Timestamp"].to_numpy(), regime, night_start_min, night_end_min)
    return df[mask].copy()


# --- Natural sort, so week_10 orders after week_2 ---------------------------
def _natkey(name: str):
    m = re.findall(r"\d+", name)
    return (int(m[-1]) if m else 10**9, name)


def _extract_week_number(week_str: str) -> int:
    """
    Extract integer from week folder name.
    Examples: 'week_12' -> 12, '12' -> 12, 'week_1' -> 1
    """
    match = re.search(r'(\d+)', str(week_str))
    return int(match.group(1)) if match else 0


# --- Discovery ---------------------------------------------------------------
def _folds_of(features_dir: Path, cohort: str, patient: str) -> List[str]:
    base = features_dir / cohort / patient
    if not base.is_dir():
        return []
    return sorted((d.name for d in base.iterdir() if d.is_dir()), key=_natkey)


def discover_patients(features_dir: Path) -> List[str]:
    """Patients that have BOTH the expected remission and relapse fold counts."""
    rem_base = features_dir / REMISSION
    rel_base = features_dir / RELAPSE
    rem = {d.name for d in rem_base.iterdir() if d.is_dir()} if rem_base.is_dir() else set()
    rel = {d.name for d in rel_base.iterdir() if d.is_dir()} if rel_base.is_dir() else set()
    patients = []
    for p in sorted(rem & rel, key=_natkey):
        nrem = len(_folds_of(features_dir, REMISSION, p))
        nrel = len(_folds_of(features_dir, RELAPSE, p))
        if nrem == N_REMISSION_FOLDS and nrel == N_RELAPSE_FOLDS:
            patients.append(p)
        else:
            log.warning("Skipping patient %s: %d remission folds, %d relapse folds "
                        "(expected %d / %d)", p, nrem, nrel,
                        N_REMISSION_FOLDS, N_RELAPSE_FOLDS)
    return patients


# --- Loading -----------------------------------------------------------------
def _load_fold_bins(features_dir: Path, cohort: str, patient: str, fold: str) -> pd.DataFrame:
    """Concatenate every week's features.csv under one fold into one bin table."""
    fold_dir = features_dir / cohort / patient / fold
    frames = []
    for csv_path in sorted(fold_dir.rglob("features.csv")):
        try:
            df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        except Exception as exc:
            log.warning("Could not read %s: %s", csv_path, exc)
            continue
        # Guarantee the feature columns exist (a week may lack some); fill absent
        # ones with NaN so imputation handles them uniformly.
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = np.nan
        df = df[df["Timestamp"].notna()].copy()
        if df.empty:
            continue
        df["cohort"] = cohort
        df["fold"]   = fold
        df["week"]   = csv_path.parent.name
        if "day_offset" not in df.columns:
            df["day_offset"] = -1
        frames.append(df[_DAY_KEY_COLS + ["Timestamp", "calendar_date"]
                         if "calendar_date" in df.columns
                         else _DAY_KEY_COLS + ["Timestamp"]].join(df[FEATURE_COLS]))
    if not frames:
        return pd.DataFrame(columns=_DAY_KEY_COLS + ["Timestamp"] + FEATURE_COLS)
    return pd.concat(frames, ignore_index=True)


def _load_all_weeks_chronological(features_dir: Path, cohort: str, patient: str) -> pd.DataFrame:
    """
    Load every week for a patient across every fold and sort chronologically,
    rebuilding the patient's real timeline rather than its fold assignment. This
    is what makes the relapse validation/test split temporal.
    """
    base = features_dir / cohort / patient
    if not base.is_dir():
        return pd.DataFrame()

    all_weeks = []
    week_dirs = []

    for fold_dir in base.iterdir():
        if not fold_dir.is_dir():
            continue
        for week_dir in fold_dir.iterdir():
            if not week_dir.is_dir():
                continue
            week_num = _extract_week_number(week_dir.name)
            week_dirs.append((week_num, fold_dir.name, week_dir))

    week_dirs.sort(key=lambda x: x[0])

    for week_num, fold_name, week_dir in week_dirs:
        csv_path = week_dir / "features.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
        except Exception as exc:
            log.warning("Could not read %s: %s", csv_path, exc)
            continue

        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = np.nan

        df = df[df["Timestamp"].notna()].copy()
        if df.empty:
            continue

        df["cohort"] = cohort
        df["fold"] = fold_name
        df["week"] = week_dir.name
        if "day_offset" not in df.columns:
            df["day_offset"] = -1

        df["_chrono_week_num"] = week_num

        all_weeks.append(df[_DAY_KEY_COLS + ["Timestamp", "calendar_date", "_chrono_week_num"]
                           if "calendar_date" in df.columns
                           else _DAY_KEY_COLS + ["Timestamp", "_chrono_week_num"]].join(df[FEATURE_COLS]))

    if not all_weeks:
        return pd.DataFrame(columns=_DAY_KEY_COLS + ["Timestamp"] + FEATURE_COLS)

    return pd.concat(all_weeks, ignore_index=True)


# --- Per-patient artifacts: median imputation then standardisation ----------
def fit_artifacts(train_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = train_df[FEATURE_COLS].to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    imp = arr.copy()
    nan_mask = np.isnan(imp)
    imp[nan_mask] = np.take(med, np.where(nan_mask)[1])
    mean = imp.mean(axis=0)
    std  = imp.std(axis=0)
    std[std == 0] = 1.0
    return med, mean, std


def _transform(arr: np.ndarray, art: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    med, mean, std = art
    out = arr.copy()
    nan_mask = np.isnan(out)
    out[nan_mask] = np.take(med, np.where(nan_mask)[1])
    return (out - mean) / std


# --- Windowing ---------------------------------------------------------------
def _day_items(df: pd.DataFrame,
               art: Tuple[np.ndarray, np.ndarray, np.ndarray],
               patient: str,
               label: int) -> List[Dict]:
    """
    Transform a split's bins for one patient, then slide windows within each day.
    Returns one item per day that yields >= 1 valid window:
        {patient, day_key, label, windows: np.ndarray (k, WINDOW_BINS, F),
         calendar_date: str | None}

    calendar_date is the ISO date string (e.g. "2022-02-28") for that day, stored
    so the evaluation scripts can sort items chronologically and detect real
    calendar gaps between consecutive days. It is None when the column is absent
    from the source CSV.
    """
    if df.empty:
        return []
    X = _transform(df[FEATURE_COLS].to_numpy(dtype=float), art)
    df = df.reset_index(drop=True)
    ts_all = df["Timestamp"].to_numpy()
    expected = np.timedelta64(BIN_MINUTES * (WINDOW_BINS - 1), "m")

    has_cal = "calendar_date" in df.columns

    items: List[Dict] = []
    for key, idx in df.groupby(_DAY_KEY_COLS, sort=False).indices.items():
        pos = np.asarray(idx)
        order = np.argsort(ts_all[pos])
        pos = pos[order]
        ts = ts_all[pos]
        wins = []
        for s in range(0, len(pos) - WINDOW_BINS + 1, STRIDE_BINS):
            span = ts[s + WINDOW_BINS - 1] - ts[s]
            if span == expected:                     # contiguous, no internal gap
                wins.append(X[pos[s:s + WINDOW_BINS]])
        if wins:
            key_t = key if isinstance(key, tuple) else (key,)

            cal_date = None
            if has_cal:
                # All bins in a day share one calendar_date, so take the first
                # non-null value.
                vals = df.loc[pos, "calendar_date"].dropna()
                if not vals.empty:
                    cal_date = str(vals.iloc[0])

            items.append({
                "patient":       patient,
                "day_key":       (patient,) + tuple(key_t),
                "label":         label,
                "windows":       np.stack(wins).astype(np.float32),
                "calendar_date": cal_date,
            })
    return items


# --- Fold assembly -----------------------------------------------------------
class FoldData:
    """Container for one experiment (held-out remission fold + relapse ordering)."""
    def __init__(self, train_windows_by_patient, val_items, test_items, artifacts,
                 feat_dim, half_assignment, fold_idx, relapse_order):
        self.train_windows_by_patient = train_windows_by_patient  # patient -> (n,T,F)
        self.val_items = val_items              # list of day-item dicts
        self.test_items = test_items            # list of day-item dicts
        self.artifacts = artifacts              # patient -> (med, mean, std)
        self.feat_dim = feat_dim
        self.half_assignment = half_assignment  # patient -> bool (swap halves?)
        self.fold_idx = fold_idx
        self.relapse_order = relapse_order


def _ordered_day_keys(df: pd.DataFrame) -> List[tuple]:
    """
    Order days by real time. Week numbers are parsed as integers, so week_10
    follows week_2 rather than preceding it, and calendar_date takes precedence
    whenever the column is present.
    """
    if df.empty:
        return []

    tmp = df.drop_duplicates(_DAY_KEY_COLS).copy()

    tmp['_week_num'] = tmp['week'].apply(_extract_week_number)

    # If calendar_date exists, use it as primary sort (most accurate)
    if "calendar_date" in tmp.columns:
        tmp['_d'] = pd.to_datetime(tmp["calendar_date"], errors="coerce")
        # Sort by: date -> week number -> day_offset (chronological order)
        tmp = tmp.sort_values(["_d", "_week_num", "day_offset"],
                             na_position="last")
    else:
        # Fallback: sort by week number -> day_offset (chronological)
        tmp = tmp.sort_values(["_week_num", "day_offset"])

    return [tuple(r) for r in tmp[_DAY_KEY_COLS].to_numpy()]


def _subset_by_days(df: pd.DataFrame, day_keys: set) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df[_DAY_KEY_COLS].apply(lambda r: tuple(r) in day_keys, axis=1)
    return df[mask].copy()


def _split_data_chronologically(df: pd.DataFrame,
                                val_ratio: float = 0.5,
                                rng: Optional[random.Random] = None,
                                patient: str = None,
                                features_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into two halves chronologically.
    Uses the chronological ordering from _ordered_day_keys.
    """
    if df.empty:
        return df.copy(), df.copy()

    day_order = _ordered_day_keys(df)
    cut = int(len(day_order) * val_ratio)

    first_keys = set(day_order[:cut])
    second_keys = set(day_order[cut:])

    return _subset_by_days(df, first_keys), _subset_by_days(df, second_keys)


def build_fold_data(features_dir,
                    fold_idx: int,
                    relapse_order: int = 0,
                    precomputed_artifacts: Optional[Dict] = None,
                    half_assignment: Optional[Dict] = None,
                    rng: Optional[random.Random] = None,
                    train_tod: str = "both",
                    test_tod: str = "both",
                    night_start_min: int = NIGHT_START_MIN,
                    night_end_min: int = NIGHT_END_MIN) -> FoldData:
    """
    Assemble train/val/test for one experiment.

    An experiment is identified by:
      * `fold_idx` (0..N_REMISSION_FOLDS-1), the remission fold held out, and
      * `relapse_order` (0 or 1), which half of the relapse timeline is used for
        validation and which for test:
            0 -> first half validation, second half test
            1 -> first half test, second half validation

    Per patient:
      * train = the 4 remission folds other than position `fold_idx`,
      * the held-out remission fold is split into two contiguous temporal halves,
        and a per-patient coin flip decides which half is val-normal and which is
        test-normal,
      * every relapse week across both relapse folds is loaded, sorted by real
        time and split into two temporal halves, assigned by `relapse_order`.

    Both relapse subsets are therefore temporally coherent and no split follows
    fold order.

    `half_assignment` (patient -> bool) reuses a recorded coin flip instead of
    drawing a new one. `precomputed_artifacts` (patient -> (med, mean, std))
    reuses the fitted scaling instead of refitting it. Together they let the
    evaluation scripts reproduce an experiment exactly while learning nothing.

    Time of day:
      * `train_tod` filters the training windows, the artifact fit, and the
        validation normal and relapse days used for early stopping and
        thresholding,
      * `test_tod` filters the test normal and relapse days used for the final
        metrics.
      The chronological val/test split is computed on unfiltered days first and
      each half is filtered afterwards, so the split and the half assignment are
      identical across regimes and stay reproducible. With train_tod='both' the
      filter is a no-op. The five meaningful regime pairs are the matched ones
      (day/day, night/night, both/both) plus a both-trained model evaluated on
      day or on night.
    """
    features_dir = Path(features_dir)
    patients = discover_patients(features_dir)
    if not patients:
        raise RuntimeError(f"No usable patients found under {features_dir}")
    if rng is None:
        rng = random.Random()

    train_windows_by_patient: Dict[str, np.ndarray] = {}
    val_items: List[Dict] = []
    test_items: List[Dict] = []
    artifacts: Dict = {}
    half_out: Dict = {}

    for p in patients:
        rem = _folds_of(features_dir, REMISSION, p)
        held = rem[fold_idx]
        train_rem = [f for k, f in enumerate(rem) if k != fold_idx]

        train_df = pd.concat(
            [_load_fold_bins(features_dir, REMISSION, p, f) for f in train_rem],
            ignore_index=True,
        )
        if train_df.empty:
            log.warning("Patient %s has empty training remission; skipping.", p)
            continue

        # Artifacts are fit on the filtered set so scaling reflects the regime
        # the model is trained in.
        train_df = _filter_tod(train_df, train_tod, night_start_min, night_end_min)
        if train_df.empty:
            log.warning("Patient %s has no '%s' training bins; skipping.", p, train_tod)
            continue

        art = (precomputed_artifacts[p] if precomputed_artifacts
               else fit_artifacts(train_df))
        artifacts[p] = art

        # Kept per patient rather than pooled: one model is trained per patient.
        pw: List[np.ndarray] = []
        for it in _day_items(train_df, art, p, label=0):
            pw.extend(list(it["windows"]))
        if pw:
            train_windows_by_patient[p] = np.stack(pw).astype(np.float32)

        # Held-out remission fold, split into two temporal halves.
        held_df = _load_fold_bins(features_dir, REMISSION, p, held)
        first_half, second_half = _split_data_chronologically(held_df, val_ratio=0.5, rng=rng)

        if half_assignment is not None and p in half_assignment:
            swap = bool(half_assignment[p])
        else:
            swap = rng.random() < 0.5
        half_out[p] = swap

        if swap:
            val_norm_df, test_norm_df = second_half, first_half
        else:
            val_norm_df, test_norm_df = first_half, second_half

        # val-normal follows the train regime because it drives early stopping
        # and thresholding, test-normal follows the test regime.
        val_norm_df  = _filter_tod(val_norm_df,  train_tod, night_start_min, night_end_min)
        test_norm_df = _filter_tod(test_norm_df, test_tod,  night_start_min, night_end_min)

        # Whole relapse timeline, split into two temporal halves.
        relapse_timeline = _load_all_weeks_chronological(features_dir, RELAPSE, p)

        if relapse_timeline.empty:
            log.warning("Patient %s has empty relapse data; skipping.", p)
            val_items += _day_items(val_norm_df, art, p, label=0)
            test_items += _day_items(test_norm_df, art, p, label=0)
            continue

        rel_first_half, rel_second_half = _split_data_chronologically(
            relapse_timeline, val_ratio=0.5, rng=rng
        )

        if relapse_order == 0:
            val_rel_df, test_rel_df = rel_first_half, rel_second_half
        else:
            val_rel_df, test_rel_df = rel_second_half, rel_first_half

        val_rel_df  = _filter_tod(val_rel_df,  train_tod, night_start_min, night_end_min)
        test_rel_df = _filter_tod(test_rel_df, test_tod,  night_start_min, night_end_min)

        val_items += _day_items(val_norm_df, art, p, label=0)
        val_items += _day_items(val_rel_df, art, p, label=1)
        test_items += _day_items(test_norm_df, art, p, label=0)
        test_items += _day_items(test_rel_df, art, p, label=1)

    if not train_windows_by_patient:
        raise RuntimeError(
            f"No training windows produced for fold {fold_idx} order {relapse_order}")

    total_train = sum(len(v) for v in train_windows_by_patient.values())
    log.info("Experiment fold=%d order=%d: %d train windows across %d patients | "
             "%d val day-items | %d test day-items", fold_idx, relapse_order,
             total_train, len(train_windows_by_patient), len(val_items), len(test_items))
    return FoldData(train_windows_by_patient, val_items, test_items, artifacts,
                    len(FEATURE_COLS), half_out, fold_idx, relapse_order)
