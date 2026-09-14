"""
Evaluation of the per-patient LSTM autoencoders trained by train.py.

Loads the per-experiment checkpoint bundles, each holding one autoencoder per
patient, scores every patient with their own model and their own
validation-tuned threshold, and reports per-patient AUROC / AUPRC / AVG / F1, the
per-experiment mean across patients, and the mean and standard deviation across
the 10 experiments. AUROC is the metric to trust. AUPRC and F1 must be read
against prevalence.

Window geometry is read from the checkpoint (window_bins / stride_bins), so
dataset.py need not be edited before testing and checkpoints trained at different
window lengths can be evaluated in one run.

Day aggregation (--agg {mean,median,p90,p95,max})
    Reduces a day's window errors to one score. mean assumes the whole day is
    abnormal, whereas the tail statistics reward a few sharply abnormal windows
    and often raise AUPRC when relapse shows as a handful of bad hours.

Day-score smoothing (--smooth-days K, --smooth-mode MODE, --gap-days G)
    Relapse builds gradually, so a single day's reconstruction error is noisy.
    Per-day scores are filtered with a causal trailing window of width K.

    All test day-items for a patient are first sorted by calendar_date, then
    split into contiguous segments wherever the gap between consecutive days
    exceeds --gap-days. The filter is applied independently within each segment,
    so a long gap between the remission half and the relapse episode never
    contaminates episode scores with remission filter history, and a spike before
    a short recording gap cannot inflate scores after it. The same gap-aware
    smoothing is applied to validation items so the tuned threshold is comparable
    to the smoothed test scores. K=1 means no smoothing, only chronological
    ordering.

    Modes: mean (default), median, ewma (span=K), linear (triangular trailing
    weights) and trimmed (mean after trimming --trim from each end).

Per-patient anomaly signature (--signature)
    The scalar score is a mean of squared error over time and features. This pass
    reopens the feature axis (model.reconstruction_error_per_feature), partitions
    the physiological features into
        heart_rate : heartRate_*
        hrv        : rRInterval_*
        motion     : acc_* and gyr_*
    and asks, per patient, which group separates relapse from normal. The two
    circadian encodings sin_t and cos_t stay inside the scalar score but are not
    reported as a group. The same gap-aware smoothing is applied per group. The
    pass is read-only and changes no headline metric.

Run:
    # headline metrics
    python evaluate_unimodal.py --features-dir /gpu-data/eprevention/ntsal/features \
                   --ckpt-dir ./checkpoints --smooth-days 7 --smooth-mode ewma

    # per-patient anomaly signature
    python evaluate_unimodal.py --features-dir /gpu-data/eprevention/ntsal/features \
                   --ckpt-dir ./checkpoints --smooth-days 7 --smooth-mode ewma \
                   --signature --fig-dir ./figures

    # score-level ensemble across experiments (one number per patient)
    python evaluate_unimodal.py --features-dir /gpu-data/eprevention/ntsal/features \
                   --ckpt-dir ./checkpoints --smooth-days 7 --smooth-mode ewma \
                   --ensemble

    # diagnostics: is smoothing helping, or reading temporal layout?
    python evaluate_unimodal.py --features-dir /gpu-data/eprevention/ntsal/features \
                   --ckpt-dir ./checkpoints --smooth-days 7 --diagnose
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import dataset as ds
from model import (LSTMAutoencoder, reconstruction_error,
                   reconstruction_error_per_feature)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

_AGG = {
    "mean":   np.mean,
    "median": np.median,
    "p90":    lambda a: np.percentile(a, 90),
    "p95":    lambda a: np.percentile(a, 95),
    "max":    np.max,
}
SMOOTH_MODES = ("mean", "median", "ewma", "linear", "trimmed")

# Reported physiological groups for the anomaly signature.
GROUP_ORDER = ("heart_rate", "hrv", "motion")
GROUP_LABEL = {"heart_rate": "Heart rate", "hrv": "HRV", "motion": "Motion"}


# --- Window-level scoring ----------------------------------------------------

@torch.no_grad()
def _window_errors(model, windows, device, batch_size=512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(windows[i:i + batch_size]).to(device)
        out.append(reconstruction_error(x, model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


@torch.no_grad()
def _window_errors_per_feature(model, windows, device, batch_size=512) -> np.ndarray:
    """Per-window, per-feature errors: (n_windows, F). Empty -> (0, F)."""
    model.eval()
    F = int(model.input_dim)
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.from_numpy(windows[i:i + batch_size]).to(device)
        out.append(reconstruction_error_per_feature(x, model(x)).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, F), dtype=np.float32)


def _safe_int(x, default=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# --- Day-level scoring, one score per day-item for one patient --------------

def _patient_raw_records(model, items, patient, device, agg):
    """
    Score every day-item belonging to `patient` and return a list of dicts:
        {score, label, calendar_date (pd.Timestamp | NaT)}

    Items from different cohorts/folds are mixed together; chronological
    sorting and gap segmentation happen downstream in _gap_aware_smooth.
    """
    aggf = _AGG[agg]
    records = []
    for it in items:
        if it["patient"] != patient:
            continue
        errs = _window_errors(model, it["windows"], device)
        if errs.size == 0:
            continue
        raw_date = it.get("calendar_date")                       # "2022-02-28" or None
        cal = pd.to_datetime(raw_date, errors="coerce")          # NaT if missing
        records.append({
            "score":         float(aggf(errs)),
            "label":         int(it["label"]),
            "coverage":      int(errs.size),
            "calendar_date": cal,
        })
    return records


# --- Causal filter kernels ---------------------------------------------------

def _filter_series(vals: np.ndarray, k: int, mode: str, trim: float) -> np.ndarray:
    """
    Apply a causal (trailing) filter of width k to a 1-D array.
    At position i the filter sees only vals[max(0,i-k+1) : i+1].
    k <= 1 -> identity.
    """
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if k <= 1 or n == 0:
        return vals.copy()

    if mode == "ewma":
        alpha = 2.0 / (k + 1.0)            # span=k convention
        out = np.empty(n)
        acc = vals[0]
        out[0] = acc
        for i in range(1, n):
            acc = alpha * vals[i] + (1.0 - alpha) * acc
            out[i] = acc
        return out

    out = np.empty(n)
    for i in range(n):
        w = vals[max(0, i - k + 1): i + 1]
        if mode == "mean":
            out[i] = w.mean()
        elif mode == "median":
            out[i] = np.median(w)
        elif mode == "linear":
            ww = np.arange(1, len(w) + 1, dtype=float)   # most recent = highest weight
            out[i] = float((ww * w).sum() / ww.sum())
        elif mode == "trimmed":
            m = len(w)
            c = int(np.floor(m * trim))
            if c > 0 and m - 2 * c >= 1:
                out[i] = np.sort(w)[c: m - c].mean()
            else:
                out[i] = w.mean()
        else:
            out[i] = w.mean()
    return out


# --- Gap-aware smoothing: sort, split at calendar gaps, filter each segment -

def _gap_aware_smooth(records: list, k: int, mode: str,
                      trim: float, gap_days: int) -> list:
    """
    Sort records by calendar_date (undated records go last and form isolated
    segments, still scored but never blended with dated ones), split into
    contiguous segments wherever the gap between consecutive dates exceeds
    `gap_days`, filter each segment independently, and return the sorted list
    with smoothed scores.

    With k <= 1 the records are sorted but the scores are unchanged. This is the
    single smoothing entry point for every path in this file.
    """
    if not records:
        return records

    records = sorted(
        records,
        key=lambda r: (
            r["calendar_date"] is pd.NaT or pd.isna(r["calendar_date"]),
            r["calendar_date"],
        ),
    )

    if k <= 1:
        return records

    segments: list[list[dict]] = []
    current: list[dict] = [records[0]]

    for prev, cur in zip(records[:-1], records[1:]):
        prev_d, cur_d = prev["calendar_date"], cur["calendar_date"]
        if pd.isna(prev_d) or pd.isna(cur_d):
            gap = True
        else:
            gap = (cur_d - prev_d).days > gap_days
        if gap:
            segments.append(current)
            current = [cur]
        else:
            current.append(cur)
    segments.append(current)

    smoothed: list[dict] = []
    for seg in segments:
        vals = np.array([r["score"] for r in seg], dtype=float)
        sm   = _filter_series(vals, k, mode, trim)
        for rec, s in zip(seg, sm):
            smoothed.append({**rec, "score": float(s)})

    return smoothed


def _recs_to_arrays(recs):
    if not recs:
        return np.array([]), np.array([])
    s = np.array([r["score"] for r in recs], dtype=float)
    y = np.array([r["label"] for r in recs], dtype=int)
    return s, y


def _scored(model, items, patient, device, agg, smooth_days, smooth_mode, trim, gap_days):
    """Full pipeline: window errors → day agg → chronological sort →
    gap segmentation → causal filter → (scores, labels) arrays."""
    recs = _patient_raw_records(model, items, patient, device, agg)
    recs = _gap_aware_smooth(recs, k=smooth_days, mode=smooth_mode,
                             trim=trim, gap_days=gap_days)
    return _recs_to_arrays(recs)


# --- Threshold and metrics --------------------------------------------------

def _best_f1_threshold(scores, labels) -> float:
    labels = np.asarray(labels)
    if len(scores) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        normal = scores[labels == 0]
        return float(np.percentile(normal, 95)) if normal.size else 0.0
    cands = np.unique(scores)
    best_t, best_f1 = cands[0], -1.0
    for t in cands:
        pred = (scores >= t).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _metrics(test_scores, test_labels, threshold):
    y = np.asarray(test_labels); s = np.asarray(test_scores)
    prevalence = float(y.mean()) if len(y) else float("nan")
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        auroc = auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y, s))
        auprc = float(average_precision_score(y, s))
    avg = float((auroc + auprc) / 2.0)
    pred = (s >= threshold).astype(int)
    f1 = float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan")
    return {"auroc": auroc, "auprc": auprc, "avg": avg, "f1": f1,
            "prevalence": prevalence, "n_days": int(len(y))}


def _mean_across_patients(per_patient):
    aurocs = [r["auroc"] for r in per_patient.values() if r["auroc"] == r["auroc"]]
    auprcs = [r["auprc"] for r in per_patient.values() if r["auprc"] == r["auprc"]]
    avgs   = [r["avg"]   for r in per_patient.values() if r["avg"]   == r["avg"]]
    f1s    = [r["f1"]    for r in per_patient.values()]
    return (float(np.mean(aurocs)) if aurocs else float("nan"),
            float(np.mean(auprcs)) if auprcs else float("nan"),
            float(np.mean(avgs))   if avgs   else float("nan"),
            float(np.mean(f1s))    if f1s    else float("nan"))


# --- Checkpoint utilities ----------------------------------------------------

def _apply_checkpoint_geometry(bundle):
    wb = int(bundle.get("window_bins", ds.WINDOW_BINS))
    sb = int(bundle.get("stride_bins", ds.STRIDE_BINS))
    ds.WINDOW_BINS = wb
    ds.STRIDE_BINS = sb
    return wb, sb


def _is_clean(train_tod: str, test_tod: str) -> bool:
    """The five meaningful regime pairs: matched (day/day, night/night,
    both/both) or a both-trained model evaluated on day or night."""
    if test_tod == train_tod:
        return True
    return train_tod == "both"


def _load_bundle_and_fold(features_dir, ckpt_path, device, test_tod):
    """Shared setup for every evaluation path: load the checkpoint, apply its
    window geometry and regime, and rebuild the exact fold, reusing the recorded
    artifacts and half-assignment so nothing is learned at test time."""
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Fail fast if the checkpoint was trained on a different feature set from the
    # one dataset.py now defines. Editing FEATURE_COLS does not retrofit existing
    # checkpoints: the weights, model_kwargs input_dim and stored artifacts are
    # all bound to the training feature list.
    ckpt_feats = list(bundle.get("feature_cols", []))
    if ckpt_feats and ckpt_feats != list(ds.FEATURE_COLS):
        only_ckpt = [c for c in ckpt_feats if c not in set(ds.FEATURE_COLS)]
        only_now  = [c for c in ds.FEATURE_COLS if c not in set(ckpt_feats)]
        raise RuntimeError(
            f"Feature mismatch for {ckpt_path.name}: checkpoint has "
            f"{len(ckpt_feats)} features but dataset.py defines "
            f"{len(ds.FEATURE_COLS)}. Retrain so the checkpoint matches the "
            f"current FEATURE_COLS.\n  only in checkpoint: {only_ckpt}\n"
            f"  only in dataset:    {only_now}")

    fold_idx = bundle["fold_idx"]; relapse_order = bundle["relapse_order"]
    wb, sb = _apply_checkpoint_geometry(bundle)

    train_tod = bundle.get("train_tod", "both")
    nsm = int(bundle.get("night_start_min", ds.NIGHT_START_MIN))
    nem = int(bundle.get("night_end_min", ds.NIGHT_END_MIN))
    eff_test_tod = train_tod if test_tod is None else test_tod
    if not _is_clean(train_tod, eff_test_tod):
        log.warning("CONFOUNDED PAIR train=%s test=%s: a single-regime model "
                    "evaluated on the other regime is out of distribution "
                    "(time encoding + scaling). Not a clean relapse AUROC.",
                    train_tod, eff_test_tod)

    fold = ds.build_fold_data(
        features_dir, fold_idx, relapse_order=relapse_order,
        precomputed_artifacts=bundle["artifacts"],
        half_assignment=bundle["half_assignment"],
        train_tod=train_tod, test_tod=eff_test_tod,
        night_start_min=nsm, night_end_min=nem,
    )
    return bundle, fold, fold_idx, relapse_order, train_tod, eff_test_tod, wb, sb


# --- Experiment evaluation ---------------------------------------------------

def evaluate_experiment(features_dir, ckpt_path, device, agg,
                        smooth_days, smooth_mode, trim, gap_days, test_tod):
    (bundle, fold, fold_idx, relapse_order, train_tod, eff_test_tod,
     wb, sb) = _load_bundle_and_fold(features_dir, ckpt_path, device, test_tod)

    log.info("=" * 70)
    log.info("Evaluating fold=%d order=%d  (%s)  train_tod=%s test_tod=%s  "
             "window=%d stride=%d  smooth=%d/%s  gap=%dd",
             fold_idx, relapse_order, ckpt_path.name, train_tod, eff_test_tod,
             wb, sb, smooth_days, smooth_mode, gap_days)

    per_patient = {}
    for p in sorted(bundle["state_dicts"], key=ds._natkey):
        model = LSTMAutoencoder(**bundle["model_kwargs"]).to(device)
        model.load_state_dict(bundle["state_dicts"][p])

        vs, vy = _scored(model, fold.val_items, p, device,
                         agg, smooth_days, smooth_mode, trim, gap_days)
        thr = _best_f1_threshold(vs, vy)

        ts, ty = _scored(model, fold.test_items, p, device,
                         agg, smooth_days, smooth_mode, trim, gap_days)
        per_patient[p] = _metrics(ts, ty, thr)

    return fold_idx, relapse_order, per_patient


# --- Per-patient anomaly signature ------------------------------------------

def _feature_groups():
    """Index sets into ds.FEATURE_COLS for the three reported groups. Built by
    name, so a reordering of FEATURE_COLS is tracked automatically. sin_t and
    cos_t are left unassigned: they stay inside the scalar score but are not a
    reported group."""
    groups = {g: [] for g in GROUP_ORDER}
    for i, c in enumerate(ds.FEATURE_COLS):
        if c.startswith("heartRate"):
            groups["heart_rate"].append(i)
        elif c.startswith("rRInterval"):
            groups["hrv"].append(i)
        elif c.startswith("acc_") or c.startswith("gyr_"):
            groups["motion"].append(i)
    return {g: np.asarray(idx, dtype=int) for g, idx in groups.items()}


def _agg_axis0(errs: np.ndarray, agg: str) -> np.ndarray:
    """Daily aggregation over windows, applied per feature (axis 0)."""
    if agg == "median":
        return np.median(errs, axis=0)
    if agg == "p90":
        return np.percentile(errs, 90, axis=0)
    if agg == "p95":
        return np.percentile(errs, 95, axis=0)
    if agg == "max":
        return errs.max(axis=0)
    return errs.mean(axis=0)


def _patient_group_raw_records(model, items, patient, device, agg, groups):
    """
    Per-DAY group scores for ONE patient. Each record carries:
        {label, calendar_date, heart_rate, hrv, motion}
    Calendar-date-aware gap smoothing is applied downstream in _gap_aware_smooth.
    """
    records = []
    for it in items:
        if it["patient"] != patient:
            continue
        errs = _window_errors_per_feature(model, it["windows"], device)   # (k, F)
        if errs.size == 0:
            continue
        day_feat = _agg_axis0(errs, agg)                                  # (F,)

        raw_date = it.get("calendar_date")
        cal = pd.to_datetime(raw_date, errors="coerce")

        rec = {
            "label":         int(it["label"]),
            "calendar_date": cal,
        }
        for g, gidx in groups.items():
            rec[g] = float(day_feat[gidx].sum()) if gidx.size else float("nan")
        records.append(rec)
    return records


def _gap_aware_smooth_groups(records: list, k: int, mode: str,
                              trim: float, gap_days: int) -> list:
    """
    Gap-aware causal smoothing applied independently to each group's per-day
    series. Same sorting and segmentation rule as _gap_aware_smooth, but the
    score field is per-group rather than scalar.
    """
    if not records or k <= 1:
        return sorted(
            records,
            key=lambda r: (
                r["calendar_date"] is pd.NaT or pd.isna(r["calendar_date"]),
                r["calendar_date"],
            ),
        )

    records = sorted(
        records,
        key=lambda r: (
            r["calendar_date"] is pd.NaT or pd.isna(r["calendar_date"]),
            r["calendar_date"],
        ),
    )

    seg_indices: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, len(records)):
        prev_d = records[i - 1]["calendar_date"]
        cur_d  = records[i]["calendar_date"]
        if pd.isna(prev_d) or pd.isna(cur_d):
            gap = True
        else:
            gap = (cur_d - prev_d).days > gap_days
        if gap:
            seg_indices.append(current)
            current = [i]
        else:
            current.append(i)
    seg_indices.append(current)

    out = [dict(r) for r in records]
    for seg in seg_indices:
        for g in GROUP_ORDER:
            vals = np.array([records[i][g] for i in seg], dtype=float)
            sm   = _filter_series(vals, k, mode, trim)
            for j, idx in enumerate(seg):
                out[idx][g] = float(sm[j])

    return out


METRIC_KEYS = ("auroc", "auprc", "avg", "contrast")


def _group_metrics(recs):
    """Per-group threshold-free metrics for one patient in one experiment: AUROC,
    AUPRC, AVG (their mean) and a relapse-versus-normal contrast in normal-day
    standard deviations. Returns (res, prevalence) with res keyed
    {group: {auroc, auprc, avg, contrast}}, or None if only one class is
    present."""
    y = np.array([r["label"] for r in recs], dtype=int)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return None
    prevalence = float(y.mean())
    pos = y == 1
    neg = y == 0
    res = {}
    for g in GROUP_ORDER:
        s = np.array([r[g] for r in recs], dtype=float)
        if not np.isfinite(s).all():
            res[g] = {k: float("nan") for k in METRIC_KEYS}
            continue
        auroc    = float(roc_auc_score(y, s))
        auprc    = float(average_precision_score(y, s))
        avg      = float((auroc + auprc) / 2.0)
        sd_n     = s[neg].std()
        sd_n     = sd_n if sd_n > 0 else 1.0
        contrast = float((s[pos].mean() - s[neg].mean()) / sd_n)
        res[g]   = {"auroc": auroc, "auprc": auprc, "avg": avg, "contrast": contrast}
    return res, prevalence


def evaluate_experiment_signature(features_dir, ckpt_path, device, agg,
                                  smooth_days, smooth_mode, trim, gap_days,
                                  test_tod, groups):
    (bundle, fold, fold_idx, relapse_order, train_tod, eff_test_tod,
     wb, sb) = _load_bundle_and_fold(features_dir, ckpt_path, device, test_tod)

    log.info("=" * 70)
    log.info("Signature fold=%d order=%d  (%s)  train_tod=%s test_tod=%s  "
             "window=%d stride=%d  smooth=%d/%s  gap=%dd",
             fold_idx, relapse_order, ckpt_path.name, train_tod, eff_test_tod,
             wb, sb, smooth_days, smooth_mode, gap_days)

    per_patient_group = {}
    for p in sorted(bundle["state_dicts"], key=ds._natkey):
        model = LSTMAutoencoder(**bundle["model_kwargs"]).to(device)
        model.load_state_dict(bundle["state_dicts"][p])

        recs = _patient_group_raw_records(model, fold.test_items, p, device, agg, groups)
        recs = _gap_aware_smooth_groups(recs, k=smooth_days, mode=smooth_mode,
                                        trim=trim, gap_days=gap_days)
        res = _group_metrics(recs) if recs else None
        if res is not None:
            gm, prevalence = res
            gm["_prevalence"] = prevalence
            per_patient_group[p] = gm
    return fold_idx, relapse_order, per_patient_group


# --- Reporting helpers -------------------------------------------------------

def _ms(vals):
    v = [x for x in vals if x == x]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def _save_heatmap(matrix, row_labels, col_labels, title, cbar_label, path, fmt="{:.2f}"):
    """Patients x groups heatmap. Skipped with a warning if matplotlib is
    unavailable, in which case the per-patient table is still logged."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib unavailable (%s); skipping figure %s.", exc, path)
        return
    m = np.asarray(matrix, dtype=float)
    nrow, ncol = m.shape
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * ncol, 1.2 + 0.42 * nrow))
    im = ax.imshow(m, aspect="auto", cmap="viridis")
    ax.set_xticks(range(ncol)); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=9)
    vmid = np.nanmin(m) + 0.5 * (np.nanmax(m) - np.nanmin(m)) if np.isfinite(m).any() else 0.0
    for i in range(nrow):
        for j in range(ncol):
            if np.isfinite(m[i, j]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                        fontsize=8, color="white" if m[i, j] < vmid else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  saved figure -> %s", path)


def run_signature(args, device, ckpt_dir):
    groups = _feature_groups()
    eff_test_tod = args.train_tod if args.test_tod is None else args.test_tod
    log.info("ANOMALY SIGNATURE | groups: %s | regime train_tod=%s test_tod=%s | "
             "agg=%s | smooth=%d/%s | gap=%dd",
             ", ".join(f"{g}({groups[g].size})" for g in GROUP_ORDER),
             args.train_tod, eff_test_tod, args.agg,
             args.smooth_days, args.smooth_mode, args.gap_days)

    acc      = defaultdict(lambda: {g: {k: [] for k in METRIC_KEYS} for g in GROUP_ORDER})
    prev_acc = defaultdict(list)
    winners  = defaultdict(list)
    n_exp    = 0

    for fold_idx in args.folds:
        for relapse_order in args.orders:
            ckpt_path = ckpt_dir / f"fold_{fold_idx}_rel_{relapse_order}_{args.train_tod}.pt"
            if not ckpt_path.exists():
                log.warning("Missing checkpoint %s — skipping.", ckpt_path)
                continue
            n_exp += 1
            _, _, per_patient_group = evaluate_experiment_signature(
                args.features_dir, ckpt_path, device, args.agg,
                args.smooth_days, args.smooth_mode, args.trim,
                args.gap_days, args.test_tod, groups)
            for p, gm in per_patient_group.items():
                prev_acc[p].append(gm.get("_prevalence", float("nan")))
                run_avgs = {}
                for g in GROUP_ORDER:
                    for k in METRIC_KEYS:
                        acc[p][g][k].append(gm[g][k])
                    if gm[g]["avg"] == gm[g]["avg"]:
                        run_avgs[g] = gm[g]["avg"]
                if run_avgs:
                    winners[p].append(max(run_avgs, key=run_avgs.get))

    if not acc:
        log.warning("No scorable patients for the signature pass.")
        return

    patients   = sorted(acc, key=ds._natkey)
    col_labels = [GROUP_LABEL[g] for g in GROUP_ORDER]

    def _matrices(metric):
        M = np.full((len(patients), len(GROUP_ORDER)), np.nan)
        S = np.full((len(patients), len(GROUP_ORDER)), np.nan)
        for ri, p in enumerate(patients):
            for ci, g in enumerate(GROUP_ORDER):
                m, s = _ms(acc[p][g][metric])
                M[ri, ci] = m
                S[ri, ci] = s
        return M, S

    auroc_M,    auroc_S    = _matrices("auroc")
    auprc_M,    auprc_S    = _matrices("auprc")
    avg_M,      avg_S      = _matrices("avg")
    contrast_M, _          = _matrices("contrast")
    prev_mean = {p: _ms(prev_acc[p])[0] for p in patients}

    def _dominant(ri):
        row   = {g: avg_M[ri, ci] for ci, g in enumerate(GROUP_ORDER)}
        valid = {g: m for g, m in row.items() if m == m}
        return max(valid, key=valid.get) if valid else None

    def _log_metric_table(title, M, S, tail=None):
        log.info("=" * 70)
        log.info(title)
        for ri, p in enumerate(patients):
            cells = "  ".join(f"{GROUP_LABEL[g]} {M[ri, ci]:.3f}+/-{S[ri, ci]:.3f}"
                              for ci, g in enumerate(GROUP_ORDER))
            log.info("  patient %-8s  %s%s", p, cells, tail(ri, p) if tail else "")

    def _avg_tail(ri, p):
        dom = _dominant(ri)
        if dom is None:
            return "  -> dominant n/a"
        wl    = winners.get(p, [])
        agree = sum(1 for w in wl if w == dom)
        return f"  -> dominant {GROUP_LABEL[dom]} (agree {agree}/{len(wl)})"

    _log_metric_table(
        f"PER-PATIENT GROUP AUROC (mean +/- std over {n_exp} experiments)  "
        f"[primary trust metric]",
        auroc_M, auroc_S)
    _log_metric_table(
        "PER-PATIENT GROUP AUPRC (mean +/- std)  [read against prevalence]",
        auprc_M, auprc_S, tail=lambda ri, p: f"  prev {prev_mean[p]:.3f}")
    _log_metric_table(
        "PER-PATIENT GROUP AVG (mean +/- std)  [dominant = argmax mean AVG]",
        avg_M, avg_S, tail=_avg_tail)

    def _col_ms(M, ci):
        col = M[:, ci]
        col = col[np.isfinite(col)]
        return (float(col.mean()), float(col.std())) if col.size else (float("nan"), float("nan"))

    log.info("=" * 70)
    log.info("COHORT GROUP SUMMARY (mean +/- std across %d patients)", len(patients))
    for ci, g in enumerate(GROUP_ORDER):
        au_m, au_s = _col_ms(auroc_M, ci)
        ap_m, ap_s = _col_ms(auprc_M, ci)
        av_m, av_s = _col_ms(avg_M, ci)
        log.info("  %-11s AUROC %.3f +/- %.3f  AUPRC %.3f +/- %.3f  AVG %.3f +/- %.3f",
                 GROUP_LABEL[g], au_m, au_s, ap_m, ap_s, av_m, av_s)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _save_heatmap(auroc_M, patients, col_labels,
                  title="Per-patient anomaly signature (group AUROC)",
                  cbar_label="AUROC", path=str(fig_dir / "signature_auroc.png"), fmt="{:.2f}")
    _save_heatmap(auprc_M, patients, col_labels,
                  title="Per-patient anomaly signature (group AUPRC)",
                  cbar_label="AUPRC", path=str(fig_dir / "signature_auprc.png"), fmt="{:.2f}")
    _save_heatmap(avg_M, patients, col_labels,
                  title="Per-patient anomaly signature (group AVG)",
                  cbar_label="AVG", path=str(fig_dir / "signature_avg.png"), fmt="{:.2f}")
    _save_heatmap(contrast_M, patients, col_labels,
                  title="Relapse-vs-normal contrast (normal-day std units)",
                  cbar_label="effect size",
                  path=str(fig_dir / "signature_contrast.png"), fmt="{:+.1f}")

    dom_count = defaultdict(int)
    for ri in range(len(patients)):
        dom = _dominant(ri)
        if dom is not None:
            dom_count[dom] += 1
    log.info("=" * 70)
    log.info("COHORT dominant-channel tally (by AVG): %s",
             "  ".join(f"{GROUP_LABEL[g]}={dom_count.get(g, 0)}" for g in GROUP_ORDER))


# --- Score-level ensemble across experiments (per patient-day) ---------------

def _scored_records(model, items, patient, device, agg,
                    smooth_days, smooth_mode, trim, gap_days):
    """Like _scored, but returns the smoothed record dicts, keeping
    calendar_date, so scores can be pooled across experiments by day."""
    recs = _patient_raw_records(model, items, patient, device, agg)
    recs = _gap_aware_smooth(recs, k=smooth_days, mode=smooth_mode,
                             trim=trim, gap_days=gap_days)
    return recs


def evaluate_experiment_records(features_dir, ckpt_path, device, agg,
                                smooth_days, smooth_mode, trim, gap_days, test_tod):
    """Per-patient smoothed (val_records, test_records) for one experiment. Used
    by the ensemble path and by evaluate_gated_fusion.py."""
    (bundle, fold, fold_idx, relapse_order, train_tod, eff_test_tod,
     wb, sb) = _load_bundle_and_fold(features_dir, ckpt_path, device, test_tod)

    log.info("=" * 70)
    log.info("Ensemble pass fold=%d order=%d  (%s)  train_tod=%s test_tod=%s  "
             "window=%d stride=%d  smooth=%d/%s  gap=%dd",
             fold_idx, relapse_order, ckpt_path.name, train_tod, eff_test_tod,
             wb, sb, smooth_days, smooth_mode, gap_days)

    out = {}
    for p in sorted(bundle["state_dicts"], key=ds._natkey):
        model = LSTMAutoencoder(**bundle["model_kwargs"]).to(device)
        model.load_state_dict(bundle["state_dicts"][p])
        vrecs = _scored_records(model, fold.val_items, p, device, agg,
                                smooth_days, smooth_mode, trim, gap_days)
        trecs = _scored_records(model, fold.test_items, p, device, agg,
                                smooth_days, smooth_mode, trim, gap_days)
        out[p] = (vrecs, trecs)
    return fold_idx, relapse_order, out


def run_ensemble(args, device, ckpt_dir):
    """Average each patient-day's smoothed score across all experiments in which
    that day was scored (test days pooled separately from val days), then compute
    ONE AUROC/AUPRC/AVG/F1 per patient on the pooled scores. This is a model
    ensemble: variance is reduced by averaging the per-patient models trained
    under different fold/order splits, which usually nudges AUPRC up for free.

    Coverage caveat: a calendar day only enters the pool from experiments where
    it landed in the test (resp. val) split, so different days are averaged over
    different numbers of models. Undated (NaT) days cannot be aligned across
    experiments and are kept as individual one-model points."""
    eff_test_tod = args.train_tod if args.test_tod is None else args.test_tod
    log.info("ENSEMBLE (score-level mean across experiments per patient-day) | "
             "agg=%s | smooth=%d/%s | gap=%dd%s",
             args.agg, args.smooth_days, args.smooth_mode, args.gap_days,
             "" if _is_clean(args.train_tod, eff_test_tod) else "  [CONFOUNDED]")

    val_pool  = defaultdict(dict)   # patient -> key -> {"label","date","scores"}
    test_pool = defaultdict(dict)
    nat_counter = [0]

    def _key(rec):
        d = rec["calendar_date"]
        if pd.isna(d):
            nat_counter[0] += 1
            return ("NaT", nat_counter[0])
        return ("D", pd.Timestamp(d).normalize())

    def _add(pool, p, rec):
        k = _key(rec)
        slot = pool[p].get(k)
        if slot is None:
            slot = {"label": rec["label"], "date": rec["calendar_date"], "scores": []}
            pool[p][k] = slot
        slot["scores"].append(rec["score"])

    n_exp = 0
    for fold_idx in args.folds:
        for relapse_order in args.orders:
            ckpt_path = ckpt_dir / f"fold_{fold_idx}_rel_{relapse_order}_{args.train_tod}.pt"
            if not ckpt_path.exists():
                log.warning("Missing checkpoint %s — skipping.", ckpt_path)
                continue
            n_exp += 1
            _, _, per_patient = evaluate_experiment_records(
                args.features_dir, ckpt_path, device, args.agg,
                args.smooth_days, args.smooth_mode, args.trim,
                args.gap_days, args.test_tod)
            for p, (vrecs, trecs) in per_patient.items():
                for rec in vrecs:
                    _add(val_pool, p, rec)
                for rec in trecs:
                    _add(test_pool, p, rec)

    patients = sorted(test_pool, key=ds._natkey)
    if not patients:
        log.warning("No scorable patients for the ensemble pass.")
        return

    def _pool_arrays(pool_p):
        items = sorted(pool_p.items(),
                       key=lambda kv: (kv[0][0] == "NaT", kv[1]["date"]))
        s   = np.array([float(np.mean(v["scores"])) for _, v in items], dtype=float)
        y   = np.array([int(v["label"]) for _, v in items], dtype=int)
        cov = np.array([len(v["scores"]) for _, v in items], dtype=int)
        return s, y, cov

    ens = {"auroc": [], "auprc": [], "avg": [], "f1": []}
    log.info("=" * 70)
    log.info("PER-PATIENT ENSEMBLE  [pooled over %d experiments, agg=%s, "
             "smooth=%d/%s, gap=%dd]", n_exp, args.agg,
             args.smooth_days, args.smooth_mode, args.gap_days)
    for p in patients:
        ts, ty, tcov = _pool_arrays(test_pool[p])
        if p in val_pool and val_pool[p]:
            vs, vy, _ = _pool_arrays(val_pool[p])
        else:
            vs, vy = ts, ty
        thr = _best_f1_threshold(vs, vy)
        m = _metrics(ts, ty, thr)
        for k in ("auroc", "auprc", "avg", "f1"):
            ens[k].append(m[k])
        log.info("    patient %-8s  AUROC %s  AUPRC %s  AVG %s  F1 %.3f  "
                 "prev %.3f  (%d days, mean coverage %.1f models)",
                 p,
                 f"{m['auroc']:.3f}" if m['auroc'] == m['auroc'] else "  nan",
                 f"{m['auprc']:.3f}" if m['auprc'] == m['auprc'] else "  nan",
                 f"{m['avg']:.3f}"   if m['avg']   == m['avg']   else "  nan",
                 m["f1"], m["prevalence"], m["n_days"],
                 float(tcov.mean()) if tcov.size else 0.0)

    log.info("=" * 70)
    log.info("ENSEMBLE COHORT SUMMARY (mean +/- std across %d patients)", len(patients))
    for k in ("auroc", "auprc", "avg", "f1"):
        mm, ss = _ms(ens[k])
        log.info("  %-6s : %.3f +/- %.3f", k.upper(), mm, ss)


# --- Diagnostics: rank baseline, raw versus smoothed AUROC, segmentation ----

def _auroc_or_nan(scores, labels) -> float:
    y = np.asarray(labels)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=float)))


def _sorted_by_date(records):
    return sorted(records,
                  key=lambda r: (pd.isna(r["calendar_date"]), r["calendar_date"]))


def _rank_baseline_auroc(records) -> float:
    """AUROC of pure chronological position vs label. If high, a continuous
    smoother can score well by reading temporal layout rather than physiology;
    an honest detector should beat raw, not merely match this."""
    if not records:
        return float("nan")
    recs = _sorted_by_date(records)
    y = np.array([r["label"] for r in recs], dtype=int)
    pos = np.arange(len(recs), dtype=float)
    return _auroc_or_nan(pos, y)


def _segment_lengths(records, gap_days):
    """Lengths of the contiguous segments produced by the same sort+gap rule the
    smoother uses. Days in segments shorter than K get lag without real
    smoothing benefit."""
    if not records:
        return []
    recs = _sorted_by_date(records)
    lengths, cur = [], 1
    for prev, nxt in zip(recs[:-1], recs[1:]):
        pd_, cd_ = prev["calendar_date"], nxt["calendar_date"]
        gap = pd.isna(pd_) or pd.isna(cd_) or (cd_ - pd_).days > gap_days
        if gap:
            lengths.append(cur); cur = 1
        else:
            cur += 1
    lengths.append(cur)
    return lengths


def run_diagnostics(args, device, ckpt_dir):
    """Settle 'why is smoothing worse' in one pass. Per patient (averaged over
    experiments) it reports:
      rankAU : AUROC of chronological position alone (temporal-layout shortcut)
      rawAU  : AUROC at K=1 (no smoothing)
      smAU   : AUROC at the requested K
      d-sm   : smAU - rawAU  (smoothing's actual effect; negative = it hurts)
      segs   : number of gap-split segments
      segMed : median segment length
      %<K    : fraction of test days in segments shorter than K (lag, no benefit)
      NaT%   : fraction of days with no calendar_date (smoothing silently skips
               them; high here means the filter is barely active)
    Read: rawAU >= smAU means smoothing is not helping; smAU ~ rankAU means the
    score tracks position; high %<K or NaT% means the filter is mostly warm-up on
    fragmented data."""
    eff_test_tod = args.train_tod if args.test_tod is None else args.test_tod
    K = args.smooth_days
    log.info("DIAGNOSTICS | agg=%s | K=%d mode=%s | gap=%dd | regime %s/%s",
             args.agg, K, args.smooth_mode, args.gap_days,
             args.train_tod, eff_test_tod)

    rank_a = defaultdict(list); raw_a = defaultdict(list); sm_a = defaultdict(list)
    segc_a = defaultdict(list); segm_a = defaultdict(list); short_a = defaultdict(list)
    nat_n  = defaultdict(int);  day_n = defaultdict(int)

    for fold_idx in args.folds:
        for relapse_order in args.orders:
            ckpt_path = ckpt_dir / f"fold_{fold_idx}_rel_{relapse_order}_{args.train_tod}.pt"
            if not ckpt_path.exists():
                log.warning("Missing checkpoint %s — skipping.", ckpt_path)
                continue
            (bundle, fold, fidx, rorder, train_tod, eff_tt,
             wb, sb) = _load_bundle_and_fold(args.features_dir, ckpt_path,
                                             device, args.test_tod)
            for p in sorted(bundle["state_dicts"], key=ds._natkey):
                model = LSTMAutoencoder(**bundle["model_kwargs"]).to(device)
                model.load_state_dict(bundle["state_dicts"][p])
                raw = _patient_raw_records(model, fold.test_items, p, device, args.agg)
                if not raw:
                    continue
                day_n[p] += len(raw)
                nat_n[p] += sum(1 for r in raw if pd.isna(r["calendar_date"]))
                rank_a[p].append(_rank_baseline_auroc(raw))
                rs, ry = _recs_to_arrays(
                    _gap_aware_smooth(raw, 1, args.smooth_mode, args.trim, args.gap_days))
                raw_a[p].append(_auroc_or_nan(rs, ry))
                ss, sy = _recs_to_arrays(
                    _gap_aware_smooth(raw, K, args.smooth_mode, args.trim, args.gap_days))
                sm_a[p].append(_auroc_or_nan(ss, sy))
                lens = _segment_lengths(raw, args.gap_days)
                segc_a[p].append(len(lens))
                segm_a[p].append(float(np.median(lens)) if lens else float("nan"))
                tot = max(sum(lens), 1)
                short_a[p].append(sum(L for L in lens if L < K) / tot)

    patients = sorted(day_n, key=ds._natkey)
    if not patients:
        log.warning("No scorable patients for diagnostics.")
        return

    log.info("=" * 70)
    log.info("PER-PATIENT DIAGNOSTICS (mean over experiments)  [K=%d]", K)
    log.info("  %-8s  %6s  %6s  %6s  %7s  %5s  %6s  %5s  %5s",
             "patient", "rankAU", "rawAU", "smAU", "d-sm", "segs",
             "segMed", "%<K", "NaT%")
    cohort = defaultdict(list)
    for p in patients:
        rk, _ = _ms(rank_a[p]); rw, _ = _ms(raw_a[p]); sm, _ = _ms(sm_a[p])
        sc, _ = _ms(segc_a[p]); sl, _ = _ms(segm_a[p]); sh, _ = _ms(short_a[p])
        natpct = 100.0 * nat_n[p] / max(day_n[p], 1)
        dsm = (sm - rw) if (sm == sm and rw == rw) else float("nan")
        log.info("  %-8s  %6.3f  %6.3f  %6.3f  %+7.3f  %5.1f  %6.1f  %4.0f%%  %4.0f%%",
                 p, rk, rw, sm, dsm, sc, sl, 100.0 * sh, natpct)
        cohort["rank"].append(rk); cohort["raw"].append(rw)
        cohort["sm"].append(sm);   cohort["dsm"].append(dsm)

    log.info("=" * 70)
    rk_m, _ = _ms(cohort["rank"]); rw_m, _ = _ms(cohort["raw"])
    sm_m, _ = _ms(cohort["sm"]);   d_m, _ = _ms(cohort["dsm"])
    log.info("COHORT MEAN  rankAU %.3f  rawAU %.3f  smAU %.3f  d-sm %+.3f",
             rk_m, rw_m, sm_m, d_m)
    if d_m == d_m and d_m <= 0:
        log.info("READ: smoothing is NOT helping on average (d-sm <= 0). Prefer "
                 "K=1 for the headline, or address gap/feature fragmentation.")
    if rk_m == rk_m and sm_m == sm_m and abs(sm_m - rk_m) < 0.03:
        log.info("READ: smoothed AUROC ~ chronological-position AUROC. Scores may "
                 "be tracking temporal layout rather than physiology.")


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate per-patient LSTM autoencoders.")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--folds",  type=int, nargs="*", default=list(range(ds.N_REMISSION_FOLDS)))
    ap.add_argument("--orders", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--agg", choices=["mean", "median", "p90", "p95", "max"],
                    default="mean",
                    help="Within-day reduction over window errors. Tail stats "
                         "(p90/p95/max) sharpen partial-day anomalies and tend to "
                         "help AUPRC; mean assumes the whole day is abnormal.")
    ap.add_argument("--smooth-days", type=int, default=1,
                    help="Causal filter width in days. 1 = no smoothing.")
    ap.add_argument("--smooth-mode", choices=list(SMOOTH_MODES), default="mean",
                    help="Filter type applied within each continuous segment.")
    ap.add_argument("--trim", type=float, default=0.2,
                    help="Fraction trimmed from each end for --smooth-mode trimmed.")
    ap.add_argument("--gap-days", type=int, default=7,
                    help="Calendar gap (days) that triggers a segment boundary "
                         "and resets the filter. Default 7.")
    ap.add_argument("--train-tod", choices=list(ds.TOD_CHOICES), default="both",
                    help="Which checkpoint set to evaluate (the training regime "
                         "baked into the filenames: fold_*_rel_*_<train-tod>.pt).")
    ap.add_argument("--test-tod", choices=list(ds.TOD_CHOICES), default=None,
                    help="Evaluation regime. Default = matched (same as --train-tod). "
                         "Only meaningful to differ when --train-tod both.")
    ap.add_argument("--signature", action="store_true",
                    help="Run the per-patient anomaly-signature pass (per-group "
                         "AUROC + heatmaps). Read-only: headline metrics unchanged.")
    ap.add_argument("--fig-dir", default="./figures",
                    help="Where the --signature heatmaps are written.")
    ap.add_argument("--ensemble", action="store_true",
                    help="Score-level ensemble: average each patient-day's "
                         "smoothed score across all experiments, then report one "
                         "AUROC/AUPRC/AVG/F1 per patient. Reduces variance.")
    ap.add_argument("--diagnose", action="store_true",
                    help="Diagnostic pass: per-patient rank-baseline AUROC, "
                         "raw-vs-smoothed AUROC, segment lengths and NaT counts, "
                         "to explain whether smoothing helps or hurts.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.smooth_days < 1:
        ap.error("--smooth-days must be >= 1")
    if not (0.0 <= args.trim < 0.5):
        ap.error("--trim must be in [0, 0.5)")

    device   = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)

    if args.signature:
        run_signature(args, device, ckpt_dir)
        return

    if args.diagnose:
        run_diagnostics(args, device, ckpt_dir)
        return

    if args.ensemble:
        run_ensemble(args, device, ckpt_dir)
        return

    eff_test_tod = args.train_tod if args.test_tod is None else args.test_tod
    log.info("regime: train_tod=%s test_tod=%s%s | agg=%s | smooth=%d/%s%s | "
             "gap=%dd | window geometry: per-checkpoint",
             args.train_tod, eff_test_tod,
             "" if _is_clean(args.train_tod, eff_test_tod) else "  [CONFOUNDED]",
             args.agg, args.smooth_days, args.smooth_mode,
             f" (trim {args.trim})" if args.smooth_mode == "trimmed" else "",
             args.gap_days)

    ae_means = {"auroc": [], "auprc": [], "avg": [], "f1": []}
    per_patient_runs = defaultdict(
        lambda: {"auroc": [], "auprc": [], "avg": [], "f1": [],
                 "prevalence": [], "n_days": []})

    for fold_idx in args.folds:
        for relapse_order in args.orders:
            ckpt_path = ckpt_dir / f"fold_{fold_idx}_rel_{relapse_order}_{args.train_tod}.pt"
            if not ckpt_path.exists():
                log.warning("Missing checkpoint %s — skipping.", ckpt_path)
                continue

            fidx, rorder, per_patient = evaluate_experiment(
                args.features_dir, ckpt_path, device, args.agg,
                args.smooth_days, args.smooth_mode, args.trim,
                args.gap_days, args.test_tod)

            log.info("  AE per-patient (fold %d, order %d):", fidx, rorder)
            for p in sorted(per_patient, key=ds._natkey):
                r = per_patient[p]
                log.info("    patient %-8s  AUROC %s  AUPRC %s  AVG %s  F1 %.3f  prev %.3f  (%d days)",
                         p,
                         f"{r['auroc']:.3f}" if r['auroc'] == r['auroc'] else "  nan",
                         f"{r['auprc']:.3f}" if r['auprc'] == r['auprc'] else "  nan",
                         f"{r['avg']:.3f}"   if r['avg']   == r['avg']   else "  nan",
                         r["f1"], r["prevalence"], r["n_days"])
                per_patient_runs[p]["auroc"].append(r["auroc"])
                per_patient_runs[p]["auprc"].append(r["auprc"])
                per_patient_runs[p]["avg"].append(r["avg"])
                per_patient_runs[p]["f1"].append(r["f1"])
                per_patient_runs[p]["prevalence"].append(r["prevalence"])
                per_patient_runs[p]["n_days"].append(r["n_days"])
            a_au, a_ap, a_avg, a_f1 = _mean_across_patients(per_patient)
            log.info("  AE     MEAN (fold %d, order %d)  AUROC %.3f  AUPRC %.3f  AVG %.3f  F1 %.3f",
                     fidx, rorder, a_au, a_ap, a_avg, a_f1)
            ae_means["auroc"].append(a_au)
            ae_means["auprc"].append(a_ap)
            ae_means["avg"].append(a_avg)
            ae_means["f1"].append(a_f1)

    log.info("=" * 70)
    log.info("PER-PATIENT SUMMARY across runs (mean +/- std over the experiments "
             "each patient appears in)  [smooth=%d mode=%s gap=%dd]",
             args.smooth_days, args.smooth_mode, args.gap_days)
    for p in sorted(per_patient_runs, key=ds._natkey):
        rec = per_patient_runs[p]
        au_m,  au_s  = _ms(rec["auroc"])
        ap_m,  ap_s  = _ms(rec["auprc"])
        avg_m, avg_s = _ms(rec["avg"])
        f1_m,  f1_s  = _ms(rec["f1"])
        prev_m, _    = _ms(rec["prevalence"])
        n_runs = len(rec["auroc"])
        n_au   = sum(1 for x in rec["auroc"] if x == x)
        log.info("  patient %-8s  AUROC %.3f +/- %.3f  AUPRC %.3f +/- %.3f  "
                 "AVG %.3f +/- %.3f  F1 %.3f +/- %.3f  prev %.3f  (%d runs, %d scorable)",
                 p, au_m, au_s, ap_m, ap_s, avg_m, avg_s, f1_m, f1_s, prev_m, n_runs, n_au)

    log.info("=" * 70)
    log.info("SUMMARY across %d experiments (each experiment = mean across patients)  "
             "[agg=%s, smooth=%d, mode=%s, gap=%dd]",
             len(ae_means["auroc"]), args.agg,
             args.smooth_days, args.smooth_mode, args.gap_days)
    for k in ("auroc", "auprc", "avg", "f1"):
        m, s = _ms(ae_means[k])
        log.info("  %-6s : %.3f +/- %.3f", k.upper(), m, s)


if __name__ == "__main__":
    main()
