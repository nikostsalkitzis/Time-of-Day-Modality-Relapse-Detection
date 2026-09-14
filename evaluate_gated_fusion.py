"""
Learned soft combination of frozen, personalised, unsupervised LSTM-AE experts.

Each expert e is one checkpoint set produced by train.py for one modality or one
time-of-day regime. Experts stay frozen: no retraining, no fine-tuning, and no
gradient ever reaches an expert. Every expert produces its reconstruction-based
day score s_e(t) through evaluate_unimodal.evaluate_experiment_records, which is
z-scored on that expert's own validation-normal days.

The day set is the union over experts, so a day survives if at least one expert
scored it. A gating network g consumes, for day t of one patient,
        [ s_1..s_E (masked to 0 where absent),
          m_1..m_E (binary availability),
          optional coverage channel,
          optional patient embedding ]
and emits E logits. Unavailable experts are masked with -inf before the softmax,
so pi_e(t) = 0 exactly for an absent expert and graceful degradation is
structural rather than learned. The fused score is the convex combination
        shat(t) = sum_e pi_e(t) * s_e(t)
which then passes through the causal smoother.

The gate head is zero-initialised, so at epoch 0 pi is uniform over the available
experts and shat is exactly the masked mean-fusion baseline. Any departure from
that baseline has to be earned on validation days.

Causal discipline
  * The gate is fitted on validation days only, pooled across patients, one gate
    per (fold, order) experiment. No test label is read at any point.
  * Epoch selection uses a held-out slice of the validation days
    (--gate-val-frac), chronologically the last portion per patient by default.
  * The gate reads only day-t quantities and the smoother is a trailing window,
    so neither can see the future of the series.
  * dataset.build_fold_data does not guarantee that validation precedes test in
    wall-clock time, since the relapse halves swap with relapse_order. The
    guarantee here is therefore that no test label is used.

Scope of the learned combiner (--scope)
  cohort   one combiner shared by all patients, fitted on the pooled validation
           days.
  patient  a separate combiner per patient, fitted on that patient's validation
           days only. Applies to --fusion weighted and --fusion gate. The other
           rules are unaffected because they are already per-patient (rank) or
           parameter-free (mean, max).

  Per-patient fitting is statistically fragile at this cohort size: a 5-expert
  simplex has 1001 grid points and a hidden-16 gate a few hundred parameters,
  against roughly a hundred validation days per patient. Two hedges:
    --shrink L          weighted only. w = (1-L)*w_patient + L*w_cohort, so L=0
                        is fully personalised and L=1 collapses to cohort scope.
    --gate-warm-start   gate only, default "cohort". Each patient's gate starts
                        from the cohort gate rather than from the mean-fusion
                        baseline, so a patient with little signal stays near the
                        cohort solution instead of wandering.
  A patient whose validation days are single-class cannot be fitted and falls
  back to the cohort combiner. The count is logged per experiment.

  --patient-embed is forced to 0 under --scope patient: a gate that only ever
  sees one patient has nothing to embed, and the warm start requires the cohort
  and per-patient gates to share an architecture.

Baselines, evaluated in this file on the same union day set
    --fusion gate         the masked-softmax gate
    --fusion rank         per-patient expert ranking on validation, then the
                          first available expert at test
    --fusion mean         masked mean over available experts
    --fusion max          masked max over available experts
    --fusion weighted     validation-tuned simplex weights, renormalised over the
                          experts available on each day
    --fusion single:NAME  one expert only, on the days it is available

Smoother placement
    --smooth-stage post  (default) experts are scored raw and the fused series is
                         smoothed once, giving the gate -> smoother ordering.
    --smooth-stage pre   experts are scored with smoothing already applied by
                         evaluate_unimodal, the gate runs on smoothed scores and
                         nothing is smoothed afterwards.
    --verify-smoother    scores one expert both ways and reports the maximum
                         absolute deviation between evaluate_unimodal's smoother
                         and the local reimplementation used in the post stage.
                         Run this once per configuration before trusting
                         post-stage numbers.

Run:
    python evaluate_gated_fusion.py \
        --features-dir features_reorganized/ \
        --modality motion=./checkpoints_train_night_hope_motion \
        --modality sleep=./checkpoints_train_both_hope_sleep:night \
        --modality hrv=./checkpoints_train_day_hope_hrv_right \
        --fusion gate --agg mean \
        --smooth-stage post --smooth-days 14 --smooth-mode mean --gap-days 7
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

import dataset as ds
import evaluate_unimodal as ev

log = logging.getLogger("fusion")

_COVERAGE_WARNED = [False]
_LABEL_WARNED = [False]

SMOOTH_MODES = tuple(ev.SMOOTH_MODES)


# --- Record plumbing --------------------------------------------------------

def _parse_modalities(pairs):
    """Parse NAME=CKPTDIR[:TEST_TOD] -> {name: (Path, test_tod_override)}."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--modality must be NAME=CKPTDIR[:TEST_TOD], got '{p}'")
        name, rest = p.split("=", 1)
        name = name.strip()
        if ":" in rest:
            ckpt_dir, test_tod_override = rest.rsplit(":", 1)
            test_tod_override = test_tod_override.strip()
            if test_tod_override not in ds.TOD_CHOICES:
                raise SystemExit(
                    f"TEST_TOD must be one of {ds.TOD_CHOICES}, got '{test_tod_override}'")
        else:
            ckpt_dir, test_tod_override = rest, None
        out[name] = (Path(ckpt_dir.strip()), test_tod_override)
    if len(out) < 2:
        log.warning("Only %d modality given; gating is trivial.", len(out))
    return out


def _dkey(rec):
    d = pd.to_datetime(rec.get("calendar_date"), errors="coerce")
    return None if pd.isna(d) else pd.Timestamp(d).normalize()


def _rec_coverage(rec):
    if "coverage" in rec:
        return int(rec["coverage"])
    if not _COVERAGE_WARNED[0]:
        log.warning("Records carry no 'coverage' field — falling back to "
                    "coverage=1 for all days.")
        _COVERAGE_WARNED[0] = True
    return 1


def _recs_to_datemap(recs):
    m = {}
    for r in recs:
        k = _dkey(r)
        if k is not None:
            m[k] = (float(r["score"]), int(r["label"]), _rec_coverage(r))
    return m


def _normal_stats(datemap):
    """Mean/std of that expert's validation-NORMAL scores for this patient."""
    vals = [v[0] for v in datemap.values() if v[1] == 0]
    if not vals:
        vals = [v[0] for v in datemap.values()]
    mu = float(np.mean(vals)) if vals else 0.0
    sd = float(np.std(vals)) if vals else 1.0
    return mu, (sd if sd > 1e-9 else 1.0)


def _cov_weight(cov, transform, cap):
    c = max(float(cov), 0.0)
    if transform == "sqrt":
        w = float(np.sqrt(c))
    elif transform == "log":
        w = float(np.log1p(c))
    else:
        w = c
    if cap is not None:
        w = min(w, float(cap))
    return w


# --- Expert scoring ----------------------------------------------------------

def _score_one_modality(features_dir, ckpt_path, device, agg,
                        smooth_days, smooth_mode, trim, gap_days,
                        test_tod_override=None):
    """
    Score one frozen expert from one checkpoint bundle. Returns
    {patient: (val_datemap, test_datemap)}, or None if the checkpoint is absent.
    Set smooth_days=1 for raw daily scores.
    """
    if not ckpt_path.exists():
        return None
    bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ds.FEATURE_COLS = list(bundle.get("feature_cols", ds.FEATURE_COLS))
    train_tod_ckpt = bundle.get("train_tod", "both")
    eff_test_tod = test_tod_override if test_tod_override is not None else train_tod_ckpt
    _, _, out = ev.evaluate_experiment_records(
        features_dir, ckpt_path, device, agg,
        smooth_days, smooth_mode, trim, gap_days, eff_test_tod)
    return {p: (_recs_to_datemap(v), _recs_to_datemap(t)) for p, (v, t) in out.items()}


def _resolve_ckpt(ckpt_dir, fold_idx, order, fallback_train_tod):
    candidates = sorted(ckpt_dir.glob(f"fold_{fold_idx}_rel_{order}_*.pt"))
    if not candidates:
        return ckpt_dir / f"fold_{fold_idx}_rel_{order}_{fallback_train_tod}.pt"
    return candidates[0]


def _score_modalities_one_experiment(features_dir, modalities, fold_idx, order,
                                     fallback_train_tod, device, agg,
                                     smooth_days, smooth_mode, trim, gap_days):
    """Returns {name: {patient: (val_datemap, test_datemap)}} or None."""
    per_mod = {}
    for name, (ckpt_dir, test_tod_override) in modalities.items():
        ckpt_path = _resolve_ckpt(ckpt_dir, fold_idx, order, fallback_train_tod)
        scores = _score_one_modality(
            features_dir, ckpt_path, device, agg,
            smooth_days, smooth_mode, trim, gap_days,
            test_tod_override=test_tod_override)
        if scores is None:
            log.warning("  [%s] missing %s — skipping whole experiment.", name, ckpt_path)
            return None
        per_mod[name] = scores
    return per_mod


# --- Causal smoother, used only in --smooth-stage post ----------------------

def _reduce_window(w, mode, trim):
    if len(w) == 0:
        return float("nan")
    if mode == "mean":
        return float(np.mean(w))
    if mode == "median":
        return float(np.median(w))
    if mode == "max":
        return float(np.max(w))
    if mode == "min":
        return float(np.min(w))
    if mode in ("trim", "trimmed", "trimmed_mean"):
        n = len(w)
        c = int(math.floor(n * float(trim)))
        if 2 * c >= n:
            return float(np.median(w))
        return float(np.mean(np.sort(w)[c:n - c]))
    raise SystemExit(f"--smooth-stage post cannot reproduce smooth mode '{mode}'. "
                     f"Use --smooth-stage pre, or extend _reduce_window.")


def _smooth_causal(dates, vals, smooth_days, mode, trim, gap_days, window="records"):
    """
    Trailing causal smoothing of a date-sorted series, segmented on gaps.

    A new segment starts whenever the day gap to the previous record exceeds
    gap_days, so no window straddles a monitoring seam. Day t only ever sees days
    up to and including t.

    window="records" : the last `smooth_days` available records in the segment
    window="days"    : all records inside [t - smooth_days + 1, t] in the segment
    """
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0 or smooth_days is None or smooth_days <= 1 or mode == "none":
        return vals
    ordn = np.array([pd.Timestamp(d).toordinal() for d in dates], dtype=np.int64)

    if mode == "ewma":
        alpha = 2.0 / (float(smooth_days) + 1.0)
        out = np.empty(len(vals))
        acc = None
        for i in range(len(vals)):
            if i > 0 and (ordn[i] - ordn[i - 1]) > gap_days:
                acc = None
            acc = vals[i] if acc is None else (alpha * vals[i] + (1.0 - alpha) * acc)
            out[i] = acc
        return out

    out = np.empty(len(vals))
    seg_start = 0
    for i in range(len(vals)):
        if i > 0 and (ordn[i] - ordn[i - 1]) > gap_days:
            seg_start = i
        if window == "days":
            lo = seg_start + int(np.searchsorted(
                ordn[seg_start:i + 1], ordn[i] - smooth_days + 1, side="left"))
        else:
            lo = max(seg_start, i - smooth_days + 1)
        out[i] = _reduce_window(vals[lo:i + 1], mode, trim)
    return out


# --- Per-patient day matrices with availability mask -------------------------

def _day_axis(maps, names, policy):
    sets = [set(maps[n]) for n in names if n in maps]
    if not sets:
        return []
    if policy == "intersect":
        return sorted(set.intersection(*sets))
    u = set()
    for s in sets:
        u |= s
    return sorted(u)


def _build_matrices(maps, names, stats, dates, transform, cap):
    """
    S : (E, T) z-scored expert scores, 0 where the expert is absent
    M : (E, T) binary availability
    C : (E, T) coverage weight, 0 where absent
    y : (T,)   day label
    """
    E, T = len(names), len(dates)
    S = np.zeros((E, T), dtype=float)
    M = np.zeros((E, T), dtype=float)
    C = np.zeros((E, T), dtype=float)
    y = np.full(T, -1, dtype=int)
    for e, n in enumerate(names):
        mu, sd = stats[n]
        m = maps[n]
        for i, d in enumerate(dates):
            if d in m:
                sc, lb, cov = m[d]
                S[e, i] = (sc - mu) / sd
                M[e, i] = 1.0
                C[e, i] = _cov_weight(cov, transform, cap)
                if y[i] < 0:
                    y[i] = int(lb)
                elif y[i] != int(lb) and not _LABEL_WARNED[0]:
                    log.warning("Label disagreement across experts on %s "
                                "(keeping first seen).", d)
                    _LABEL_WARNED[0] = True
    return S, M, C, y


# --- Non-learned fusion rules, all mask-aware -------------------------------

def _fuse_masked(S, M, C, alpha=None, use_cov=False, mode="mean"):
    if S.shape[1] == 0:
        return np.empty(0)
    if mode == "max":
        return np.where(M > 0.5, S, -np.inf).max(axis=0)
    W = C if use_cov else np.ones_like(M)
    eff = M * W
    if alpha is not None:
        eff = eff * np.asarray(alpha, dtype=float).reshape(-1, 1)
    denom = eff.sum(axis=0)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return (eff * S).sum(axis=0) / denom


def _simplex_grid(k, step):
    n = int(round(1.0 / step))
    out = []

    def rec(prefix, rem, left):
        if left == 1:
            out.append(tuple((prefix + [rem])[i] / n for i in range(k)))
            return
        for i in range(rem + 1):
            rec(prefix + [i], rem - i, left - 1)

    rec([], n, k)
    return out


def _avg_metric(y, s, objective="avg"):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 2:
        return float("nan")
    y, s = y[ok], s[ok]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    au = roc_auc_score(y, s)
    if objective == "auroc":
        return float(au)
    return float(0.5 * (au + average_precision_score(y, s)))


def _pick_weights_on_val(val_mats, grid, use_cov, objective="avg"):
    """
    Grid-search simplex weights on the validation days of whatever is passed in:
    every patient's matrices under cohort scope, one patient's under patient
    scope.

    Returns (weights, ok). ok is False when the validation days were empty or
    single-class, so the caller can fall back instead of trusting the uniform
    vector returned in that case.
    """
    Ss, Ms, Cs, ys = [], [], [], []
    for m in val_mats:
        S, M, C, y = m[0], m[1], m[2], m[3]
        if S.shape[1]:
            Ss.append(S); Ms.append(M); Cs.append(C); ys.append(y)
    if not Ss:
        E = 1 if not grid else len(grid[0])
        return tuple(1.0 / E for _ in range(E)), False
    S = np.hstack(Ss); M = np.hstack(Ms); C = np.hstack(Cs); y = np.concatenate(ys)
    E = S.shape[0]
    if y.sum() == 0 or y.sum() == len(y):
        return tuple(1.0 / E for _ in range(E)), False
    best_w, best = None, -np.inf
    for w in grid:
        v = _avg_metric(y, _fuse_masked(S, M, C, alpha=w, use_cov=use_cov), objective)
        if v == v and v > best:
            best, best_w = v, w
    if best_w is None:
        return tuple(1.0 / E for _ in range(E)), False
    return best_w, True


def _rank_experts_on_val(S, M, y, objective="avg"):
    """
    Per-patient expert ranking on validation days. Each expert is scored only on
    the days it is available, so a rarely-available expert is not penalised for
    its absences. Returns (order, per_expert_score).
    """
    E = S.shape[0]
    scores = np.full(E, -np.inf)
    for e in range(E):
        idx = M[e] > 0.5
        if idx.sum() < 2:
            continue
        v = _avg_metric(y[idx], S[e, idx], objective)
        scores[e] = v if v == v else -np.inf
    order = list(np.argsort(-scores, kind="stable"))
    return order, scores


def _fuse_rank(S, M, order):
    """First available expert in the validation ranking, per day."""
    T = S.shape[1]
    out = np.full(T, np.nan)
    for i in range(T):
        for e in order:
            if M[e, i] > 0.5:
                out[i] = S[e, i]
                break
    return out


# --- The gate ----------------------------------------------------------------

class ScoreGate(nn.Module):
    """
    Masked-softmax mixture over frozen expert scores.

    Input per day: expert z-scores (zeroed where absent), the binary availability
    mask, optionally a standardised coverage channel and a patient embedding.
    Output: pi on the simplex restricted to the available experts, and the convex
    combination shat = sum_e pi_e * s_e. The head is zero-initialised, so the
    untrained gate is exactly masked mean fusion.
    """

    def __init__(self, n_experts, hidden=16, use_cov=False,
                 n_patients=0, patient_dim=0):
        super().__init__()
        self.E = n_experts
        self.use_cov = use_cov
        self.patient_dim = patient_dim if n_patients > 0 else 0
        in_dim = 2 * n_experts + (n_experts if use_cov else 0) + self.patient_dim
        if self.patient_dim > 0:
            self.emb = nn.Embedding(n_patients, self.patient_dim)
            # Small random init rather than zeros. The zeroed head below is what
            # makes the untrained gate equal masked mean fusion. Zeroing the
            # embedding too would leave it with zero gradient forever when
            # gate_hidden == 0, and dead for the first step otherwise.
            nn.init.normal_(self.emb.weight, std=0.1)
        else:
            self.emb = None
        if hidden and hidden > 0:
            self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh())
            self.head = nn.Linear(hidden, n_experts)
        else:
            self.body = nn.Identity()
            self.head = nn.Linear(in_dim, n_experts)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, s, m, c=None, pid=None):
        feats = [s * m, m]
        if self.use_cov:
            feats.append(c if c is not None else torch.zeros_like(m))
        if self.emb is not None:
            feats.append(self.emb(pid))
        x = torch.cat(feats, dim=-1)
        logits = self.head(self.body(x))
        logits = logits.masked_fill(m < 0.5, float("-inf"))
        dead = (m.sum(dim=-1, keepdim=True) < 0.5).expand_as(logits)
        logits = torch.where(dead, torch.zeros_like(logits), logits)
        pi = torch.softmax(logits, dim=-1)
        pi = pi * m
        shat = (pi * s * m).sum(dim=-1)
        return shat, pi


def _rows_from_mats(mats, pid_of_patient):
    """Stack per-patient (S, M, C, y) into flat row tensors plus index arrays."""
    S, M, C, Y, P, G = [], [], [], [], [], []
    for gi, (s, m, c, y, _d, patient) in enumerate(mats):
        if s.shape[1] == 0:
            continue
        S.append(s.T); M.append(m.T); C.append(c.T); Y.append(y)
        P.append(np.full(len(y), pid_of_patient.get(patient, 0), dtype=np.int64))
        G.append(np.full(len(y), gi, dtype=np.int64))
    if not S:
        return None
    return (np.vstack(S), np.vstack(M), np.vstack(C),
            np.concatenate(Y), np.concatenate(P), np.concatenate(G))


def _chrono_split_rows(mats, frac, mode, rng):
    """
    Per-patient split of the validation days into (fit, select) index sets.
    mode='chrono' holds out the last `frac` of each patient's validation days.
    """
    fit_flags = []
    for (s, _m, _c, y, dates, _p) in mats:
        T = s.shape[1]
        if T == 0:
            continue
        keep = np.ones(T, dtype=bool)
        n_sel = int(round(T * frac))
        if n_sel > 0 and n_sel < T:
            if mode == "random":
                idx = rng.sample(range(T), n_sel)
            else:
                order = np.argsort([pd.Timestamp(d).toordinal() for d in dates])
                idx = order[-n_sel:]
            keep[list(idx)] = False
        fit_flags.append(keep)
    return np.concatenate(fit_flags) if fit_flags else np.zeros(0, dtype=bool)


def _pairwise_loss(shat, y, groups, rng, max_pairs):
    """
    Smooth AUROC surrogate. Pairs are drawn within a patient, so cross-patient
    score offsets cannot be exploited.
    """
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    pos_idx, neg_idx = [], []
    uniq = np.unique(groups)
    per_group = max(1, max_pairs // max(len(uniq), 1))
    for g in uniq:
        gi = np.where(groups == g)[0]
        p = gi[y_np[gi] == 1]
        n = gi[y_np[gi] == 0]
        if len(p) == 0 or len(n) == 0:
            continue
        k = min(per_group, len(p) * len(n))
        pos_idx.extend(rng.choices(list(p), k=k))
        neg_idx.extend(rng.choices(list(n), k=k))
    if not pos_idx:
        return None
    pi_t = torch.as_tensor(pos_idx, dtype=torch.long, device=shat.device)
    ni_t = torch.as_tensor(neg_idx, dtype=torch.long, device=shat.device)
    return torch.nn.functional.softplus(-(shat[pi_t] - shat[ni_t])).mean()


def _entropy_penalty(pi, m):
    """Mean KL(pi || uniform over available). Zero at the mean-fusion baseline."""
    n_av = m.sum(dim=-1).clamp(min=1.0)
    ent = -(pi.clamp_min(1e-12).log() * pi).sum(dim=-1)
    return (torch.log(n_av) - ent).mean()


def train_gate(val_mats, names, args, device, rng, init_state=None):
    """
    Fit the gate on validation days only. Returns (gate, cov_mu, cov_sd,
    pid_of_patient, report).

    val_mats holds every patient under cohort scope, or exactly one patient under
    per-patient scope. init_state warm-starts from an already-fitted gate, which
    is how each patient's gate is initialised from the cohort gate. That requires
    matching architectures, which is why --patient-embed is forced to 0 under
    per-patient scope.
    """
    patients = sorted({m[5] for m in val_mats}, key=ds._natkey)
    pid_of_patient = {p: i for i, p in enumerate(patients)}
    rows = _rows_from_mats(val_mats, pid_of_patient)
    if rows is None:
        return None, 0.0, 1.0, pid_of_patient, {"status": "no validation days"}
    S, M, C, Y, P, G = rows
    if Y.sum() == 0 or Y.sum() == len(Y):
        return None, 0.0, 1.0, pid_of_patient, {"status": "validation has one class"}

    # Standardised on the gate-fitting rows only.
    cov_mu, cov_sd = 0.0, 1.0
    if args.gate_use_coverage:
        obs = C[M > 0.5]
        if obs.size:
            cov_mu = float(obs.mean())
            cov_sd = float(obs.std()) or 1.0
    Cn = np.where(M > 0.5, (C - cov_mu) / cov_sd, 0.0)

    fit_mask = _chrono_split_rows(val_mats, args.gate_val_frac,
                                  args.gate_split, rng)
    if fit_mask.shape[0] != len(Y):
        fit_mask = np.ones(len(Y), dtype=bool)
    sel_mask = ~fit_mask
    if sel_mask.sum() == 0 or Y[sel_mask].sum() == 0 or Y[sel_mask].sum() == sel_mask.sum():
        fit_mask = np.ones(len(Y), dtype=bool)
        sel_mask = np.zeros(len(Y), dtype=bool)

    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    S_t, M_t, C_t = t(S), t(M), t(Cn)
    P_t = torch.as_tensor(P, dtype=torch.long, device=device)
    Y_t = t(Y)

    gate = ScoreGate(len(names), hidden=args.gate_hidden,
                     use_cov=args.gate_use_coverage,
                     n_patients=len(patients) if args.patient_embed > 0 else 0,
                     patient_dim=args.patient_embed).to(device)
    if init_state is not None:
        try:
            gate.load_state_dict(init_state)
        except (RuntimeError, KeyError) as exc:
            log.warning("  warm start ignored (%s)", exc)
    params = list(gate.parameters())
    calib = None
    if args.gate_loss == "bce":
        calib = nn.Parameter(torch.tensor([1.0, 0.0], device=device))
        params.append(calib)
    opt = torch.optim.Adam(params, lr=args.gate_lr, weight_decay=args.gate_l2)

    fit_i = np.where(fit_mask)[0]
    sel_i = np.where(sel_mask)[0]
    fit_t = torch.as_tensor(fit_i, dtype=torch.long, device=device)
    best_state, best_sel, best_ep = None, -np.inf, 0
    bce = nn.BCEWithLogitsLoss()

    for ep in range(1, args.gate_epochs + 1):
        gate.train()
        opt.zero_grad()
        shat, pi = gate(S_t, M_t, C_t, P_t)
        sh_fit = shat[fit_t]
        if args.gate_loss == "bce":
            logit = torch.nn.functional.softplus(calib[0]) * sh_fit + calib[1]
            loss = bce(logit, Y_t[fit_t])
        else:
            loss = _pairwise_loss(sh_fit, Y_t[fit_t], G[fit_i], rng, args.gate_max_pairs)
            if loss is None:
                break
        if args.gate_entropy_reg > 0:
            loss = loss + args.gate_entropy_reg * _entropy_penalty(pi[fit_t], M_t[fit_t])
        loss.backward()
        opt.step()

        gate.eval()
        with torch.no_grad():
            sh, _ = gate(S_t, M_t, C_t, P_t)
            sh = sh.cpu().numpy()
        crit = (_avg_metric(Y[sel_i], sh[sel_i], args.gate_objective)
                if len(sel_i) else _avg_metric(Y[fit_i], sh[fit_i], args.gate_objective))
        if crit == crit and crit > best_sel:
            best_sel, best_ep = crit, ep
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}

    if best_state is not None:
        gate.load_state_dict(best_state)
    gate.eval()
    report = {
        "status": "ok",
        "epochs": args.gate_epochs,
        "best_epoch": best_ep,
        "select_criterion": best_sel,
        "n_fit_days": int(fit_mask.sum()),
        "n_select_days": int(sel_mask.sum()),
        "n_fit_pos": int(Y[fit_i].sum()),
    }
    return gate, cov_mu, cov_sd, pid_of_patient, report


@torch.no_grad()
def apply_gate(gate, S, M, C, patient, pid_of_patient, cov_mu, cov_sd,
               use_cov, device):
    """Returns (shat, pi) for one patient's day matrix."""
    if S.shape[1] == 0:
        return np.empty(0), np.empty((0, S.shape[0]))
    Cn = np.where(M > 0.5, (C - cov_mu) / cov_sd, 0.0) if use_cov else np.zeros_like(C)
    S_t = torch.as_tensor(S.T, dtype=torch.float32, device=device)
    M_t = torch.as_tensor(M.T, dtype=torch.float32, device=device)
    C_t = torch.as_tensor(Cn.T, dtype=torch.float32, device=device)
    P_t = torch.full((S.shape[1],), pid_of_patient.get(patient, 0),
                     dtype=torch.long, device=device)
    shat, pi = gate(S_t, M_t, C_t, P_t)
    return shat.cpu().numpy(), pi.cpu().numpy()


# --- One experiment ----------------------------------------------------------

def _patient_mats(per_mod, names, patient, transform, cap, day_policy):
    """Build (val_tuple, test_tuple) for one patient, each (S, M, C, y, dates, p)."""
    vmaps = {n: per_mod[n][patient][0] for n in names if patient in per_mod[n]}
    tmaps = {n: per_mod[n][patient][1] for n in names if patient in per_mod[n]}
    if len(vmaps) < len(names) or len(tmaps) < len(names):
        # An expert with no record for this patient stays absent everywhere,
        # which the mask already handles.
        for n in names:
            vmaps.setdefault(n, {})
            tmaps.setdefault(n, {})
    stats = {n: _normal_stats(vmaps[n]) if vmaps[n] else (0.0, 1.0) for n in names}
    vdates = _day_axis(vmaps, names, day_policy)
    tdates = _day_axis(tmaps, names, day_policy)
    Sv, Mv, Cv, yv = _build_matrices(vmaps, names, stats, vdates, transform, cap)
    St, Mt, Ct, yt = _build_matrices(tmaps, names, stats, tdates, transform, cap)
    return (Sv, Mv, Cv, yv, vdates, patient), (St, Mt, Ct, yt, tdates, patient)


def _fuse_experiment(per_mod, names, args, device, rng, dump_rows=None,
                     fold=None, order=None):
    """
    Score every patient for one (fold, order) experiment.

    The cohort combiner is always fitted first. Under --scope cohort every
    patient uses it. Under --scope patient it becomes the warm start (gate), the
    shrinkage target (weighted) and the fallback for any patient whose own
    validation days are single-class.
    """
    patients = sorted(set.union(*[set(per_mod[n]) for n in names]), key=ds._natkey)
    val_mats, test_mats = [], []
    for p in patients:
        v, t = _patient_mats(per_mod, names, p, args.coverage_transform,
                             args.coverage_cap, args.day_policy)
        if t[0].shape[1] == 0:
            continue
        val_mats.append(v)
        test_mats.append(t)
    if not test_mats:
        return {}, {}

    use_cov = bool(args.coverage_weight)
    per_patient = (args.scope == "patient")
    grid = (_simplex_grid(len(names), args.weight_step)
            if args.fusion == "weighted" else None)

    cohort_ctx, cohort_w = None, None
    if args.fusion == "gate":
        cg, cmu, csd, cpid, crep = train_gate(val_mats, names, args, device, rng)
        cohort_ctx = (cg, cmu, csd, cpid)
        if cg is None:
            log.warning("  cohort gate not fitted (%s) — masked mean is used "
                        "wherever no gate is available.", crep.get("status"))
        else:
            log.info("  cohort gate: %d val days (%d relapse), select %s=%.3f "
                     "at epoch %d/%d",
                     crep["n_fit_days"], crep["n_fit_pos"],
                     args.gate_objective.upper(), crep["select_criterion"],
                     crep["best_epoch"], crep["epochs"])
    elif args.fusion == "weighted":
        cohort_w, ok = _pick_weights_on_val(val_mats, grid, use_cov,
                                            args.weight_objective)
        log.info("  cohort weights (val-tuned on %s%s): %s",
                 args.weight_objective.upper(), "" if ok else ", DEGENERATE",
                 ", ".join(f"{n}={w:.2f}" for n, w in zip(names, cohort_w)))

    def _combine(S, M, C, patient, ctx, w_p, order_rank):
        if args.fusion == "gate":
            g, mu, sd, pid = ctx if ctx is not None else (None, 0.0, 1.0, {})
            if g is not None:
                return apply_gate(g, S, M, C, patient, pid, mu, sd,
                                  args.gate_use_coverage, device)
            return _fuse_masked(S, M, C, use_cov=use_cov, mode="mean"), None
        if args.fusion == "rank":
            return _fuse_rank(S, M, order_rank), None
        if args.fusion.startswith("single:"):
            e = names.index(args.fusion.split(":", 1)[1])
            return np.where(M[e] > 0.5, S[e], np.nan), None
        if args.fusion == "max":
            return _fuse_masked(S, M, C, mode="max"), None
        return _fuse_masked(S, M, C, alpha=w_p, use_cov=use_cov, mode="mean"), None

    result, gate_stats = {}, defaultdict(list)
    n_fitted, n_fallback = 0, 0
    for v, t in zip(val_mats, test_mats):
        Sv, Mv, Cv, yv, vdates, p = v
        St, Mt, Ct, yt, tdates, _ = t

        ctx, w_p, order_rank = cohort_ctx, cohort_w, None

        if args.fusion == "rank":
            order_rank, rank_scores = _rank_experts_on_val(Sv, Mv, yv,
                                                           args.rank_objective)
            if args.verbose_patient:
                log.info("    %-8s ranking: %s", p, " > ".join(
                    f"{names[e]}({rank_scores[e]:.3f})" for e in order_rank))

        elif args.fusion == "weighted" and per_patient:
            w_own, ok = _pick_weights_on_val([v], grid, use_cov,
                                             args.weight_objective)
            if ok:
                n_fitted += 1
                L = float(args.shrink)
                w_p = (tuple((1.0 - L) * a + L * b
                             for a, b in zip(w_own, cohort_w)) if L > 0 else w_own)
            else:
                n_fallback += 1
            if args.verbose_patient:
                log.info("    %-8s weights: %s%s", p,
                         ", ".join(f"{n}={w:.2f}" for n, w in zip(names, w_p)),
                         "" if ok else "   (fallback: cohort)")

        elif args.fusion == "gate" and per_patient:
            init = (cohort_ctx[0].state_dict()
                    if (cohort_ctx is not None and cohort_ctx[0] is not None
                        and args.gate_warm_start == "cohort") else None)
            g_p, mu_p, sd_p, pid_p, rep_p = train_gate(
                [v], names, args, device, rng, init_state=init)
            if g_p is not None:
                n_fitted += 1
                ctx = (g_p, mu_p, sd_p, pid_p)
            else:
                n_fallback += 1
            if args.verbose_patient:
                if g_p is not None:
                    log.info("    %-8s gate: %d val days (%d relapse), select "
                             "%.3f at epoch %d", p, rep_p["n_fit_days"],
                             rep_p["n_fit_pos"], rep_p["select_criterion"],
                             rep_p["best_epoch"])
                else:
                    log.info("    %-8s gate: fallback to cohort (%s)",
                             p, rep_p.get("status"))

        vs, vpi = _combine(Sv, Mv, Cv, p, ctx, w_p, order_rank)
        ts, tpi = _combine(St, Mt, Ct, p, ctx, w_p, order_rank)
        ts_raw = np.array(ts, dtype=float, copy=True)

        if args.smooth_stage == "post":
            vs = _smooth_causal(vdates, vs, args.smooth_days, args.smooth_mode,
                                args.trim, args.gap_days, args.smooth_window)
            ts = _smooth_causal(tdates, ts, args.smooth_days, args.smooth_mode,
                                args.trim, args.gap_days, args.smooth_window)

        finite_t = np.isfinite(ts)
        if finite_t.sum() == 0:
            continue
        thr_src = (vs, yv) if (len(yv) and 0 < yv.sum() < len(yv)) else (ts[finite_t],
                                                                        yt[finite_t])
        f_thr = np.isfinite(thr_src[0])
        thr = ev._best_f1_threshold(thr_src[0][f_thr], np.asarray(thr_src[1])[f_thr])
        result[p] = ev._metrics(ts[finite_t], yt[finite_t], thr)

        if tpi is not None:
            gate_stats["pi"].append(tpi)
            gate_stats["mask"].append(Mt.T)
        if dump_rows is not None:
            for i, d in enumerate(tdates):
                row = {"fold": fold, "order": order, "patient": p,
                       "date": pd.Timestamp(d).date(), "label": int(yt[i]),
                       "fused_raw": float(ts_raw[i]),
                       "fused_smoothed": float(ts[i])}
                for e, n in enumerate(names):
                    row[f"z_{n}"] = float(St[e, i])
                    row[f"m_{n}"] = int(Mt[e, i] > 0.5)
                    row[f"pi_{n}"] = float(tpi[i, e]) if tpi is not None else float("nan")
                    if args.fusion == "weighted":
                        row[f"w_{n}"] = (float(w_p[e]) if w_p is not None
                                         else float("nan"))
                dump_rows.append(row)

    if per_patient and args.fusion in ("gate", "weighted"):
        log.info("  per-patient %s: %d fitted, %d fell back to cohort",
                 args.fusion, n_fitted, n_fallback)

    summary = {}
    if gate_stats["pi"]:
        PI = np.vstack(gate_stats["pi"])
        MK = np.vstack(gate_stats["mask"])
        summary["pi_mean"] = PI.mean(axis=0)
        summary["pi_mean_when_available"] = np.array([
            PI[MK[:, e] > 0.5, e].mean() if (MK[:, e] > 0.5).any() else float("nan")
            for e in range(len(names))])
        summary["availability"] = MK.mean(axis=0)
    return result, summary


# --- Smoother verification ---------------------------------------------------

def verify_smoother(features_dir, modalities, names, args, device):
    """
    Score the first expert twice: once with evaluate_unimodal smoothing, once raw
    plus the local causal smoother. Reports the maximum absolute deviation under
    both window conventions, which is what tells you whether post-stage numbers
    can be trusted.
    """
    name = names[0]
    ckpt_dir, tod = modalities[name]
    fold = args.folds[0] if args.folds else 0
    order = args.orders[0] if args.orders else 0
    ckpt = _resolve_ckpt(ckpt_dir, fold, order, args.train_tod)
    log.info("Verifying smoother on %s / %s", name, ckpt)
    ref = _score_one_modality(features_dir, ckpt, device, args.agg,
                              args.smooth_days, args.smooth_mode, args.trim,
                              args.gap_days, tod)
    raw = _score_one_modality(features_dir, ckpt, device, args.agg,
                              1, args.smooth_mode, args.trim,
                              args.gap_days, tod)
    if ref is None or raw is None:
        log.warning("  cannot verify: checkpoint missing."); return
    for window in ("records", "days"):
        worst, worst_where = 0.0, None
        for p in ref:
            for split in (0, 1):
                m_ref, m_raw = ref[p][split], raw[p][split]
                dates = sorted(set(m_ref) & set(m_raw))
                if not dates:
                    continue
                got = _smooth_causal(dates, [m_raw[d][0] for d in dates],
                                     args.smooth_days, args.smooth_mode,
                                     args.trim, args.gap_days, window)
                exp = np.array([m_ref[d][0] for d in dates])
                dev = float(np.nanmax(np.abs(got - exp)))
                if dev > worst:
                    worst, worst_where = dev, (p, "val" if split == 0 else "test")
        log.info("  window=%-7s  max |local - reference| = %.3e  %s",
                 window, worst, worst_where or "")
    log.info("  A deviation near 0 means --smooth-window with that value "
             "reproduces evaluate_unimodal exactly. Otherwise use "
             "--smooth-stage pre.")


# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Gated score-level fusion over frozen per-patient LSTM-AE experts.")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--modality", action="append", required=True,
                    help="NAME=CKPTDIR[:TEST_TOD]. One expert per flag.")
    ap.add_argument("--folds", type=int, nargs="*",
                    default=list(range(ds.N_REMISSION_FOLDS)))
    ap.add_argument("--orders", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--agg", choices=["mean", "median", "p90", "p95", "max"],
                    default="mean")
    ap.add_argument("--train-tod", choices=list(ds.TOD_CHOICES), default="day",
                    help="Fallback tod for checkpoint filename lookup only.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Fusion rule
    ap.add_argument("--fusion", default="gate",
                    help="gate | rank | mean | max | weighted | single:NAME")
    ap.add_argument("--day-policy", choices=["union", "intersect"], default="union",
                    help="union (default) scores a day if any expert is present, "
                         "which is what the availability mask is for. intersect "
                         "keeps only days every expert scored.")
    ap.add_argument("--scope", choices=["cohort", "patient"], default="cohort",
                    help="cohort = one combiner for everyone, fitted on the pooled "
                         "validation days. patient = one combiner per patient, "
                         "fitted on that patient's validation days only. Affects "
                         "--fusion weighted and --fusion gate.")
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="--scope patient + --fusion weighted only. Blend toward "
                         "the cohort weights: 0 fully personalised, 1 equals "
                         "cohort scope.")
    ap.add_argument("--gate-warm-start", choices=["cohort", "none"], default="cohort",
                    help="--scope patient + --fusion gate only. Initialise each "
                         "patient's gate from the cohort gate (default) or from "
                         "the mean-fusion baseline.")
    ap.add_argument("--verbose-patient", action="store_true",
                    help="Log each patient's fitted weights, gate selection or "
                         "expert ranking. 14 patients x 10 experiments is noisy.")
    ap.add_argument("--weight-step", type=float, default=0.1)
    ap.add_argument("--weight-objective", choices=["auroc", "avg"], default="avg")
    ap.add_argument("--rank-objective", choices=["auroc", "avg"], default="avg")

    # Gate
    ap.add_argument("--gate-hidden", type=int, default=16,
                    help="Hidden units in the gate MLP. 0 = linear gate.")
    ap.add_argument("--gate-epochs", type=int, default=300)
    ap.add_argument("--gate-lr", type=float, default=1e-2)
    ap.add_argument("--gate-l2", type=float, default=1e-3)
    ap.add_argument("--gate-loss", choices=["pairwise", "bce"], default="pairwise",
                    help="pairwise = smooth AUROC surrogate on within-patient "
                         "positive/negative day pairs (default).")
    ap.add_argument("--gate-max-pairs", type=int, default=20000)
    ap.add_argument("--gate-objective", choices=["auroc", "avg"], default="avg",
                    help="Criterion maximised for epoch selection on held-out "
                         "validation days.")
    ap.add_argument("--gate-val-frac", type=float, default=0.25,
                    help="Fraction of validation days held out for epoch "
                         "selection. 0 disables selection (fixed epochs).")
    ap.add_argument("--gate-split", choices=["chrono", "random"], default="chrono")
    ap.add_argument("--gate-entropy-reg", type=float, default=0.0,
                    help="Pull pi toward uniform-over-available. Larger values "
                         "shrink the gate toward mean fusion.")
    ap.add_argument("--gate-use-coverage", action="store_true",
                    help="Feed per-expert day coverage into the gate.")
    ap.add_argument("--patient-embed", type=int, default=0,
                    help="Patient embedding width. 0 = patient-agnostic gate.")

    # Smoothing
    ap.add_argument("--smooth-stage", choices=["post", "pre"], default="post",
                    help="post = smooth the FUSED series (gate -> smoother). "
                         "pre = fuse already-smoothed expert scores.")
    ap.add_argument("--smooth-days", type=int, default=14)
    ap.add_argument("--smooth-mode", choices=list(SMOOTH_MODES), default="mean")
    ap.add_argument("--smooth-window", choices=["records", "days"], default="records",
                    help="Trailing window convention for --smooth-stage post. "
                         "Confirm with --verify-smoother.")
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--gap-days", type=int, default=7)
    ap.add_argument("--verify-smoother", action="store_true")

    # Coverage weighting (non-gate rules)
    ap.add_argument("--coverage-weight", action="store_true")
    ap.add_argument("--coverage-transform", choices=["linear", "sqrt", "log"],
                    default="sqrt")
    ap.add_argument("--coverage-cap", type=float, default=None)

    ap.add_argument("--dump-gate", default=None,
                    help="CSV path for per-day z-scores, masks, pi, the raw convex "
                         "combination and the smoothed fused score (test days).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    modalities = _parse_modalities(args.modality)
    names = list(modalities)
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.fusion.startswith("single:"):
        pick = args.fusion.split(":", 1)[1]
        if pick not in names:
            raise SystemExit(f"single:{pick} is not one of {names}")
    elif args.fusion not in ("gate", "rank", "mean", "max", "weighted"):
        raise SystemExit(f"unknown --fusion '{args.fusion}'")

    if args.scope == "patient":
        if args.patient_embed > 0:
            log.warning("--patient-embed is meaningless under --scope patient "
                        "(each gate sees one patient); forcing it to 0.")
            args.patient_embed = 0
        if args.fusion not in ("gate", "weighted"):
            log.warning("--scope patient has no effect on --fusion %s.", args.fusion)
    if args.shrink and not (args.scope == "patient" and args.fusion == "weighted"):
        log.warning("--shrink only applies to --scope patient + --fusion weighted; "
                    "ignored.")

    if args.verify_smoother:
        verify_smoother(args.features_dir, modalities, names, args, device)
        return

    # In the post stage the experts are scored raw and the fused series is
    # smoothed once. In the pre stage the smoothing is done per expert.
    expert_smooth_days = 1 if args.smooth_stage == "post" else args.smooth_days

    mod_desc = ", ".join(
        f"{n}={str(d).split('checkpoints_train_')[-1]}"
        f"{'(matched)' if t is None else f':{t}'}"
        for n, (d, t) in modalities.items())
    log.info("GATED FUSION | %s | fusion=%s scope=%s | days=%s | "
             "smooth=%s %d/%s(%s) gap=%dd",
             mod_desc, args.fusion, args.scope, args.day_policy, args.smooth_stage,
             args.smooth_days, args.smooth_mode, args.smooth_window, args.gap_days)
    if args.scope == "patient" and args.fusion == "weighted" and args.shrink:
        log.info("  per-patient weights shrunk toward cohort with L=%.2f", args.shrink)
    if args.scope == "patient" and args.fusion == "gate":
        log.info("  per-patient gates, warm start = %s", args.gate_warm_start)
    if args.fusion == "gate":
        log.info("  gate: hidden=%d loss=%s objective=%s val-frac=%.2f (%s) "
                 "l2=%g entropy=%g cov=%s pat-emb=%d",
                 args.gate_hidden, args.gate_loss, args.gate_objective,
                 args.gate_val_frac, args.gate_split, args.gate_l2,
                 args.gate_entropy_reg, args.gate_use_coverage, args.patient_embed)

    exp_means = {"auroc": [], "auprc": [], "avg": [], "f1": []}
    per_runs = defaultdict(lambda: {"auroc": [], "auprc": [], "avg": [],
                                    "f1": [], "prevalence": []})
    pi_runs, avail_runs = [], []
    dump_rows = [] if args.dump_gate else None
    n_exp = 0

    for f in args.folds:
        for o in args.orders:
            per_mod = _score_modalities_one_experiment(
                args.features_dir, modalities, f, o, args.train_tod,
                device, args.agg, expert_smooth_days, args.smooth_mode,
                args.trim, args.gap_days)
            if per_mod is None:
                continue
            n_exp += 1
            res, summ = _fuse_experiment(per_mod, names, args, device, rng,
                                         dump_rows=dump_rows, fold=f, order=o)
            if "pi_mean" in summ:
                pi_runs.append(summ["pi_mean"])
                avail_runs.append(summ["availability"])
            aus, aps, avs, f1s = [], [], [], []
            for p in sorted(res, key=ds._natkey):
                r = res[p]
                for k in ("auroc", "auprc", "avg", "f1", "prevalence"):
                    per_runs[p][k].append(r[k])
                if r["auroc"] == r["auroc"]:
                    aus.append(r["auroc"])
                    aps.append(r["auprc"])
                    avs.append(r["avg"])
                f1s.append(r["f1"])
            exp_means["auroc"].append(np.mean(aus) if aus else float("nan"))
            exp_means["auprc"].append(np.mean(aps) if aps else float("nan"))
            exp_means["avg"].append(np.mean(avs) if avs else float("nan"))
            exp_means["f1"].append(np.mean(f1s) if f1s else float("nan"))
            log.info("  fused fold %d order %d  AUROC %.3f  AUPRC %.3f  "
                     "AVG %.3f  F1 %.3f",
                     f, o, exp_means["auroc"][-1], exp_means["auprc"][-1],
                     exp_means["avg"][-1], exp_means["f1"][-1])

    if n_exp == 0:
        log.warning("No fusable experiments (missing checkpoints?).")
        return

    log.info("=" * 70)
    log.info("PER-PATIENT SUMMARY  [fusion=%s scope=%s smooth=%s %d/%s gap=%dd]",
             args.fusion, args.scope, args.smooth_stage, args.smooth_days,
             args.smooth_mode, args.gap_days)
    for p in sorted(per_runs, key=ds._natkey):
        r = per_runs[p]
        au_m, au_s = ev._ms(r["auroc"])
        ap_m, ap_s = ev._ms(r["auprc"])
        avg_m, avg_s = ev._ms(r["avg"])
        f1_m, f1_s = ev._ms(r["f1"])
        prev_m, _ = ev._ms(r["prevalence"])
        n_au = sum(1 for x in r["auroc"] if x == x)
        log.info("  patient %-8s  AUROC %.3f +/- %.3f  AUPRC %.3f +/- %.3f  "
                 "AVG %.3f +/- %.3f  F1 %.3f +/- %.3f  prev %.3f  "
                 "(%d runs, %d scorable)",
                 p, au_m, au_s, ap_m, ap_s, avg_m, avg_s, f1_m, f1_s, prev_m,
                 len(r["auroc"]), n_au)

    if pi_runs:
        PI = np.vstack(pi_runs).mean(axis=0)
        AV = np.vstack(avail_runs).mean(axis=0)
        log.info("=" * 70)
        log.info("GATE WEIGHTS (test days, averaged over experiments)")
        for e, n in enumerate(names):
            log.info("  %-10s  mean pi %.3f   available on %.1f%% of days",
                     n, PI[e], 100.0 * AV[e])

    log.info("=" * 70)
    log.info("SUMMARY across %d experiments  "
             "[fusion=%s scope=%s agg=%s smooth=%s %d/%s gap=%dd]",
             n_exp, args.fusion, args.scope, args.agg, args.smooth_stage,
             args.smooth_days, args.smooth_mode, args.gap_days)
    for k in ("auroc", "auprc", "avg", "f1"):
        m, s = ev._ms(exp_means[k])
        log.info("  %-6s : %.3f +/- %.3f", k.upper(), m, s)

    if dump_rows:
        out = Path(args.dump_gate)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dump_rows).to_csv(out, index=False)
        log.info("wrote %d gated day rows -> %s", len(dump_rows), out)


if __name__ == "__main__":
    main()
