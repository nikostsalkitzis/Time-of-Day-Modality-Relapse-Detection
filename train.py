"""
Trains a separate LSTM autoencoder per patient for each experiment and writes one
checkpoint bundle per experiment holding all per-patient weights.

An experiment = (held-out remission fold `fold_idx`) x (relapse ordering 0/1).
With 5 remission folds and 2 relapse orderings this is 10 experiments. For each
experiment and each patient:
  * the patient's model is trained only on that patient's own training-remission
    windows (4 of the 5 remission folds) under an MSE reconstruction objective,
  * early stopping uses that patient's own validation-normal reconstruction loss,
  * scaling and imputation are that patient's own artifacts.

This is many small trainings, roughly n_patients per experiment. Each is fast
because a patient holds a fraction of the pooled data, but the whole run is
longer than a single global model.

Checkpoint bundle (per experiment):
    {
      "fold_idx", "relapse_order",
      "state_dicts"     : patient -> state_dict,
      "artifacts"       : patient -> (median, mean, std),
      "half_assignment" : patient -> bool,
      "feature_cols", "window_bins", "stride_bins",
      "model_kwargs",   # shared architecture across patients
      "best_val_recon"  : patient -> float,
    }

Run:
    python train.py --features-dir /gpu-data/eprevention/ntsal/features \
                    --ckpt-dir ./checkpoints
Restrict with --folds / --orders / --patients (subset of user IDs).
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import dataset as ds
from model import LSTMAutoencoder, reconstruction_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def _patient_val_normal(fold, patient) -> np.ndarray:
    """That patient's validation-NORMAL windows, for early stopping."""
    wins = [it["windows"] for it in fold.val_items
            if it["patient"] == patient and it["label"] == 0]
    if not wins:
        return np.empty((0, ds.WINDOW_BINS, fold.feat_dim), dtype=np.float32)
    return np.concatenate(wins, axis=0)


@torch.no_grad()
def _mean_recon(model, arr, device, batch_size=512) -> float:
    if len(arr) == 0:
        return float("nan")
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(arr), batch_size):
        x = torch.from_numpy(arr[i:i + batch_size]).to(device)
        total += float(reconstruction_error(x, model(x)).sum())
        n += len(x)
    return total / max(n, 1)


def _train_patient_model(train_arr, val_norm, model_kwargs, args, device):
    """Train one patient's AE; return (best_state_cpu, best_val)."""
    loader = DataLoader(TensorDataset(torch.from_numpy(train_arr)),
                        batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = LSTMAutoencoder(**model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_left = args.patience
    have_val = len(val_norm) > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = reconstruction_error(xb, model(xb)).mean()
            loss.backward()
            opt.step()
            running += float(loss) * len(xb)
            seen += len(xb)
        train_loss = running / max(seen, 1)

        # Val-normal reconstruction when available, train loss otherwise.
        monitor = _mean_recon(model, val_norm, device) if have_val else train_loss
        if monitor < best_val - 1e-6:
            best_val = monitor
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    return best_state, best_val


def train_one_experiment(features_dir, fold_idx, relapse_order, rng, args, device) -> Path:
    log.info("=" * 70)
    log.info("Training experiment: fold %d, relapse order %d, tod %s",
             fold_idx, relapse_order, args.tod)
    fold = ds.build_fold_data(
        features_dir, fold_idx, relapse_order=relapse_order, rng=rng,
        train_tod=args.tod, test_tod=args.tod,
        night_start_min=args.night_start_min, night_end_min=args.night_end_min,
    )

    model_kwargs = dict(
        input_dim=fold.feat_dim,
        hidden_dim=args.hidden,
        latent_dim=args.latent,
        num_layers=args.layers,
        seq_len=ds.WINDOW_BINS,
        dropout=args.dropout,
    )

    patients = sorted(fold.train_windows_by_patient, key=ds._natkey)
    if args.patients:
        patients = [p for p in patients if p in set(args.patients)]

    state_dicts = {}
    best_vals = {}
    for p in patients:
        train_arr = fold.train_windows_by_patient[p]
        val_norm = _patient_val_normal(fold, p)
        best_state, best_val = _train_patient_model(
            train_arr, val_norm, model_kwargs, args, device)
        state_dicts[p] = best_state
        best_vals[p] = best_val
        log.info("  patient %-8s | %5d train windows | best monitor MSE %.6f",
                 p, len(train_arr), best_val)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold_{fold_idx}_rel_{relapse_order}_{args.tod}.pt"
    torch.save({
        "fold_idx": fold_idx,
        "relapse_order": relapse_order,
        "state_dicts": state_dicts,
        "artifacts": fold.artifacts,
        "half_assignment": fold.half_assignment,
        "feature_cols": ds.FEATURE_COLS,
        "window_bins": ds.WINDOW_BINS,
        "stride_bins": ds.STRIDE_BINS,
        "model_kwargs": model_kwargs,
        "best_val_recon": best_vals,
        "train_tod": args.tod,
        "night_start_min": args.night_start_min,
        "night_end_min": args.night_end_min,
    }, ckpt_path)
    log.info("  saved %d per-patient models -> %s", len(state_dicts), ckpt_path)
    return ckpt_path


def main():
    ap = argparse.ArgumentParser(description="Train PER-PATIENT LSTM-AEs per experiment.")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(ds.N_REMISSION_FOLDS)))
    ap.add_argument("--orders", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--patients", nargs="*", default=None,
                    help="Optional subset of patient IDs to train.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--latent", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--tod", choices=list(ds.TOD_CHOICES), default="both",
                    help="Time-of-day regime for TRAINING (filters train windows, "
                         "artifact fit, and validation). 'both' = original pipeline. "
                         "Train one set each of day/night/both; the both-set is "
                         "later evaluated on day or night at test time.")
    ap.add_argument("--night-start", default="22:00",
                    help="Night start clock HH:MM (default 22:00). Night wraps midnight.")
    ap.add_argument("--night-end", default="07:00",
                    help="Night end clock HH:MM (default 07:00).")
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, seeds weight init AND the half-assignment RNG.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.night_start_min = ds.parse_clock(args.night_start)
    args.night_end_min = ds.parse_clock(args.night_end)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        log.info("Device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        log.info("Device: %s", device)
    log.info("Experiments: folds %s x orders %s = %d total (per-patient models each)",
             args.folds, args.orders, len(args.folds) * len(args.orders))
    log.info("Training regime (tod): %s | night %02d:%02d -> %02d:%02d",
             args.tod,
             args.night_start_min // 60, args.night_start_min % 60,
             args.night_end_min // 60, args.night_end_min % 60)

    for fold_idx in args.folds:
        for relapse_order in args.orders:
            train_one_experiment(args.features_dir, fold_idx, relapse_order, rng, args, device)


if __name__ == "__main__":
    main() 
