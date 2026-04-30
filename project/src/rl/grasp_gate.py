"""Learned grasp gate: small MLP classifier on top of the hand-coded
geometry features that is_grasp_ready already computes.

Pipeline:
  1) Logging — gym_env appends one JSON line per grasp attempt to
     GRASP_GATE_DATASET_PATH; the line carries the geometry feature dict
     plus the attempt's final outcome (lifted? fallen?).
  2) Offline training — `python -m src.rl.grasp_gate train` reads the
     JSONL, fits a 10 -> 64 -> 64 -> 1 MLP with binary cross-entropy.
  3) Inference — env loads the trained classifier when
     GRASP_GATE_MODE != "heuristic" and consults it before deciding to
     attach.

Feature order is fixed by FEATURE_KEYS so the same vector layout is used
at logging, training, and inference time. Adding a feature requires both
re-collecting data and retraining; do not silently reorder this list.
"""

import argparse
import json
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from src import config


# Feature order — DO NOT REORDER (silently invalidates trained checkpoints).
FEATURE_KEYS: Sequence[str] = (
    "ee_to_cube_dist",
    "min_finger_to_cube_dist",
    "total_contacts",
    "finger_contacts",
    "bracket_t",
    "bracket_score",
    "worst_tip_above_top",
    "face_plane_err",
    "face_ortho_err",
    "faces_opposite",
)
FEATURE_DIM = len(FEATURE_KEYS)


def featurize(debug: dict) -> np.ndarray:
    """Build a fixed-length numeric vector from is_grasp_ready's debug dict.

    Booleans are coerced to {0.0, 1.0}. Missing keys default to 0.0
    (defensive — should not happen on a fresh debug dict).
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, k in enumerate(FEATURE_KEYS):
        v = debug.get(k, 0.0)
        if isinstance(v, bool):
            out[i] = 1.0 if v else 0.0
        else:
            try:
                out[i] = float(v)
            except (TypeError, ValueError):
                out[i] = 0.0
    return out


class _GraspGateModel(nn.Module):
    """Small MLP: feature_dim -> 64 -> 64 -> 1 (sigmoid output)."""

    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logit; sigmoid applied at predict time


# -------------------------------- inference --------------------------------

class LearnedGraspGate:
    """Trained classifier wrapper for env-side inference.

    Constructor reads the checkpoint at config.GRASP_GATE_MODEL_PATH (or
    a caller-supplied path). If the file does not exist, the gate marks
    itself unloaded and predict() returns the heuristic fallback. Caller
    can check .loaded before consulting.
    """

    def __init__(self, model_path: Optional[str] = None,
                 threshold: Optional[float] = None):
        self.threshold = (
            float(getattr(config, "GRASP_GATE_THRESHOLD", 0.5))
            if threshold is None else float(threshold)
        )
        self._model = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self.loaded = False
        path = model_path or getattr(config, "GRASP_GATE_MODEL_PATH", None)
        if path and os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                self._model = _GraspGateModel(FEATURE_DIM)
                self._model.load_state_dict(ckpt["state_dict"])
                self._model.eval()
                self._mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
                self._std = np.asarray(ckpt["feature_std"], dtype=np.float32)
                self.loaded = True
            except (OSError, KeyError, RuntimeError) as exc:
                print(f"[grasp_gate] failed to load {path}: {exc}; "
                      f"falling back to heuristic")

    def predict_prob(self, debug: dict) -> float:
        """Return P(grasp succeeds) ∈ [0, 1]. Returns 1.0 (open gate) if
        the classifier is unloaded so callers degrade to heuristic-only."""
        if not self.loaded or self._model is None:
            return 1.0
        feat = featurize(debug)
        feat_n = (feat - self._mean) / (self._std + 1e-6)
        with torch.no_grad():
            logit = self._model(torch.from_numpy(feat_n).unsqueeze(0))
            prob = float(torch.sigmoid(logit).item())
        return prob


# --------------------------- dataset + offline train ----------------------

def _read_dataset(dataset_path: str) -> tuple:
    """Read a JSONL grasp dataset; return (X, y, n_pos, n_neg)."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    feats, labels = [], []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Label: did the attempt actually succeed (cube lifted, not fallen)?
            label = bool(row.get("lifted", False)) and not bool(row.get("fallen", False))
            feats.append(featurize(row))
            labels.append(1 if label else 0)
    if not feats:
        raise RuntimeError(f"Dataset {dataset_path} contained no usable rows.")
    X = np.stack(feats).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y, int(y.sum()), int((1.0 - y).sum())


def train_classifier(dataset_path: Optional[str] = None,
                     model_path: Optional[str] = None,
                     epochs: int = 80, batch_size: int = 64,
                     lr: float = 1e-3, val_frac: float = 0.2,
                     seed: int = 0) -> dict:
    """Fit the classifier; return a dict of training stats."""
    dataset_path = dataset_path or config.GRASP_GATE_DATASET_PATH
    model_path = model_path or config.GRASP_GATE_MODEL_PATH

    X, y, n_pos, n_neg = _read_dataset(dataset_path)
    n = len(X)
    print(f"Loaded {n} samples from {dataset_path}: {n_pos} pos, {n_neg} neg")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    n_val = max(1, int(round(n * val_frac)))
    X_tr, X_va = X[:-n_val], X[-n_val:]
    y_tr, y_va = y[:-n_val], y[-n_val:]

    # Standardize features using train-set stats; persist with the model.
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    X_tr_n = (X_tr - mean) / std
    X_va_n = (X_va - mean) / std

    torch.manual_seed(seed)
    model = _GraspGateModel(FEATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Imbalanced binary classes: weight positives by neg/pos ratio.
    pos_weight = max(1.0, float(n_neg) / max(1, n_pos))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    X_tr_t = torch.from_numpy(X_tr_n)
    y_tr_t = torch.from_numpy(y_tr).unsqueeze(-1)
    X_va_t = torch.from_numpy(X_va_n)
    y_va_t = torch.from_numpy(y_va).unsqueeze(-1)

    n_train = len(X_tr_t)
    best_val_acc = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            b = idx[i:i + batch_size]
            opt.zero_grad()
            logits = model(X_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_logits = model(X_tr_t)
            va_logits = model(X_va_t)
            tr_acc = float(((torch.sigmoid(tr_logits) > 0.5).float() == y_tr_t).float().mean())
            va_acc = float(((torch.sigmoid(va_logits) > 0.5).float() == y_va_t).float().mean())
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "feature_keys": list(FEATURE_KEYS),
        "n_train": n - n_val,
        "n_val": n_val,
        "best_val_acc": best_val_acc,
        "pos_weight": pos_weight,
    }, model_path)
    print(f"Saved classifier to {model_path} (val_acc={best_val_acc:.3f})")
    return {
        "n_total": n, "n_pos": n_pos, "n_neg": n_neg,
        "n_train": n - n_val, "n_val": n_val,
        "best_val_acc": best_val_acc, "model_path": model_path,
    }


# ------------------------------ CLI entry ---------------------------------

def _build_parser():
    p = argparse.ArgumentParser(prog="python -m src.rl.grasp_gate",
                                description="Train / inspect the learned grasp gate.")
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="Train classifier from logged dataset.")
    tr.add_argument("--dataset", default=None,
                    help="Path to grasp_dataset.jsonl (default: from config).")
    tr.add_argument("--out", default=None,
                    help="Where to write grasp_gate.pt (default: from config).")
    tr.add_argument("--epochs", type=int, default=80)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = _build_parser().parse_args()
    if args.command == "train":
        train_classifier(
            dataset_path=args.dataset, model_path=args.out,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed,
        )


if __name__ == "__main__":
    main()
