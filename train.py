#!/usr/bin/env python3
"""
train.py

Smart Mobility Engineering Experiment 2 Final Project

Algorithm:
    CF-AQARF
    Coarse-to-Fine Adaptive Quantile Annulus Reliability Field

Role:
    1. Load DH_FR1.mat
    2. Use provided 700 labeled samples
    3. Compute anchor-wise RTT error quantiles
    4. Validate with 5-fold cross validation
    5. Save final model parameters to model.npz

Required packages:
    numpy
    scipy

No requirements.txt is needed because numpy and scipy are included
in the standard grading environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.io as sio


DEFAULT_MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.npz"


def as_2_by_n(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D. Current shape: {arr.shape}")

    if arr.shape[0] == 2:
        return arr

    if arr.shape[1] == 2:
        return arr.T

    raise ValueError(f"{name} must have one dimension equal to 2. Current shape: {arr.shape}")


def load_training_data(mat_path: str):
    data = sio.loadmat(mat_path, squeeze_me=False)

    if "p" not in data:
        raise KeyError("MAT file must contain variable 'p'.")
    if "d_hat" not in data:
        raise KeyError("MAT file must contain variable 'd_hat'.")

    if "BS_positions" in data:
        bs_raw = data["BS_positions"]
    elif "p_bs" in data:
        bs_raw = data["p_bs"]
    else:
        raise KeyError("MAT file must contain variable 'BS_positions'.")

    p = as_2_by_n(data["p"], "p")
    bs = as_2_by_n(bs_raw, "BS_positions")
    d_hat = np.asarray(data["d_hat"], dtype=float)

    if d_hat.ndim != 2:
        raise ValueError(f"d_hat must be 2-D. Current shape: {d_hat.shape}")

    if d_hat.shape[0] != bs.shape[1] and d_hat.shape[1] == bs.shape[1]:
        d_hat = d_hat.T

    if d_hat.shape[0] != bs.shape[1]:
        raise ValueError(
            f"d_hat and BS_positions are inconsistent. "
            f"d_hat shape={d_hat.shape}, BS_positions shape={bs.shape}"
        )

    if p.shape[1] != d_hat.shape[1]:
        raise ValueError(
            f"p and d_hat sample counts are inconsistent. "
            f"p shape={p.shape}, d_hat shape={d_hat.shape}"
        )

    return p, d_hat, bs


def true_distances(p: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """
    p:  (2, N)
    bs: (2, 18)

    return:
        true distance matrix with shape (18, N)
    """
    dx = p[0:1, :] - bs[0:1, :].T
    dy = p[1:2, :] - bs[1:2, :].T
    return np.sqrt(dx * dx + dy * dy)


def fit_cf_aqarf_model(p: np.ndarray, d_hat: np.ndarray, bs: np.ndarray) -> dict:
    """
    Fit CF-AQARF parameters from labeled training data.

    The model does not store training labels directly.
    It stores anchor-wise error quantiles and reliability parameters.
    """
    dist_true = true_distances(p, bs)
    error = d_hat - dist_true

    quantile_pairs = np.array(
        [
            [0.35, 0.75],
            [0.25, 0.90],
            [0.10, 0.95],
        ],
        dtype=float,
    )

    q_low_all = []
    q_high_all = []

    for q_low, q_high in quantile_pairs:
        q_low_all.append(np.quantile(error, q_low, axis=1))
        q_high_all.append(np.quantile(error, q_high, axis=1))

    q_low_all = np.vstack(q_low_all)
    q_high_all = np.vstack(q_high_all)

    error_std = np.std(error, axis=1)
    anchor_weights = 1.0 / (error_std + 1e-6)
    anchor_weights = anchor_weights / (np.mean(anchor_weights) + 1e-9)

    error_iqr = np.quantile(error, 0.75, axis=1) - np.quantile(error, 0.25, axis=1)

    sigma_in = np.maximum(6.0, 0.80 * error_iqr + 2.0)
    sigma_out = np.maximum(3.0, 0.45 * error_iqr + 1.0)

    cv = np.std(d_hat, axis=0) / (np.mean(np.abs(d_hat), axis=0) + 1e-9)
    cv_low = np.quantile(cv, 0.33)
    cv_high = np.quantile(cv, 0.66)

    x_min = min(np.min(p[0]), np.min(bs[0])) - 15.0
    x_max = max(np.max(p[0]), np.max(bs[0])) + 15.0
    y_min = min(np.min(p[1]), np.min(bs[1])) - 15.0
    y_max = max(np.max(p[1]), np.max(bs[1])) + 15.0

    model = {
        "q_low_all": q_low_all,
        "q_high_all": q_high_all,
        "anchor_weights": anchor_weights,
        "sigma_in": sigma_in,
        "sigma_out": sigma_out,
        "cv_low": np.array(cv_low),
        "cv_high": np.array(cv_high),
        "x_min": np.array(x_min),
        "x_max": np.array(x_max),
        "y_min": np.array(y_min),
        "y_max": np.array(y_max),

        "coarse_step": np.array(2.0),
        "fine_step": np.array(0.25),
        "fine_half_width": np.array(6.0),
        "coarse_top_ratio": np.array(0.015),
        "fine_top_ratio": np.array(0.030),
        "softmax_beta": np.array(7.0),
    }

    return model


def save_model(model: dict, out_path: str = MODEL_PATH) -> None:
    np.savez(out_path, **model)


def scalar(model: dict, key: str) -> float:
    return float(np.asarray(model[key]).reshape(-1)[0])


def make_grid(x_min: float, x_max: float, y_min: float, y_max: float, step: float) -> np.ndarray:
    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=float)
    X, Y = np.meshgrid(xs, ys)
    return np.column_stack((X.ravel(), Y.ravel()))


def choose_band(d: np.ndarray, model: dict) -> int:
    cv = float(np.std(d) / (np.mean(np.abs(d)) + 1e-9))

    if cv <= scalar(model, "cv_low"):
        return 0
    if cv >= scalar(model, "cv_high"):
        return 2
    return 1


def field_score(grid: np.ndarray, d: np.ndarray, bs: np.ndarray, model: dict, band_id: int) -> np.ndarray:
    q_low = model["q_low_all"][band_id]
    q_high = model["q_high_all"][band_id]

    rho_in = d - q_high
    rho_out = d - q_low

    rho_in = np.maximum(rho_in, 0.0)
    rho_out = np.maximum(rho_out, rho_in + 1e-6)

    dx = grid[:, 0:1] - bs[0:1, :]
    dy = grid[:, 1:2] - bs[1:2, :]
    dist = np.sqrt(dx * dx + dy * dy)

    too_close = np.maximum(0.0, rho_in[None, :] - dist)
    too_far = np.maximum(0.0, dist - rho_out[None, :])

    sigma_in = model["sigma_in"]
    sigma_out = model["sigma_out"]
    weights = model["anchor_weights"]

    score_each = np.exp(-0.5 * (too_close / (sigma_in[None, :] + 1e-9)) ** 2)
    score_each *= np.exp(-0.5 * (too_far / (sigma_out[None, :] + 1e-9)) ** 2)

    score = np.sum(score_each * weights[None, :], axis=1)
    score = score / (np.sum(weights) + 1e-9)

    return score


def soft_centroid(grid: np.ndarray, score: np.ndarray, top_ratio: float, beta: float) -> np.ndarray:
    k = max(5, int(np.ceil(len(score) * top_ratio)))
    idx = np.argpartition(score, -k)[-k:]

    pts = grid[idx]
    selected_score = score[idx]

    z = beta * (selected_score - np.max(selected_score))
    w = np.exp(z)

    return np.sum(pts * w[:, None], axis=0) / (np.sum(w) + 1e-9)


def predict_one(d: np.ndarray, bs: np.ndarray, model: dict) -> np.ndarray:
    band_id = choose_band(d, model)

    coarse_grid = make_grid(
        scalar(model, "x_min"),
        scalar(model, "x_max"),
        scalar(model, "y_min"),
        scalar(model, "y_max"),
        scalar(model, "coarse_step"),
    )

    coarse_score = field_score(coarse_grid, d, bs, model, band_id)

    coarse_center = soft_centroid(
        coarse_grid,
        coarse_score,
        scalar(model, "coarse_top_ratio"),
        scalar(model, "softmax_beta"),
    )

    half = scalar(model, "fine_half_width")

    x_min = max(scalar(model, "x_min"), coarse_center[0] - half)
    x_max = min(scalar(model, "x_max"), coarse_center[0] + half)
    y_min = max(scalar(model, "y_min"), coarse_center[1] - half)
    y_max = min(scalar(model, "y_max"), coarse_center[1] + half)

    fine_grid = make_grid(
        x_min,
        x_max,
        y_min,
        y_max,
        scalar(model, "fine_step"),
    )

    fine_score = field_score(fine_grid, d, bs, model, band_id)

    pred = soft_centroid(
        fine_grid,
        fine_score,
        scalar(model, "fine_top_ratio"),
        scalar(model, "softmax_beta"),
    )

    return pred


def predict_all(d_hat: np.ndarray, bs: np.ndarray, model: dict) -> np.ndarray:
    n = d_hat.shape[1]
    pred = np.zeros((2, n), dtype=float)

    for u in range(n):
        pred[:, u] = predict_one(d_hat[:, u], bs, model)

    return pred


def linear_ls_position(d: np.ndarray, bs: np.ndarray, k: int = 6) -> np.ndarray:
    idx = np.argsort(d)[:k]
    ref = idx[0]

    x1, y1 = bs[:, ref]
    d1 = d[ref]

    A = []
    q = []

    for j in idx[1:]:
        xj, yj = bs[:, j]
        dj = d[j]

        A.append([2.0 * (xj - x1), 2.0 * (yj - y1)])
        q.append(d1 * d1 - dj * dj + xj * xj - x1 * x1 + yj * yj - y1 * y1)

    A = np.asarray(A, dtype=float)
    q = np.asarray(q, dtype=float)

    try:
        sol, *_ = np.linalg.lstsq(A, q, rcond=None)

        if np.all(np.isfinite(sol)):
            return sol

    except np.linalg.LinAlgError:
        pass

    return np.mean(bs[:, idx], axis=1)


def predict_linear_ls(d_hat: np.ndarray, bs: np.ndarray, k: int = 6) -> np.ndarray:
    n = d_hat.shape[1]
    pred = np.zeros((2, n), dtype=float)

    for u in range(n):
        pred[:, u] = linear_ls_position(d_hat[:, u], bs, k=k)

    return pred


def position_error(p_true: np.ndarray, p_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((p_true - p_pred) ** 2, axis=0))


def summarize_errors(error: np.ndarray) -> dict:
    return {
        "MAE": float(np.mean(error)),
        "RMSE": float(np.sqrt(np.mean(error * error))),
        "Min": float(np.min(error)),
        "Max": float(np.max(error)),
        "Median": float(np.median(error)),
        "P95": float(np.quantile(error, 0.95)),
    }


def print_metric_table(rows: list[tuple[str, dict]]) -> None:
    print()
    print("| Method | MAE [m] | RMSE [m] | Min [m] | Max [m] | Median [m] | P95 [m] |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    for name, m in rows:
        print(
            f"| {name} | {m['MAE']:.3f} | {m['RMSE']:.3f} | {m['Min']:.3f} | "
            f"{m['Max']:.3f} | {m['Median']:.3f} | {m['P95']:.3f} |"
        )


def make_kfold_indices(n: int, n_folds: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.array_split(perm, n_folds)


def cross_validate(p: np.ndarray, d_hat: np.ndarray, bs: np.ndarray, n_folds: int = 5) -> None:
    folds = make_kfold_indices(p.shape[1], n_folds=n_folds, seed=42)

    pred_cf = np.zeros_like(p, dtype=float)
    pred_ls = np.zeros_like(p, dtype=float)

    all_idx = np.arange(p.shape[1])

    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx)

        model = fit_cf_aqarf_model(p[:, train_idx], d_hat[:, train_idx], bs)

        pred_cf[:, test_idx] = predict_all(d_hat[:, test_idx], bs, model)
        pred_ls[:, test_idx] = predict_linear_ls(d_hat[:, test_idx], bs, k=6)

        fold_error = position_error(p[:, test_idx], pred_cf[:, test_idx])

        print(
            f"Fold {fold_id}: "
            f"CF-AQARF MAE={np.mean(fold_error):.3f}, "
            f"RMSE={np.sqrt(np.mean(fold_error * fold_error)):.3f}"
        )

    rows = [
        ("Linear LS K=6", summarize_errors(position_error(p, pred_ls))),
        ("CF-AQARF", summarize_errors(position_error(p, pred_cf))),
    ]

    print_metric_table(rows)


def main() -> None:
    mat_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_MAT_PATH

    if not Path(mat_path).exists():
        raise FileNotFoundError(
            f"{mat_path} not found. Put DH_FR1.mat in this folder, "
            f"or run: python train.py path/to/DH_FR1.mat"
        )

    p, d_hat, bs = load_training_data(mat_path)

    print("Loaded data")
    print("p:", p.shape)
    print("d_hat:", d_hat.shape)
    print("BS_positions:", bs.shape)

    print()
    print("5-fold validation. Quantiles are computed only from each train fold.")
    cross_validate(p, d_hat, bs, n_folds=5)

    print()
    print("Training final CF-AQARF model on all provided labeled samples.")

    model = fit_cf_aqarf_model(p, d_hat, bs)
    save_model(model, MODEL_PATH)

    print(f"Saved final model: {MODEL_PATH}")


if __name__ == "__main__":
    main()