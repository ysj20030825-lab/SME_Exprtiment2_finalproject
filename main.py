#!/usr/bin/env python3
"""
main.py
Smart Mobility Engineering Experiment 2 Final Project

Algorithm:
    CF-AQARF
    Coarse-to-Fine Adaptive Quantile Annulus Reliability Field

Required files in the same folder:
    main.py
    train.py
    model.npz
    DH_FR1.mat

The grader will place DH_FR1.mat in the current working directory.
main() returns p_hat with shape (2, num_user).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio


MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.npz"
_MODEL_CACHE = None


def as_2_by_n(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D. Current shape: {arr.shape}")

    if arr.shape[0] == 2:
        return arr

    if arr.shape[1] == 2:
        return arr.T

    raise ValueError(f"{name} must have one dimension equal to 2. Current shape: {arr.shape}")


def load_input_data(mat_path: str = MAT_PATH):
    data = sio.loadmat(mat_path, squeeze_me=False)

    if "BS_positions" in data:
        bs_raw = data["BS_positions"]
    elif "p_bs" in data:
        bs_raw = data["p_bs"]
    else:
        raise KeyError("MAT file must contain variable 'BS_positions' or 'p_bs'.")

    if "d_hat" not in data:
        raise KeyError("MAT file must contain variable 'd_hat'.")

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

    return bs, d_hat


def load_model(model_path: str = MODEL_PATH) -> dict:
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not Path(model_path).exists():
        raise FileNotFoundError(
            "model.npz not found. Run 'python train.py' first and include "
            "model.npz in the repository root."
        )

    raw = np.load(model_path, allow_pickle=False)
    _MODEL_CACHE = {key: raw[key] for key in raw.files}

    return _MODEL_CACHE


def scalar(model: dict, key: str) -> float:
    return float(np.asarray(model[key]).reshape(-1)[0])


def make_grid(x_min: float, x_max: float, y_min: float, y_max: float, step: float) -> np.ndarray:
    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=float)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def choose_adaptive_band(d: np.ndarray, model: dict) -> int:
    cv = float(np.std(d) / (np.mean(np.abs(d)) + 1e-9))

    if cv <= scalar(model, "cv_low"):
        return 0

    if cv >= scalar(model, "cv_high"):
        return 2

    return 1


def field_score(grid: np.ndarray, d: np.ndarray, bs: np.ndarray, model: dict, band_id: int) -> np.ndarray:
    q_low_all = np.asarray(model["q_low_all"], dtype=float)
    q_high_all = np.asarray(model["q_high_all"], dtype=float)

    q_low = q_low_all[band_id]
    q_high = q_high_all[band_id]

    rho_in = d - q_high
    rho_out = d - q_low

    rho_in = np.maximum(rho_in, 0.0)
    rho_out = np.maximum(rho_out, rho_in + 1e-6)

    dx = grid[:, 0:1] - bs[0:1, :]
    dy = grid[:, 1:2] - bs[1:2, :]
    dist = np.sqrt(dx * dx + dy * dy)

    too_close = np.maximum(0.0, rho_in[None, :] - dist)
    too_far = np.maximum(0.0, dist - rho_out[None, :])

    sigma_in = np.asarray(model["sigma_in"], dtype=float)
    sigma_out = np.asarray(model["sigma_out"], dtype=float)
    weights = np.asarray(model["anchor_weights"], dtype=float)

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

    denom = np.sum(w)

    if not np.isfinite(denom) or denom <= 0:
        return pts[np.argmax(selected_score)]

    return np.sum(pts * w[:, None], axis=0) / denom


def your_algorithm(d_one: np.ndarray, BS_positions: np.ndarray) -> np.ndarray:
    model = load_model(MODEL_PATH)

    d = np.asarray(d_one, dtype=float).reshape(-1)
    bs = as_2_by_n(BS_positions, "BS_positions")

    if d.shape[0] != bs.shape[1]:
        raise ValueError(
            f"d_one length {d.shape[0]} and anchor count {bs.shape[1]} do not match."
        )

    band_id = choose_adaptive_band(d, model)

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

    return np.asarray(pred, dtype=float)


def main():
    BS_positions, d_hat = load_input_data(MAT_PATH)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    load_model(MODEL_PATH)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    return p_hat


if __name__ == "__main__":
    p_hat = main()
    print("p_hat shape:", p_hat.shape)