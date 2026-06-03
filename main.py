import numpy as np
import scipy.io as sio


MODEL_PATH = "model.npz"
_MODEL_CACHE = None


def _as_2_by_n(arr, name):
    """
    좌표 배열을 (2, N) 형태로 맞추기 위한 보조 함수
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D. Current shape: {arr.shape}")

    if arr.shape[0] == 2:
        return arr

    if arr.shape[1] == 2:
        return arr.T

    raise ValueError(f"{name} must have one dimension equal to 2. Current shape: {arr.shape}")


def _load_model():
    """
    train.py에서 생성한 model.npz를 불러오는 함수
    """
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    raw = np.load(MODEL_PATH, allow_pickle=False)
    _MODEL_CACHE = {key: raw[key] for key in raw.files}

    return _MODEL_CACHE


def _scalar(model, key):
    """
    model.npz에 저장된 scalar 값을 float으로 변환
    """
    return float(np.asarray(model[key]).reshape(-1)[0])


def _make_grid(x_min, x_max, y_min, y_max, step):
    """
    2차원 후보 위치 격자 생성
    """
    xs = np.arange(x_min, x_max + 0.5 * step, step, dtype=float)
    ys = np.arange(y_min, y_max + 0.5 * step, step, dtype=float)
    x_grid, y_grid = np.meshgrid(xs, ys)

    return np.column_stack((x_grid.ravel(), y_grid.ravel()))


def _choose_adaptive_band(d, model):
    """
    입력 RTT의 산포를 기준으로 narrow / normal / wide band 선택
    """
    cv = float(np.std(d) / (np.mean(np.abs(d)) + 1e-9))

    if cv <= _scalar(model, "cv_low"):
        return 0

    if cv >= _scalar(model, "cv_high"):
        return 2

    return 1


def _field_score(grid, d, p_bs, model, band_id):
    """
    후보 위치 grid에 대해 고리형 신뢰도장 점수 계산
    """
    q_low_all = np.asarray(model["q_low_all"], dtype=float)
    q_high_all = np.asarray(model["q_high_all"], dtype=float)

    q_low = q_low_all[band_id]
    q_high = q_high_all[band_id]

    rho_in = d - q_high
    rho_out = d - q_low

    rho_in = np.maximum(rho_in, 0.0)
    rho_out = np.maximum(rho_out, rho_in + 1e-6)

    dx = grid[:, 0:1] - p_bs[0:1, :]
    dy = grid[:, 1:2] - p_bs[1:2, :]
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


def _soft_centroid(grid, score, top_ratio, beta):
    """
    상위 신뢰도 영역의 soft centroid 계산
    """
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


def your_algorithm(d_one, p_bs):
    """
    본인 알고리즘 작성(필요시 상단에 추가적인 함수 작성 가능)

    CF-AQARF:
    Coarse-to-Fine Adaptive Quantile Annulus Reliability Field

    입력:
        d_one : 한 사용자에 대한 18개 앵커 RTT 측정값, shape (18,)
        p_bs  : 18개 앵커 좌표, shape (2, 18)

    출력:
        측위결과 : [x, y], shape (2,)
    """
    model = _load_model()

    d = np.asarray(d_one, dtype=float).reshape(-1)
    p_bs = _as_2_by_n(p_bs, "BS_positions")

    if d.shape[0] != p_bs.shape[1]:
        raise ValueError(
            f"d_one length {d.shape[0]} and anchor count {p_bs.shape[1]} do not match."
        )

    band_id = _choose_adaptive_band(d, model)

    # 1) Coarse search
    coarse_grid = _make_grid(
        _scalar(model, "x_min"),
        _scalar(model, "x_max"),
        _scalar(model, "y_min"),
        _scalar(model, "y_max"),
        _scalar(model, "coarse_step"),
    )

    coarse_score = _field_score(coarse_grid, d, p_bs, model, band_id)

    coarse_center = _soft_centroid(
        coarse_grid,
        coarse_score,
        _scalar(model, "coarse_top_ratio"),
        _scalar(model, "softmax_beta"),
    )

    # 2) Fine search
    half = _scalar(model, "fine_half_width")

    x_min = max(_scalar(model, "x_min"), coarse_center[0] - half)
    x_max = min(_scalar(model, "x_max"), coarse_center[0] + half)
    y_min = max(_scalar(model, "y_min"), coarse_center[1] - half)
    y_max = min(_scalar(model, "y_max"), coarse_center[1] + half)

    fine_grid = _make_grid(
        x_min,
        x_max,
        y_min,
        y_max,
        _scalar(model, "fine_step"),
    )

    fine_score = _field_score(fine_grid, d, p_bs, model, band_id)

    # 3) 최종 측위 결과
    result = _soft_centroid(
        fine_grid,
        fine_score,
        _scalar(model, "fine_top_ratio"),
        _scalar(model, "softmax_beta"),
    )

    return np.asarray(result, dtype=float)


def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = 'DH_FR1.mat'

    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)     # (2, 18)
    d_hat = np.asarray(data['d_hat'], dtype=float)                   # (18, num_user)
    p = np.asarray(data['p'], dtype=float) if 'p' in data else None  # (2, num_user) — GT 위치

    # 입력 모양 보정
    BS_positions = _as_2_by_n(BS_positions, "BS_positions")

    if d_hat.shape[0] != BS_positions.shape[1] and d_hat.shape[1] == BS_positions.shape[1]:
        d_hat = d_hat.T

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat


if __name__ == "__main__":
    main()
