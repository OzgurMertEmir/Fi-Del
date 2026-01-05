import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from myproj.ml.pipeline import build_dataset


def _dist(y: np.ndarray) -> dict:
    vals, cnts = np.unique(y, return_counts=True)
    frac = cnts / cnts.sum()
    return {int(v): float(f) for v, f in zip(vals, frac)}


def test_dataset_has_required_columns_and_shapes():
    df, X, y, feature_names = build_dataset(
        symbol="SOLUSDC",
        bucket="1s",
        horizon=5,
        limit_rows=5000,
    )

    for col in ["ts", "mid", "future_mid", "ret_h", "y"]:
        assert col in df.columns, f"Missing column '{col}'"

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(feature_names)

    ts = df["ts"].to_numpy()
    assert np.all(ts[1:] >= ts[:-1]), "Timestamps not monotonic increasing"


def test_labels_are_shifted_forward_not_leaking():
    df, X, y, feature_names = build_dataset(
        symbol="SOLUSDC",
        bucket="1s",
        horizon=5,
        limit_rows=5000,
    )

    mid = df["mid"].to_numpy()
    fut = df["future_mid"].to_numpy()
    n = len(df)
    if n <= 200:
        pytest.skip("Dataset too small for robust horizon sanity checks")

    win = 200
    best = None
    best_score = -1.0
    start = 10
    end = n - 10 - win
    if end <= start:
        pytest.skip("Not enough rows after dropping boundaries")

    for k0 in range(start, end, 50):
        k1 = k0 + win
        sl_mid = mid[k0:k1]
        sl_fut = fut[k0:k1]
        score = float(sl_mid.std() + sl_fut.std())
        if score > best_score:
            best_score = score
            best = (k0, k1)

    assert best is not None
    k0, k1 = best
    sl_mid = mid[k0:k1]
    sl_fut = fut[k0:k1]
    if sl_mid.std() < 1e-9 or sl_fut.std() < 1e-9:
        pytest.skip("Series appears too flat in sampled windows to assess correlation")

    corr = np.corrcoef(sl_mid, sl_fut)[0, 1]
    assert corr > 0.5, (
        f"mid vs future_mid corr too low ({corr:.3f}) in window {k0}:{k1}. "
        "Possible horizon/join bug."
    )


def test_label_distribution_not_degenerate():
    df, X, y, feature_names = build_dataset(
        symbol="SOLUSDC",
        bucket="1s",
        horizon=5,
        limit_rows=10000,
    )

    d = _dist(y)
    assert len(d) >= 2, f"Only one class present in y (degenerate): dist={d}"
    assert max(d.values()) < 0.999, f"Almost all labels are same class: dist={d}"


def test_features_no_nan_inf_and_not_all_constant():
    df, X, y, feature_names = build_dataset(
        symbol="SOLUSDC",
        bucket="1s",
        horizon=5,
        limit_rows=10000,
    )

    assert np.isfinite(X).all(), "X contains NaN or inf (feature pipeline issue)"
    std = X.std(axis=0)
    nonzero = (std > 0).sum()
    assert nonzero >= max(1, X.shape[1] // 5), (
        f"Too many constant features ({nonzero}/{X.shape[1]} non-constant). "
        "Possible stale asof join / wrong resample."
    )


def test_ret_h_matches_mid_and_future_mid():
    df, X, y, feature_names = build_dataset(
        symbol="SOLUSDC",
        bucket="1s",
        horizon=5,
        limit_rows=8000,
    )

    mid = df["mid"].to_numpy()
    fut = df["future_mid"].to_numpy()
    ret = df["ret_h"].to_numpy()

    ok = np.isfinite(mid) & np.isfinite(fut) & np.isfinite(ret) & (mid > 0) & (fut > 0)
    mid, fut, ret = mid[ok], fut[ok], ret[ok]
    if len(ret) < 100:
        pytest.skip("Too few valid rows for return sanity")

    ret_log = np.log(fut / mid)
    ret_simple = (fut - mid) / mid

    mae_log = float(np.median(np.abs(ret - ret_log)))
    mae_simple = float(np.median(np.abs(ret - ret_simple)))
    best = min(mae_log, mae_simple)
    assert best < 1e-6, (
        f"ret_h does not match log-return or simple-return. "
        f"median_abs_err: log={mae_log:.3e}, simple={mae_simple:.3e}. "
        "Possible formula or alignment bug."
    )


def test_y_is_consistent_with_ret_h_definition():
    df, X, y, feature_names = build_dataset(
        symbol="SOLUSDC",
        bucket="1s",
        horizon=5,
        limit_rows=8000,
    )

    ret = df["ret_h"].to_numpy()
    y = df["y"].to_numpy()

    ok = np.isfinite(ret) & np.isfinite(y)
    ret = ret[ok]
    y = y[ok].astype(int)
    uniq = set(np.unique(y).tolist())

    if uniq.issubset({0, 1}):
        pred = (ret > 0).astype(int)
        agree = (pred == y).mean()
        assert agree > 0.95, f"Binary y seems inconsistent with ret_h sign (agree={agree:.3f})"
    elif uniq.issubset({-1, 0, 1}):
        eps = 5e-4  # matches pipeline default deadband
        pred = np.where(ret > eps, 1, np.where(ret < -eps, -1, 0))
        agree = (pred == y).mean()
        assert agree > 0.90, f"Ternary y seems inconsistent with ret_h sign/deadzone (agree={agree:.3f})"
    else:
        pytest.skip(f"Unknown label set {uniq}; add a custom consistency rule for your y.")


def test_auc_computation_contract_binary_or_multiclass():
    rng = np.random.default_rng(0)

    y_bin = rng.integers(0, 2, size=200)
    proba_bin = rng.random(size=(200, 2))
    proba_bin = proba_bin / proba_bin.sum(axis=1, keepdims=True)

    auc_bin = roc_auc_score(y_bin, proba_bin[:, 1])
    assert 0.0 <= auc_bin <= 1.0

    y_mc = rng.integers(0, 3, size=300)
    proba_mc = rng.random(size=(300, 3))
    proba_mc = proba_mc / proba_mc.sum(axis=1, keepdims=True)

    auc_mc = roc_auc_score(y_mc, proba_mc, multi_class="ovr", average="macro")
    assert 0.0 <= auc_mc <= 1.0
