import numpy as np
import pandas as pd

from myproj.marketdata.features import compute_base_features, add_rolling_features
from myproj.marketdata.labels import add_return_labels, add_direction_labels, add_vol_labels
from myproj.marketdata.segment import add_segment_id


def make_small_df():
    data = {
        "rec_type": [0, 0, 0, 0, 0, 0],
        "ts_ms": [0, 1000, 2000, 3000, 4000, 5000],
        "bid_px": [100, 100, 101, 101, 102, 102],
        "ask_px": [101, 101, 102, 102, 103, 103],
        "bid_qty": [1, 1, 1, 1, 1, 1],
        "ask_qty": [1, 1, 1, 1, 1, 1],
        "vol_base": [0.5, 0.1, 0.2, 0.1, 0.3, 0.4],
    }
    return pd.DataFrame(data)


def test_features_and_labels():
    df = make_small_df()
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = add_segment_id(df, expected_dt_ms=1000, handle_bad_dt="segment")
    df = compute_base_features(df)
    df = add_rolling_features(df, [2])
    df = add_return_labels(df, [2])
    df = add_direction_labels(df, [2], deadzone="fixed_bps", fixed_bps=0.1)
    df = add_vol_labels(df, [2])

    # check mid and r1
    assert np.isclose(df.loc[0, "mid"], 100.5)
    assert np.isclose(df.loc[1, "r1"], np.log(df.loc[1, "mid"]) - np.log(df.loc[0, "mid"]))

    # direction labels in {-1,0,1}
    assert set(df["y_dir_2"].unique()).issubset({-1, 0, 1})

    # vol label non-negative
    finite = df["y_vol_2"].dropna()
    assert (finite >= 0).all()
