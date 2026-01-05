import pandas as pd
import pytest

from eda.io import load_table
from eda.quality import compare_splits, quality_report
from eda.config import EDAConfig


def test_load_table_csv(tmp_path):
    path = tmp_path / "sample.csv"
    df = pd.DataFrame({"ts_ms": [0, 1000], "a": [1.0, 2.0]})
    df.to_csv(path, index=False)
    out = load_table(path)
    assert "ts" in out.columns
    assert out["ts"].dt.tz is not None


def test_quality_detects_duplicates():
    cfg = EDAConfig()
    df = pd.DataFrame({"ts_ms": [0, 0], "a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        quality_report(df, cfg)


def test_compare_splits_overlap_detection():
    cfg = EDAConfig()
    train = pd.DataFrame({"ts_ms": [0, 1000], "a": [1, 2]})
    val = pd.DataFrame({"ts_ms": [900, 1500], "a": [3, 4]})  # overlaps with train
    test = pd.DataFrame({"ts_ms": [2500, 3000], "a": [5, 6]})
    with pytest.raises(ValueError):
        compare_splits(train, val, test, cfg)
