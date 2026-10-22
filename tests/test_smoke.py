"""Smoke tests: the data/model layer imports and the preprocessing helpers behave.

Tests that need a dataset skip themselves when it is not on disk (run scripts/download_data.py first).
They never train the full model zoo, so the whole file runs in seconds.
"""
from pathlib import Path

import pandas as pd
import pytest

import core
import download_data

DATA = Path(__file__).resolve().parent.parent / "data"


def test_model_registries_are_complete():
    assert len(core.CLASSIFIERS) == 10
    assert len(core._regressors(["proto"], ["fwd_pkts_tot"])) == 10


def test_classifier_factories_build_estimators():
    for name, (make, _scaled) in core.CLASSIFIERS.items():
        assert hasattr(make(), "fit"), name


def test_ensure_data_reports_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "DATA", tmp_path)
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        core.ensure_data("nope.csv")


def test_clean_rt_iot_drops_index_constant_and_duplicates(tmp_path):
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"Unnamed: 0": [0, 1, 2], "bwd_URG_flag_count": [0, 0, 0],
                  "a": [1, 1, 2], "b": ["x", "x", "y"]}).to_csv(raw, index=False)
    out = tmp_path / "clean.csv"
    df = download_data.clean_rt_iot(raw, out)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2 and out.exists()


@pytest.mark.skipif(not (DATA / "Dry_Bean_Dataset.csv").exists(), reason="dataset not downloaded")
def test_load_beans_split_is_consistent():
    d = core.load_beans()
    assert len(d["classes"]) == 7
    assert len(d["X_tr"]) + len(d["X_te"]) == len(d["df"])
    assert "Area_Perimeter_Ratio" in d["X"].columns
    assert abs(len(d["X_te"]) / len(d["df"]) - 0.20) < 0.01


@pytest.mark.skipif(not (DATA / "RT_IOT2022_cleaned.csv").exists(), reason="dataset not prepared")
def test_load_flows_has_no_leaky_columns():
    d = core.load_flows()
    cols = [c.lower() for c in d["X_tr"].columns]
    assert not any("iat" in c or c.startswith(core.LEAK_PREFIX) for c in cols)
    assert "total_packets" in d["X_tr"].columns
    assert set(d["X_tr"].index).isdisjoint(d["X_te"].index)
