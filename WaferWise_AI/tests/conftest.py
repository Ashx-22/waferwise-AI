"""Shared fixtures.

IMPORTANT: the data generated here is SYNTHETIC and exists only to exercise the
code paths in tests. It is not SECOM and no result obtained from it is ever
reported as a project result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import run_training  # noqa: E402


def make_synthetic(n: int = 500, p: int = 60, fail_rate: float = 0.09, seed: int = 0):
    """Return (X, y) with y: 1 = FAIL. Includes NaNs, a constant, a correlated and a sparse column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (rng.random(n) < fail_rate).astype(int)
    X[y == 1, :4] += 1.5
    if p > 21:  # constant, correlated and sparse columns (only when wide enough)
        X[:, 10] = 3.0
        X[:, 11] = X[:, 0] * 2 + 0.01
    X[rng.random((n, p)) < 0.05] = np.nan
    if p > 21:
        X[:, 20] = np.where(rng.random(n) < 0.6, np.nan, X[:, 20])
    cols = [f"feature_{i:03d}" for i in range(1, p + 1)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


def write_uci_files(directory: Path, X: pd.DataFrame, y: pd.Series) -> None:
    X.to_csv(directory / "secom.data", sep=" ", header=False, index=False, na_rep="NaN")
    ts = pd.date_range("2008-07-19", periods=len(y), freq="h").strftime("%d/%m/%Y %H:%M:%S")
    with open(directory / "secom_labels.data", "w") as fh:
        for lab, t in zip(np.where(y == 1, 1, -1), ts):
            fh.write(f'{lab} "{t}"\n')


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic):
    """Quick end-to-end training on synthetic UCI-format files; returns (data_dir, models_dir)."""
    data_dir = tmp_path_factory.mktemp("data")
    models_dir = tmp_path_factory.mktemp("models")
    write_uci_files(data_dir, *synthetic)
    run_training(data_dir, models_dir, quick=True, n_jobs=1, shap_rows=100, verbose=False)
    return data_dir, models_dir
