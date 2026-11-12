"""Dataset loading for SECOM.

Two on-disk layouts are supported and both are *inspected, not assumed*:

1. UCI layout: ``secom.data`` (whitespace separated, ``NaN`` for missing) and
   ``secom_labels.data`` (label + quoted timestamp per line).
2. CSV layout (e.g. a Kaggle download): any ``*.csv`` in the data directory.
   The target column is auto-detected from common names, or given explicitly.

Internally the target is always encoded as ``1 = FAIL`` and ``0 = PASS``. The
raw values and the rule used are recorded in ``target_info`` so the encoding
is never silent.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import config

UCI_DATA_FILE = "secom.data"
UCI_LABELS_FILE = "secom_labels.data"
UCI_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


class DatasetNotFoundError(FileNotFoundError):
    """No usable dataset files were found."""


class DatasetFormatError(ValueError):
    """Files exist but their content is not in an understood format."""


class AmbiguousTargetError(ValueError):
    """Target values are valid but which value means FAIL is not known."""

    def __init__(self, message: str, values: list[Any]):
        super().__init__(message)
        self.values = values


@dataclass
class LoadedDataset:
    X: pd.DataFrame
    y: pd.Series | None
    source: str
    target_info: dict[str, Any] = field(default_factory=dict)
    timestamps: pd.Series | None = None
    dropped_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_target(self) -> bool:
        return self.y is not None


# ----------------------------------------------------------------- target ---
def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def normalize_target(series: pd.Series, fail_value: Any = None) -> tuple[pd.Series, dict[str, Any]]:
    """Encode a raw target column as int (1 = FAIL, 0 = PASS).

    * ``{-1, 1}``  -> documented UCI SECOM encoding, ``1`` is FAIL.
    * ``{0, 1}``   -> ambiguous, ``fail_value`` must be given.
    * strings      -> values containing fail/defect/bad vs pass/good/ok.
    * ``fail_value`` explicitly given -> used as-is.
    """
    s = series.copy()
    n_missing = int(s.isna().sum())
    s = s.dropna()
    numeric = pd.to_numeric(s, errors="coerce")
    is_numeric = bool(numeric.notna().all())
    if not is_numeric:  # text labels: ignore case and surrounding whitespace ("Pass" == "pass ")
        s = s.astype(str).str.strip()
        distinct = s.str.lower().nunique()
        raw_values = sorted(s.str.lower().unique().tolist())
    else:
        distinct = numeric.nunique()
        raw_values = sorted(pd.unique(s).tolist(), key=lambda v: str(v))
    if distinct != 2:
        raise DatasetFormatError(
            f"Target column must have exactly 2 distinct values, found {distinct}: {raw_values[:10]}"
        )
    if fail_value is not None:
        if is_numeric:
            fv = pd.to_numeric(pd.Series([fail_value]), errors="coerce").iloc[0]
            mask = numeric == fv
        else:
            mask = s.astype(str) == str(fail_value)
        if not mask.any() or mask.all():
            raise DatasetFormatError(f"fail_value={fail_value!r} does not split the target values {raw_values}.")
        rule = f"explicit fail_value={fail_value!r}"
    elif is_numeric:
        vals = set(numeric.unique().tolist())
        if vals == {-1, 1}:
            mask = numeric == 1
            rule = "UCI SECOM encoding: 1 = FAIL, -1 = PASS"
        else:
            raise AmbiguousTargetError(
                f"Target values {raw_values} do not state which value means FAIL. "
                "Pass fail_value explicitly (--fail-value on the CLI).",
                raw_values,
            )
    else:
        low = s.astype(str).str.strip().str.lower()
        fail_tokens, pass_tokens = ("fail", "defect", "bad", "reject"), ("pass", "good", "ok", "accept")
        uniq = sorted(low.unique())
        fails = [u for u in uniq
                 if any(t in u for t in fail_tokens) and not any(t in u for t in pass_tokens)]
        if len(fails) != 1:
            raise AmbiguousTargetError(
                f"Cannot tell which of {raw_values} means FAIL. Pass fail_value explicitly.", raw_values
            )
        mask = low == fails[0]
        rule = f"text match: '{fails[0]}' = FAIL"

    y = mask.astype(int)
    y.name = "target"
    info = {
        "raw_values": [_py(v) for v in raw_values],
        "fail_value": _py(s[mask].iloc[0]),
        "rule": rule,
        "internal_encoding": "1 = FAIL, 0 = PASS",
        "dropped_missing_target_rows": n_missing,
    }
    return y, info


def detect_target_column(columns) -> str | None:
    """Return the first column whose normalised name looks like a label."""
    for col in columns:
        if str(col).strip().lower() in config.TARGET_NAME_CANDIDATES:
            return col
    return None


# -------------------------------------------------------------- UCI format ---
def _parse_uci_labels(path: Path) -> tuple[pd.Series, pd.Series]:
    labels, stamps = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                raise DatasetFormatError(f"{path.name} line {ln}: expected '<label> \"<timestamp>\"', got {line!r}")
            labels.append(parts[0])
            stamps.append(parts[1].strip().strip('"'))
    y_raw = pd.to_numeric(pd.Series(labels), errors="coerce")
    if y_raw.isna().any():
        raise DatasetFormatError(f"{path.name} contains non-numeric labels.")
    ts = pd.to_datetime(pd.Series(stamps), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return y_raw, ts


def load_uci(data_dir: Path) -> LoadedDataset:
    data_path, label_path = data_dir / UCI_DATA_FILE, data_dir / UCI_LABELS_FILE
    try:
        X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN", "nan"], engine="python")
    except Exception as exc:
        raise DatasetFormatError(f"{UCI_DATA_FILE} could not be parsed ({exc}).") from exc
    X.columns = [f"feature_{i:03d}" for i in range(1, X.shape[1] + 1)]
    X = X.apply(pd.to_numeric, errors="coerce")
    y_raw, ts = _parse_uci_labels(label_path)
    if len(y_raw) != len(X):
        raise DatasetFormatError(
            f"Feature rows ({len(X)}) and label rows ({len(y_raw)}) do not match; files are misaligned."
        )
    y, info = normalize_target(y_raw)
    return LoadedDataset(
        X=X, y=y, source=f"UCI files in {data_dir.name}/ ({UCI_DATA_FILE}, {UCI_LABELS_FILE})",
        target_info=info, timestamps=ts if ts.notna().any() else None,
    )


# -------------------------------------------------------------- CSV format ---
def build_dataset(
    df: pd.DataFrame, source: str, target_col: str | None = None, fail_value: Any = None,
    auto_target: bool = True,
) -> LoadedDataset:
    """Split a raw dataframe into features / target / timestamps.

    Identifier and timestamp columns are separated out; non-numeric feature
    columns are coerced (columns that become entirely NaN are dropped and reported).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    notes: list[str] = []
    if target_col is None and auto_target:
        target_col = detect_target_column(df.columns)
    if target_col is not None and target_col not in df.columns:
        raise DatasetFormatError(f"Target column {target_col!r} not found. Columns: {list(df.columns)[:15]}...")

    y, info = None, {}
    if target_col is not None:
        y_raw = df.pop(target_col)
        keep = y_raw.notna()
        if not keep.all():
            notes.append(f"Dropped {int((~keep).sum())} rows with a missing target.")
        y, info = normalize_target(y_raw[keep], fail_value)
        df = df.loc[keep]
        info["column"] = target_col

    ts = None
    dropped: list[str] = []
    for col in list(df.columns):
        low = col.lower()
        if low in config.ID_NAME_CANDIDATES:
            df = df.drop(columns=col)
            dropped.append(col)
        elif low in config.TIME_NAME_CANDIDATES:
            parsed = pd.to_datetime(df.pop(col), errors="coerce", dayfirst=True)
            if parsed.notna().any() and ts is None:
                ts = parsed
            dropped.append(col)

    X = df.apply(pd.to_numeric, errors="coerce")
    all_nan = [c for c in X.columns if X[c].isna().all() and df[c].notna().any()]
    if all_nan:
        dropped.extend(all_nan)
        X = X.drop(columns=all_nan)
        notes.append(f"Dropped {len(all_nan)} non-numeric columns.")
    if X.shape[1] == 0:
        raise DatasetFormatError("No numeric feature columns were found in the file.")
    return LoadedDataset(
        X=X.reset_index(drop=True), y=None if y is None else y.reset_index(drop=True), source=source,
        target_info=info, timestamps=None if ts is None else ts.reset_index(drop=True),
        dropped_columns=dropped, notes=notes,
    )


def read_csv_bytes(raw, name: str = "upload") -> pd.DataFrame:
    """Read an uploaded CSV (bytes or file-like) with friendly errors."""
    data = raw.getvalue() if hasattr(raw, "getvalue") else raw
    if not data or not data.strip():
        raise DatasetFormatError(f"{name} is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DatasetFormatError(f"{name} exceeds the {config.MAX_UPLOAD_MB} MB upload limit.")
    try:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    except Exception as exc:  # parser errors, binary files, encoding problems
        raise DatasetFormatError(f"{name} could not be parsed as CSV ({type(exc).__name__}: {exc}).") from exc
    if df.shape[0] == 0 or df.shape[1] < 2:
        raise DatasetFormatError(f"{name} must contain a header and at least one row with 2+ columns.")
    return df


# ---------------------------------------------------------------- discovery ---
def find_dataset_files(data_dir: Path) -> tuple[str, list[Path]]:
    """Return ('uci', [data, labels]) or ('csv', [csv files]); raise if none."""
    data_dir = Path(data_dir)
    if (data_dir / UCI_DATA_FILE).exists() and (data_dir / UCI_LABELS_FILE).exists():
        return "uci", [data_dir / UCI_DATA_FILE, data_dir / UCI_LABELS_FILE]
    csvs = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    if csvs:
        return "csv", csvs
    raise DatasetNotFoundError(
        f"No SECOM dataset found in '{data_dir}'. Place either secom.data + secom_labels.data "
        "or a CSV file there (see data/README.md)."
    )


def load_dataset(
    data_dir: Path | str | None = None, target_col: str | None = None, fail_value: Any = None,
    csv_name: str | None = None,
) -> LoadedDataset:
    """Load the local dataset (UCI files preferred, otherwise a labelled CSV)."""
    data_dir = Path(data_dir) if data_dir else config.get_data_dir()
    kind, files = find_dataset_files(data_dir)
    if kind == "uci":
        return load_uci(data_dir)
    candidates = [f for f in files if csv_name in (None, f.name)]
    if not candidates:
        raise DatasetNotFoundError(f"{csv_name} not found in {data_dir}.")
    last_err: Exception | None = None
    for path in sorted(candidates, key=lambda p: ("train" not in p.name.lower(), p.name)):
        try:
            ds = build_dataset(read_csv_bytes(path.read_bytes(), path.name), f"CSV {data_dir.name}/{path.name}",
                               target_col, fail_value)
        except (DatasetFormatError, AmbiguousTargetError) as exc:
            last_err = exc
            continue
        if ds.has_target:
            return ds
        last_err = DatasetFormatError(f"{path.name}: no target column detected (use --target-col).")
    assert last_err is not None
    raise last_err


def dataset_signature(data_dir: Path | str | None = None) -> tuple:
    """Cheap cache key: names + mtimes of candidate dataset files."""
    data_dir = Path(data_dir) if data_dir else config.get_data_dir()
    if not data_dir.exists():
        return ()
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(data_dir.iterdir())
                 if p.suffix.lower() == ".csv" or p.name in (UCI_DATA_FILE, UCI_LABELS_FILE))


def download_uci(data_dir: Path | str | None = None) -> Path:
    """Download and extract the UCI SECOM archive (needs internet access)."""
    import urllib.request
    import zipfile

    data_dir = Path(data_dir) if data_dir else config.get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
            payload = resp.read()
    except Exception as exc:
        raise ConnectionError(
            f"Could not download {UCI_URL} ({exc}). Download the files manually - see data/README.md."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for member in (UCI_DATA_FILE, UCI_LABELS_FILE, "secom.names"):
            if member in zf.namelist():
                (data_dir / member).write_bytes(zf.read(member))
    return data_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Dataset utilities")
    ap.add_argument("--download-uci", action="store_true", help="download SECOM from the UCI repository")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args()
    if a.download_uci:
        print("Saved to", download_uci(a.data_dir))
    else:
        d = load_dataset(a.data_dir)
        print(d.source, d.X.shape, d.target_info)
