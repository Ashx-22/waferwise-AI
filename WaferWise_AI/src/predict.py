"""Loading trained artifacts, validating inputs and predicting PASS/FAIL."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config


class ArtifactsMissingError(FileNotFoundError):
    """Trained model files are not available (run ``python -m src.train``)."""


class InputValidationError(ValueError):
    """The supplied data cannot be used for prediction."""


@dataclass
class Artifacts:
    pipeline: object
    metadata: dict
    metrics: dict
    feature_stats: pd.DataFrame | None = None
    importance: pd.DataFrame | None = None
    shap_cache: dict | None = None
    models_dir: Path | None = None

    @property
    def expected_features(self) -> list[str]:
        return list(self.metadata["expected_features"])

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))


def artifacts_available(models_dir: Path | str | None = None) -> bool:
    d = Path(models_dir) if models_dir else config.get_models_dir()
    return all((d / f).exists() for f in (config.FINAL_PIPELINE, config.METADATA_JSON, config.METRICS_JSON))


def load_artifacts(models_dir: Path | str | None = None) -> Artifacts:
    """Load the saved pipeline and companion files; raise a helpful error if missing."""
    d = Path(models_dir) if models_dir else config.get_models_dir()
    if not artifacts_available(d):
        raise ArtifactsMissingError(
            f"Trained model files were not found in '{d}'. Add the dataset to data/ and run "
            "`python -m src.train` (see models/README.md)."
        )
    try:
        pipeline = joblib.load(d / config.FINAL_PIPELINE)
        metadata = json.loads((d / config.METADATA_JSON).read_text())
        metrics = json.loads((d / config.METRICS_JSON).read_text())
    except Exception as exc:
        raise ArtifactsMissingError(f"Model files in '{d}' are unreadable ({exc}). Re-run training.") from exc

    def _csv(name):
        p = d / name
        return pd.read_csv(p) if p.exists() else None

    cache = None
    if (d / config.SHAP_CACHE).exists():
        try:
            cache = joblib.load(d / config.SHAP_CACHE)
        except Exception:
            cache = None
    return Artifacts(pipeline, metadata, metrics, _csv(config.FEATURE_STATS_CSV),
                     _csv(config.FEATURE_IMPORTANCE_CSV), cache, d)


@dataclass
class ValidationResult:
    features: pd.DataFrame
    n_rows: int
    dropped_label_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    non_numeric_cells: int = 0
    ignored_id_columns: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def validate_features(df: pd.DataFrame, expected_features: list[str]) -> ValidationResult:
    """Check an uploaded dataframe against the model's expected input columns.

    * every expected feature column must be present (names must match exactly);
    * label-like columns (e.g. ``Pass/Fail``) are removed so they can never be
      used as inputs;
    * extra columns are ignored, non-numeric cells become missing values.
    """
    if df is None or df.empty:
        raise InputValidationError("The uploaded file contains no rows.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique().tolist()[:5]
        raise InputValidationError(f"Duplicate column names are not allowed: {dups}")

    msgs: list[str] = []
    label_cols = [c for c in df.columns if c.lower() in config.TARGET_NAME_CANDIDATES and c not in expected_features]
    id_cols = [c for c in df.columns if (c.lower() in config.ID_NAME_CANDIDATES
                                          or c.lower() in config.TIME_NAME_CANDIDATES)
               and c not in expected_features]
    if label_cols:
        df = df.drop(columns=label_cols)
        msgs.append(f"Removed label-like column(s) {label_cols} so they cannot influence predictions.")

    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        hint = ""
        if len(missing) == len(expected_features):
            hint = (" None of the expected column names were found - the file needs a header row with the "
                    "model's feature names (download the template on this page).")
        raise InputValidationError(
            f"{len(missing)} of {len(expected_features)} required feature columns are missing "
            f"(e.g. {missing[:5]}).{hint}"
        )

    extra = [c for c in df.columns if c not in expected_features and c not in id_cols]
    feats = df[expected_features]
    numeric = feats.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = int((numeric.isna() & feats.notna()).sum().sum())
    if bad:
        msgs.append(f"{bad} non-numeric cell(s) were treated as missing values.")
    if extra:
        msgs.append(f"Ignored {len(extra)} extra column(s) that the model does not use.")
    if numeric.isna().all(axis=1).any():
        raise InputValidationError("At least one row has no numeric values in any feature column.")
    return ValidationResult(numeric.reset_index(drop=True), len(numeric), label_cols, extra, bad, id_cols, msgs)


def predict_frame(pipeline, features: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Return fail_probability and PASS/FAIL prediction for each row."""
    proba = pipeline.predict_proba(features)[:, 1]
    return pd.DataFrame({
        "sample_id": np.arange(1, len(features) + 1),
        "fail_probability": np.round(proba, 6),
        "pass_probability": np.round(1 - proba, 6),
        "prediction": np.where(proba >= threshold, config.FAIL_LABEL, config.PASS_LABEL),
    })


def single_sample_frame(values: dict[str, float], expected_features: list[str]) -> pd.DataFrame:
    """One-row frame: given values filled in, everything else NaN (imputed by the pipeline)."""
    row = {c: np.nan for c in expected_features}
    for k, v in values.items():
        if k not in row:
            raise InputValidationError(f"Unknown feature '{k}'.")
        row[k] = float(v)
    return pd.DataFrame([row], columns=expected_features)


def template_csv(expected_features: list[str], medians: pd.Series | None = None) -> bytes:
    """CSV template: header = expected features, one example row of training medians."""
    row = [(float(medians[c]) if medians is not None and c in medians.index and pd.notna(medians[c]) else "")
           for c in expected_features]
    return pd.DataFrame([row], columns=expected_features).to_csv(index=False).encode()
