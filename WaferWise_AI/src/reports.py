"""Generate downloadable Markdown / HTML reports from saved artifacts and predictions."""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from . import config


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None


@dataclass
class Report:
    title: str
    sections: list[Section]

    def to_markdown(self) -> str:
        out = [f"# {self.title}", ""]
        for s in self.sections:
            out += [f"## {s.heading}", ""]
            for p in s.paragraphs:
                out += [p, ""]
            if s.table is not None and len(s.table):
                out += [_md_table(s.table), ""]
        return "\n".join(out)

    def to_html(self) -> str:
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(self.title)}</title>",
            "<style>body{font-family:Segoe UI,Roboto,Arial,sans-serif;max-width:900px;margin:2rem auto;"
            "padding:0 1rem;color:#0f172a;line-height:1.55}h1{color:#0a1628;border-bottom:3px solid #22d3ee;"
            "padding-bottom:.3rem}h2{color:#0a1628;margin-top:2rem}table{border-collapse:collapse;width:100%;"
            "font-size:.9rem}th{background:#0a1628;color:#fff;text-align:left}th,td{padding:6px 10px;"
            "border:1px solid #cbd5e1}tr:nth-child(even) td{background:#f1f5f9}</style></head><body>",
            f"<h1>{html.escape(self.title)}</h1>",
        ]
        for s in self.sections:
            parts.append(f"<h2>{html.escape(s.heading)}</h2>")
            parts += [f"<p>{html.escape(p)}</p>" for p in s.paragraphs]
            if s.table is not None and len(s.table):
                parts.append(_fmt(s.table).to_html(index=False, escape=True, border=0))
        parts.append("</body></html>")
        return "\n".join(parts)


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    return out


def _md_table(df: pd.DataFrame) -> str:
    df = _fmt(df).astype(str)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(v.replace("|", "\\|") for v in r) + " |" for r in df.to_numpy()]
    return "\n".join([head, sep, *rows])


def model_table(metrics: dict, which: str = "test_tuned") -> pd.DataFrame:
    """One row per model with FAIL-focused metrics (``which``: test_tuned | test_default | cv)."""
    rows = []
    for m in metrics["models"].values():
        if which == "cv":
            cv = m["cv"]
            rows.append({"Model": m["display_name"], "PR-AUC": cv["pr_auc_mean"], "ROC-AUC": cv["roc_auc_mean"],
                         "FAIL recall": cv["recall_mean"], "FAIL precision": cv["precision_mean"],
                         "FAIL F1": cv["f1_mean"], "Accuracy": cv["accuracy_mean"]})
        else:
            t = m[which]
            rows.append({"Model": m["display_name"], "PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"],
                         "FAIL recall": t["recall"], "FAIL precision": t["precision"], "FAIL F1": t["f1"],
                         "Accuracy": t["accuracy"], "Threshold": t["threshold"]})
    return pd.DataFrame(rows)


def build_report(artifacts, predictions: pd.DataFrame | None = None, pred_source: str | None = None,
                 pred_threshold: float | None = None) -> Report:
    md, mt = artifacts.metadata, artifacts.metrics
    ds, sp = md["dataset"], md["split"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [
        Section("Overview", [
            f"Generated {now} by WaferWise AI. Final model: {md['selected_model_display']}. "
            f"Training data: {md['data_source']}. Model trained {md['trained_at_utc']}.",
            "All figures are measured on the data named above. Predictions are model estimates, not actual factory "
            "inspection results.",
        ]),
        Section("Dataset summary", table=pd.DataFrame([
            ("Samples", ds["n_samples"]), ("Features", ds["n_features"]), ("FAIL samples", ds.get("n_fail")),
            ("FAIL rate (%)", round(ds.get("fail_pct", float("nan")), 3)),
            ("Missing cells (%)", round(ds["missing_cells_pct"], 3)), ("Duplicate rows", ds["duplicate_rows"]),
            ("Constant features", ds["constant_features"]),
            ("Train / test samples", f"{sp['n_train']} / {sp['n_test']}"),
            ("FAIL in train / test", f"{sp['fail_train']} / {sp['fail_test']}"),
        ], columns=["Item", "Value"]).astype(str)),
        Section("Target encoding", [
            f"Raw values {md['target_info'].get('raw_values')}; rule: {md['target_info'].get('rule')}; "
            f"internal encoding {md['target_info'].get('internal_encoding')}."]),
        Section("Held-out test results (decision threshold tuned on training out-of-fold predictions)",
                [f"Test set: {mt['test_set']['n']} samples, {mt['test_set']['n_fail']} FAIL. FAIL is the positive class."],
                model_table(mt, "test_tuned")),
        Section("Cross-validation results (training split)", [mt["selection_rule"]], model_table(mt, "cv")),
    ]
    if artifacts.importance is not None:
        imp = artifacts.importance.head(15)[["feature", "importance"]]
        sections.append(Section("Most influential features (global)", [
            "Importance is mean |SHAP| (or built-in importance if SHAP was unavailable). It reflects model "
            "behaviour, not proven physical causes."], imp))
    if predictions is not None and len(predictions):
        n_fail = int((predictions["prediction"] == config.FAIL_LABEL).sum())
        sections.append(Section("Prediction summary", [
            f"Source: {pred_source or 'uploaded data'}; threshold {pred_threshold if pred_threshold is not None else md['decision_threshold']:.2f}. "
            f"{len(predictions)} samples scored: {n_fail} predicted FAIL, {len(predictions) - n_fail} predicted PASS "
            f"({n_fail / len(predictions) * 100:.2f}% predicted defect rate)."],
            predictions.sort_values("fail_probability", ascending=False).head(10)))
    sections.append(Section("Limitations", [
        "The SECOM data is small, strongly imbalanced and anonymised; test-set metrics rest on few FAIL samples and "
        "are uncertain. The model has not been validated on a live production line, and thresholds should be set "
        "with the real cost of missed defects versus false alarms in mind."]))
    return Report("WaferWise AI — Model and Prediction Report", sections)
