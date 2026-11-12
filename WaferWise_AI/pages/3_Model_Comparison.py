"""Model Comparison - measured cross-validation and held-out test results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.ui import CYAN, get_artifacts, setup_page, show_df, show_fig, sidebar_footer, style_fig  # noqa: E402

setup_page("Model Comparison", "Measured results only — FAIL is the positive class; accuracy is shown but not used for selection.")
sidebar_footer()

art = get_artifacts()
mt, md = art.metrics, art.metadata
models = mt["models"]
names = {k: v["display_name"] for k, v in models.items()}
ts = mt["test_set"]

st.info(
    f"**Selected model: {names[mt['selected_model']]}.** {mt['selection_rule']}  \n"
    f"Held-out test set: **{ts['n']}** samples, **{ts['n_fail']}** FAIL ({ts['fail_prevalence'] * 100:.2f}%). "
    "With so few FAIL samples, test-set metrics are noisy — compare them with the cross-validation figures and "
    "the bootstrap intervals below."
)
if md["config"].get("quick"):
    st.warning("These artifacts came from a `--quick` smoke-test run (tiny hyper-parameter search). "
               "Re-run `python -m src.train` without `--quick` for reportable results.")

view = st.radio("Results view", ["Held-out test (tuned threshold)", "Held-out test (threshold 0.5)",
                                 "Cross-validation (training split)"], horizontal=True)


def _row(key: str, m: dict) -> dict:
    if view.startswith("Cross"):
        cv = m["cv"]
        return {"Model": names[key], "PR-AUC": cv["pr_auc_mean"], "ROC-AUC": cv["roc_auc_mean"],
                "FAIL recall": cv["recall_mean"], "FAIL precision": cv["precision_mean"], "FAIL F1": cv["f1_mean"],
                "Accuracy": cv["accuracy_mean"], "PR-AUC ±": cv["pr_auc_std"], "Threshold": 0.5}
    t = m["test_tuned"] if "tuned" in view else m["test_default"]
    return {"Model": names[key], "PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"], "FAIL recall": t["recall"],
            "FAIL precision": t["precision"], "FAIL F1": t["f1"], "Accuracy": t["accuracy"], "MCC": t["mcc"],
            "Threshold": t["threshold"]}


table = pd.DataFrame([_row(k, v) for k, v in models.items()])
if view.startswith("Cross"):
    st.caption("Cross-validated scores use the default 0.5 threshold for recall / precision / F1 / accuracy, "
               "and are averaged over the stratified folds of the training split.")
show_df(table.style.format({c: "{:.3f}" for c in table.columns if c != "Model"}))

focus = table.melt(id_vars="Model", value_vars=["PR-AUC", "FAIL recall", "FAIL precision", "FAIL F1"],
                   var_name="Metric", value_name="Score")
fig = px.bar(focus, x="Metric", y="Score", color="Model", barmode="group")
fig.update_yaxes(range=[0, 1])
show_fig(style_fig(fig, "FAIL-focused metrics by model"))

st.subheader("Confusion matrix")
sel = st.selectbox("Model", list(models), format_func=lambda k: names[k],
                   index=list(models).index(mt["selected_model"]))
t = models[sel]["test_tuned"] if "tuned" in view or view.startswith("Cross") else models[sel]["test_default"]
cm = t["confusion_matrix"]
fig = go.Figure(go.Heatmap(
    z=cm, x=["Predicted PASS", "Predicted FAIL"], y=["Actual PASS", "Actual FAIL"], text=cm, texttemplate="%{text}",
    colorscale=[[0, "#0f2038"], [1, CYAN]], showscale=False, textfont=dict(size=22)))
fig.update_yaxes(autorange="reversed")
c1, c2 = st.columns([1, 1])
with c1:
    show_fig(style_fig(fig, f"{names[sel]} — held-out test set (threshold {t['threshold']:.3f})", 340))
with c2:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    st.markdown(
        f"- **Caught defects (true FAIL):** {tp} of {tp + fn}\n"
        f"- **Missed defects (false PASS):** {fn}\n"
        f"- **False alarms (PASS flagged FAIL):** {fp}\n"
        f"- **Correct PASS:** {tn}\n\n"
        "In production a missed defect and a false alarm have different costs; the threshold trades one for the other."
    )

st.subheader("ROC and precision-recall curves")
c1, c2 = st.columns(2)
roc, pr = go.Figure(), go.Figure()
for k, m in models.items():
    roc.add_trace(go.Scatter(x=m["curves"]["roc"]["fpr"], y=m["curves"]["roc"]["tpr"], mode="lines", name=names[k]))
    pr.add_trace(go.Scatter(x=m["curves"]["pr"]["recall"], y=m["curves"]["pr"]["precision"], mode="lines",
                            name=names[k]))
roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#64748b"), name="Chance"))
pr.add_trace(go.Scatter(x=[0, 1], y=[ts["fail_prevalence"]] * 2, mode="lines",
                        line=dict(dash="dot", color="#64748b"), name="No-skill (prevalence)"))
roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate (recall)")
pr.update_layout(xaxis_title="Recall (FAIL)", yaxis_title="Precision (FAIL)")
with c1:
    show_fig(style_fig(roc, "ROC curve (held-out test)", 400))
with c2:
    show_fig(style_fig(pr, "Precision-recall curve (held-out test)", 400))
st.caption("Precision-recall is the more informative view for rare defects; the dotted line is the score of a "
           "classifier with no skill.")

st.subheader("Cross-validation detail and tuning")
cv_rows = []
for k, m in models.items():
    cv = m["cv"]
    cv_rows.append({"Model": names[k], **{f"{s.replace('_', ' ')} (mean ± sd)": f"{cv[f'{s}_mean']:.3f} ± {cv[f'{s}_std']:.3f}"
                                           for s in ("pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy")},
                    "Candidates searched": m["n_candidates_searched"], "Fit time (s)": m["fit_seconds"]})
show_df(pd.DataFrame(cv_rows))
st.caption(f"{md['config']['cv_folds']}-fold stratified CV; hyper-parameters tuned with randomised search "
           "optimising PR-AUC.")

ci_rows = [{"Model": names[k], "PR-AUC 95% CI": "{:.3f} – {:.3f}".format(*m["bootstrap_ci_95"]["pr_auc"]),
            "ROC-AUC 95% CI": "{:.3f} – {:.3f}".format(*m["bootstrap_ci_95"]["roc_auc"]),
            "Decision threshold": round(m["oof_threshold"], 3)} for k, m in models.items()]
st.markdown("**Bootstrap intervals on the held-out test set**")
show_df(pd.DataFrame(ci_rows))
with st.expander("Best hyper-parameters per model"):
    for k, m in models.items():
        st.markdown(f"**{names[k]}**")
        st.json(m["best_params"])
