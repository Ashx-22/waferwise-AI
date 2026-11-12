"""Load every Streamlit page with Streamlit's AppTest, with and without trained artifacts / data.

Synthetic artifacts are produced by the ``trained`` fixture; the 'empty' cases check that the
app degrades gracefully when the dataset and models are unavailable.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.predict import predict_frame, validate_features
from tests.conftest import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py"] + sorted(f"pages/{p.name}" for p in (ROOT / "pages").glob("*.py"))
DATA_PAGES = {"pages/1_Dataset_Explorer.py", "pages/2_Exploratory_Data_Analysis.py"}


def _run(path: str, session=None) -> AppTest:
    at = AppTest.from_file(str(ROOT / path), default_timeout=180)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    return at.run()


def test_expected_pages_exist():
    assert len(PAGES) == 8  # Home + 7 pages


@pytest.fixture
def empty_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WAFERWISE_DATA_DIR", str(tmp_path / "no_data"))
    monkeypatch.setenv("WAFERWISE_MODELS_DIR", str(tmp_path / "no_models"))


@pytest.fixture
def trained_env(trained, monkeypatch):
    monkeypatch.setenv("WAFERWISE_DATA_DIR", str(trained[0]))
    monkeypatch.setenv("WAFERWISE_MODELS_DIR", str(trained[1]))


@pytest.mark.parametrize("page", PAGES)
def test_page_without_dataset_or_models_shows_guidance(page, empty_env):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(el.value for group in (at.markdown, at.warning, at.info, at.caption) for el in group)
    assert ("not found" in text or "Setup needed" in text or "No dataset" in text or "Trained model files" in text)


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_with_trained_artifacts(page, trained_env):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_home_shows_dataset_statistics(trained_env, trained):
    at = _run("app.py")
    labels = [m.label for m in at.metric]
    assert "Samples" in labels and "Final model" in labels
    samples = next(m for m in at.metric if m.label == "Samples")
    assert samples.value.replace(",", "") == "500"


def test_single_prediction_form_and_explanation(trained_env):
    at = _run("pages/4_Defect_Prediction.py")
    assert not at.exception
    at.button[0].click().run()  # form submit button with default (median) inputs
    assert not at.exception, [e.value for e in at.exception]
    assert any("Predicted:" in m.value for m in at.markdown)
    assert at.session_state["ww_single_row"] is not None
    # explanation page picks up the single sample
    at2 = _run("pages/5_Explainable_AI.py", {"ww_single_row": at.session_state["ww_single_row"]})
    assert not at2.exception, [e.value for e in at2.exception]


def test_insights_reports_and_xai_with_batch_session(trained_env, trained):
    from src.predict import load_artifacts

    art = load_artifacts(trained[1])
    X_new, _ = make_synthetic(n=120, seed=77)
    feats = validate_features(X_new, art.expected_features).features
    res = predict_frame(art.pipeline, feats, art.threshold)
    session = {"ww_pred_results": res, "ww_pred_features": feats, "ww_pred_threshold": art.threshold,
               "ww_pred_source": "synthetic_test.csv"}
    for page in ("pages/6_Manufacturing_Insights.py", "pages/7_Downloadable_Reports.py",
                 "pages/5_Explainable_AI.py"):
        at = _run(page, session)
        assert not at.exception, (page, [e.value for e in at.exception])


def test_insights_can_score_loaded_dataset(trained_env):
    at = AppTest.from_file(str(ROOT / "pages/6_Manufacturing_Insights.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label == "Predicted defect rate" for m in at.metric)
