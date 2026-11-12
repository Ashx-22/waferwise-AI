# Models directory

Trained artifacts are **generated**, not shipped. After placing the dataset in `data/`:

```
python -m src.train            # full tuning run
python -m src.train --quick    # tiny search, for smoke tests only - do not report its numbers
```

Files written here:

| File | Content |
|---|---|
| `final_pipeline.joblib` | Complete fitted pipeline (cleaning, imputation, filters, selection, classifier) of the selected model |
| `logistic_regression.joblib`, `random_forest.joblib`, `xgboost.joblib` | The tuned pipeline for each model |
| `metadata.json` | Dataset summary, target encoding, split, config, expected input columns, selected model, threshold, library versions |
| `metrics.json` | Cross-validation and held-out test metrics, curves, best hyper-parameters |
| `feature_stats.csv` | Per-feature training statistics (medians etc.) used by the prediction page |
| `feature_importance.csv`, `shap_cache.joblib` | Global importance and a small cached sample (<= 200 preprocessed test rows) for SHAP plots |

The app never retrains: it only loads these files. Without them it shows setup instructions.

**Licensing note:** the artifacts are derived from SECOM (CC BY 4.0). If you commit them (needed for Streamlit
Community Cloud / Hugging Face), keep the attribution in the README. `shap_cache.joblib` contains a small sample of
preprocessed rows; train with `--shap-rows 0` to omit it (the Explainable AI page then shows model importances only).
