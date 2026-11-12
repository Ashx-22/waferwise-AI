# WaferWise AI

**Intelligent Semiconductor Defect Prediction and Explainable Manufacturing Quality Analysis Using Machine Learning**

B.Tech Electronics and Communication Engineering (Data Science) · academic project · Streamlit application

> **Status of results.** This repository contains the complete, tested code. It does **not** contain trained models or
> performance numbers, because the SECOM dataset could not be downloaded in the environment where the code was built.
> Every metric the app displays is read from files produced by *your* training run (`python -m src.train`).
> See [Verification status](#verification-status) for exactly what was and was not tested.

---

## Contents

1. [Project overview](#1-project-overview) · 2. [Abstract](#2-abstract) · 3. [Problem statement](#3-problem-statement) ·
4. [Objectives](#4-objectives) · 5. [Existing system](#5-existing-system) · 6. [Proposed system](#6-proposed-system) ·
7. [System architecture](#7-system-architecture) · 8. [Dataset description](#8-dataset-description) ·
9. [Data preprocessing](#9-data-preprocessing) · 10. [EDA methodology](#10-eda-methodology) ·
11. [ML algorithms](#11-ml-algorithms) · 12. [Model evaluation](#12-model-evaluation) ·
13. [Explainable AI](#13-explainable-ai) · 14. [Streamlit features](#14-streamlit-features) ·
15. [Installation](#15-installation) · 16. [Local execution](#16-local-execution) ·
17. [Kaggle dataset setup](#17-dataset-setup-kaggle-and-uci) · 18. [GitHub setup](#18-github-setup) ·
19. [Streamlit Community Cloud](#19-streamlit-community-cloud-deployment) · 20. [Docker](#20-docker-deployment) ·
21. [Hugging Face Spaces](#21-hugging-face-spaces-deployment-optional) · 22. [Screenshots](#22-screenshots) ·
23. [Limitations](#23-limitations) · 24. [Future enhancements](#24-future-enhancements) ·
25. [References](#25-references) · [Verification status](#verification-status) · [Project structure](#project-structure) · [Author](#author-and-contributions)

---

## 1. Project overview

WaferWise AI predicts whether a semiconductor manufacturing sample will **PASS** or **FAIL** in-line testing from
hundreds of process-sensor measurements, explains which measurements drive each prediction (SHAP), and exposes the
whole workflow in a multi-page Streamlit dashboard. Three models are trained and compared — Logistic Regression
(baseline), Random Forest and XGBoost — using metrics suited to a heavily imbalanced problem.

## 2. Abstract

Semiconductor fabrication generates hundreds of process signals per unit, but defects are rare and the informative
signals are buried among noisy, partly missing, redundant ones. WaferWise AI builds a leakage-free scikit-learn
pipeline (missing-value filtering, median imputation, constant/correlated feature removal, univariate selection,
class-weighted learning) and tunes three classifiers with stratified cross-validation optimising PR-AUC. The decision
threshold is chosen from out-of-fold training predictions, the final model is selected from cross-validation (not from
test results or accuracy), and predictions are explained with SHAP. A Streamlit application provides dataset
exploration, model comparison, validated single and batch prediction, explanations, manufacturing insights and
downloadable reports.

## 3. Problem statement

Given historical manufacturing measurements labelled PASS/FAIL, predict the label of new samples and identify the
measurements associated with failures — when failures are a small minority class, so that accuracy is a misleading
objective (a model that always predicts PASS scores high accuracy and detects nothing).

## 4. Objectives

- Build a reproducible, leakage-free data pipeline for noisy, high-dimensional, incomplete sensor data.
- Compare Logistic Regression, Random Forest and XGBoost using FAIL-class recall, precision, F1, PR-AUC (plus accuracy, ROC-AUC and confusion matrices).
- Handle class imbalance with class weights (optional SMOTE inside training folds only).
- Provide global and per-sample explanations and state clearly that they are not causal claims.
- Deliver a validated, deployable Streamlit app that never retrains at start-up.

## 5. Existing system

Conventional quality control relies on end-of-line inspection and univariate statistical process control (control
charts on individual signals). These approaches react after defects appear and struggle to capture interactions among
hundreds of correlated signals.

## 6. Proposed system

A supervised, multivariate model that scores samples from raw process measurements, reports a FAIL probability, and
shows the contributing features. It is a decision-support tool: it flags samples and candidate signals for engineers to
investigate; it does not replace inspection.

## 7. System architecture

```
 data/ (SECOM files)                                  Streamlit app (app.py + pages/)
        │                                                    ▲   reads only
        ▼                                                    │
  src/data_loader ─► src/train.py ───────────────► models/  (pipelines, metrics.json,
  (inspect, encode      │  stratified split                   metadata.json, SHAP cache)
   target 1=FAIL)       │  pipeline = missing filter → median impute → variance filter
                        │             → correlation filter → (scale) → top-k select → (SMOTE) → classifier
                        │  RandomizedSearchCV (stratified CV, PR-AUC)
                        │  threshold from out-of-fold predictions
                        │  select by CV PR-AUC, evaluate once on held-out test
                        └► src/explainability (SHAP) ─► shap_cache.joblib
 User CSV ─► src/predict.validate_features ─► pipeline.predict_proba ─► PASS/FAIL + probabilities
```

## 8. Dataset description

SECOM (McCann & Johnston, 2008) — measurements from a semiconductor manufacturing process with a pass/fail label per
production entity, hosted on the UCI Machine Learning Repository (CC BY 4.0) and used in a Kaggle challenge. The
UCI description lists roughly 1,567 examples and about 590 features with missing values and a strongly imbalanced label.
**The loader inspects the real files at run time and the app reports the actual statistics; no dataset statistics are
hard-coded anywhere in this project.** The raw data is not redistributed here — see [section 17](#17-dataset-setup-kaggle-and-uci).

## 9. Data preprocessing

All steps live inside one scikit-learn `Pipeline`, so they are fitted on the training portion only and re-fitted
inside every cross-validation fold (no leakage):

| Step | Class | Purpose |
|---|---|---|
| 1 | `MissingnessFilter` | numeric coercion, ±inf → NaN, drop columns with > 40 % missing (training fraction) |
| 2 | `SimpleImputer(median)` | impute remaining gaps |
| 3 | `VarianceThreshold` | drop constant columns |
| 4 | `CorrelationFilter` | drop one of each pair with \|r\| > 0.95 |
| 5 | `StandardScaler` | Logistic Regression only |
| 6 | `TopKSelector` | keep the top-k ANOVA-F features (k tuned: 20 / 40 / 80) |
| 7 | `SMOTE` (optional `--smote`) | imblearn pipeline, training folds only |

Target encoding is detected, not assumed: `{-1, 1}` is the documented UCI encoding (1 = FAIL); 0/1 targets require an
explicit `--fail-value`; text labels are matched on fail/pass tokens. Feature and label row counts are verified to
match. Duplicates, constant columns and missingness are measured and saved to `models/metadata.json`.

## 10. EDA methodology

The *Dataset Explorer* and *Exploratory Data Analysis* pages show dimensions, missing-value profiles (by column and
row), duplicates, class distribution and imbalance ratio, feature statistics, univariate F-score ranking against FAIL,
per-class distributions (histogram / box plot) and correlation heat-maps with highly-correlated pairs. EDA is
descriptive and never feeds back into model fitting.

## 11. ML algorithms

| Model | Role | Imbalance handling | Tuned (randomised search) |
|---|---|---|---|
| Logistic Regression | baseline | `class_weight="balanced"` | C, k |
| Random Forest | non-linear ensemble | `class_weight="balanced_subsample"` | trees, depth, leaf size, max features, k |
| XGBoost | gradient boosting | `scale_pos_weight` (searched) | trees, depth, learning rate, subsampling, min child weight, k |

Search: stratified K-fold (default 5), `n_iter` candidates per model (default 15), refit on **PR-AUC**, fixed seed 42.

## 12. Model evaluation

- **Split:** stratified 80/20; the test set is used only for final reporting.
- **Metrics (FAIL = positive class):** precision, recall, F1, PR-AUC, ROC-AUC, MCC, accuracy, balanced accuracy, confusion matrix. Accuracy is displayed but never used for selection.
- **Threshold:** chosen to maximise F1 (or F2 with `--threshold-metric f2`) on *out-of-fold training* predictions; results at the default 0.5 threshold are also reported.
- **Model selection:** highest mean cross-validated PR-AUC (ties → CV F1). Documented in `metrics.json`/the app.
- **Uncertainty:** the test set contains few FAIL samples, so 95 % bootstrap intervals for PR-AUC and ROC-AUC are reported alongside point estimates.

## 13. Explainable AI

SHAP `TreeExplainer` (Random Forest, XGBoost) or `LinearExplainer` (Logistic Regression) on the *selected* trained model,
computed on held-out test rows: global importance (mean |SHAP|), beeswarm summary plot, and per-sample contribution
bars for a test sample, your single-sample input, or a row of an uploaded batch. If SHAP fails for any reason the app
falls back to built-in model importances instead of breaking. **SHAP explains the model, not the factory: values show
learned association, not physical causation.**

## 14. Streamlit features

| Page | Function |
|---|---|
| Home (`app.py`) | intro, problem, objectives, methodology, key statistics read from the trained artifacts, setup status |
| 1 Dataset Explorer | local or uploaded CSV, label-column selection, preview, dimensions, missing values, class distribution, statistics |
| 2 Exploratory Data Analysis | data quality, feature relevance, distributions, correlations |
| 3 Model Comparison | measured CV and test metrics, confusion matrices, ROC / PR curves, bootstrap intervals, best parameters |
| 4 Defect Prediction | validated CSV batch prediction + template, single-sample form, probabilities, threshold slider, CSV download |
| 5 Explainable AI | SHAP importance, summary plot, individual explanations |
| 6 Manufacturing Insights | predicted FAIL/PASS counts and rate, risk ranking, feature-level comparison, trend (if timestamps) |
| 7 Downloadable Reports | Markdown / HTML report, predictions CSV, metrics CSV, metadata JSON |

Input validation: uploads must contain every expected feature column (names match, order irrelevant); label-like columns
(`Pass/Fail`, `target`, `label`…) are removed automatically so labels cannot enter the model; extra columns are ignored;
non-numeric cells become missing values; empty, unparsable, oversized (> 50 MB) or duplicate-column files get a clear error.
When data or models are missing every page shows setup instructions instead of crashing. Models are loaded once with
`st.cache_resource` and never retrained at start-up.

## 15. Installation

Python **3.12** is required (current numpy, scipy, XGBoost and SHAP releases no longer support 3.11).

**Linux / macOS**

```bash
git clone <your-repo-url> WaferWise_AI && cd WaferWise_AI
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone <your-repo-url> WaferWise_AI; cd WaferWise_AI
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(In `cmd.exe` activate with `.venv\Scripts\activate.bat`.) Add `-r requirements-dev.txt` instead if you want to run the tests.

## 16. Local execution

```bash
# 1. put the dataset in data/  (section 17)
# 2. train once (a few minutes; use --quick only for a smoke test)
python -m src.train
# 3. launch
streamlit run app.py
```

Training options: `--n-iter`, `--cv-folds`, `--smote`, `--threshold-metric f1|f2`, `--seed`, `--n-jobs`, `--shap-rows`,
`--models`, `--target-col`, `--fail-value` (see `python -m src.train --help`). Tests: `python -m pytest`.

## 17. Dataset setup (Kaggle and UCI)

Full instructions are in [`data/README.md`](data/README.md). In short:

- **UCI (no login):** download *SECOM* from https://archive.ics.uci.edu/dataset/179/secom and place `secom.data` and `secom_labels.data` in `data/` (or run `python -m src.data_loader --download-uci`).
- **Kaggle:** https://www.kaggle.com/competitions/secom-defect-detection-challenge/data — sign in, accept the rules, download, unzip the CSV(s) into `data/`. With the Kaggle CLI: `kaggle competitions download -c secom-defect-detection-challenge` using **your own** credentials in `~/.kaggle/kaggle.json`. No credentials are stored in this repo; `data/*` and `kaggle.json` are git-ignored.

The Kaggle file layout was not inspected while building this project; if the loader cannot interpret it, convert it to a single CSV with a label column or use the UCI files.

## 18. GitHub setup

```bash
git init
git add .
git status          # confirm data/ contains only README.md and no kaggle.json
git commit -m "WaferWise AI"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

`.gitignore` already excludes the raw dataset, virtual environments, caches and credentials.

## Model files and deployment

Cloud platforms build from your repository and cannot run training without the dataset, so for Streamlit Community
Cloud / Hugging Face you must commit the trained artifacts after training locally:

```bash
python -m src.train
git add -f models/*.joblib models/*.json models/*.csv
git commit -m "Add trained models" && git push
```

They are small. They derive from SECOM (CC BY 4.0) — keep the attribution below. `shap_cache.joblib` holds ≤ 200
preprocessed test rows; train with `--shap-rows 0` if you prefer not to publish any rows (the XAI page then shows
model importances only).

## 19. Streamlit Community Cloud deployment

1. Complete [section 18](#18-github-setup) including the committed model files.
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app → Deploy a public app from GitHub**.
3. Repository: your repo · Branch: `main` · **Main file path: `app.py`**.
4. **Advanced settings → Python version: 3.12.** Dependencies are read from `requirements.txt`.
5. Deploy. The `pages/` folder is picked up automatically as app pages; `.streamlit/config.toml` supplies the theme.

The raw dataset is not needed on the cloud: the app runs from `models/`. The Dataset Explorer/EDA pages work through CSV upload there.

## 20. Docker deployment

```bash
docker build -t waferwise-ai .
docker run --rm -p 8501:8501 waferwise-ai
# open http://localhost:8501
```

To use models trained on your machine without baking them in (or to train inside the container), mount the folders:

```bash
# Linux / macOS
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" waferwise-ai
# train inside the container
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" waferwise-ai python -m src.train
```

(PowerShell: use `${PWD}` instead of `$(pwd)`.) The image runs as a non-root user; if a mounted `models/` folder is not writable
by uid 1000, fix permissions before training. `.dockerignore` keeps the dataset, tests and credentials out of the image; models present at build time are included.

## 21. Hugging Face Spaces deployment (optional)

Use a **Docker** Space, which runs this repository's `Dockerfile` unchanged:

1. Create a new Space at https://huggingface.co/new-space, SDK = **Docker**.
2. Push this repository (including the trained `models/` files) to the Space's git remote.
3. In the Space's `README.md`, add the front-matter block below at the very top (do not add it to the GitHub README):

```yaml
---
title: WaferWise AI
emoji: 🔬
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 8501
---
```

Check the current Hugging Face Docker-Space documentation for any changes to required fields. This path has not been tested.

## 22. Screenshots

Placeholders — add your own captures to `assets/` after running the app:

| Page | File |
|---|---|
| Home | `assets/home.png` *(add)* |
| Model Comparison | `assets/model_comparison.png` *(add)* |
| Defect Prediction | `assets/prediction.png` *(add)* |
| Explainable AI | `assets/explainability.png` *(add)* |

## 23. Limitations

- SECOM is small, anonymised and strongly imbalanced; the test split holds few FAIL samples, so metrics are uncertain (hence bootstrap intervals). Cross-validation on the training split is the more stable guide.
- Feature names are anonymous (`feature_001`…); explanations cannot be mapped to physical parameters without the source documentation.
- The data may span a drifting process; a random split does not test robustness to drift.
- Univariate selection can miss features that matter only through interactions.
- Predictions are statistical estimates, not inspection results; SHAP values are associations, not causes.
- Uploaded files must use the training column names exactly; there is no automatic column mapping.
- The Kaggle CSV layout was not inspected (see section 17).

## 24. Future enhancements

Time-aware validation, probability calibration, cost-sensitive threshold selection, multivariate feature selection or
dimensionality reduction, drift monitoring, nested cross-validation for tighter estimates, column-mapping for uploads,
model registry/versioning, and a REST API.

## 25. References

- McCann, M., & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*, 785–794.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE, 10*(3), e0118432.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

*Citation details were written from memory; verify them before including them in a submitted report.*

---

## Verification status

Run on Linux, Python 3.12.3, with: streamlit 1.67.0, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, xgboost 3.4.1,
imbalanced-learn 0.14.2, shap 0.54.1, plotly 7.1.0, matplotlib 3.11.2.

| Item | Status |
|---|---|
| Automated tests (`python -m pytest`, 60 tests) in a fresh virtual environment installed from `requirements.txt` | **Passed** |
| `pip check` on that environment | **No broken requirements** |
| End-to-end training (LR, RF, XGBoost, SHAP, SMOTE option) | **Passed on synthetic data only** |
| Every Streamlit page loads, with and without data/models (Streamlit `AppTest`); single-prediction form; XAI/insights/report pages with session data | **Passed** (synthetic artifacts) |
| `streamlit run app.py` boots and answers its health endpoint | **Passed** |
| Invalid / empty / binary CSV, missing columns, label column in upload, missing dataset, missing models | **Covered by tests, passed** |
| **Real SECOM data** (inspection, training, metrics) | **Not done** — dataset unreachable from the build environment |
| Visual check of pages in a browser | **Not done** |
| Docker build/run, Streamlit Community Cloud, Hugging Face Spaces | **Not tested** |
| Windows, macOS, Python 3.11 | **Not tested** (3.11 is unsupported by current dependency releases) |

The synthetic data used in tests is generated inside `tests/conftest.py`; no figure derived from it is reported anywhere.

## Project structure

```
WaferWise_AI/
├── app.py                          # Home dashboard
├── pages/
│   ├── 1_Dataset_Explorer.py
│   ├── 2_Exploratory_Data_Analysis.py
│   ├── 3_Model_Comparison.py
│   ├── 4_Defect_Prediction.py
│   ├── 5_Explainable_AI.py
│   ├── 6_Manufacturing_Insights.py
│   └── 7_Downloadable_Reports.py
├── src/
│   ├── config.py                   # paths, constants (env-overridable dirs)
│   ├── data_loader.py              # UCI / CSV loading, target encoding, validation
│   ├── feature_engineering.py      # MissingnessFilter, CorrelationFilter, TopKSelector
│   ├── preprocessing.py            # pipeline builder, split, feature names
│   ├── eda.py                      # descriptive analysis helpers
│   ├── train.py                    # tuning, evaluation, model selection, saving (CLI)
│   ├── evaluate.py                 # metrics, thresholds, curves, bootstrap CIs
│   ├── predict.py                  # artifact loading, input validation, prediction
│   ├── explainability.py           # SHAP + fallback
│   ├── reports.py                  # Markdown / HTML reports
│   └── ui.py                       # theme and shared Streamlit helpers
├── tests/                          # 60 tests (synthetic data)
├── data/README.md                  # dataset instructions (data itself is git-ignored)
├── models/README.md                # generated artifacts description
├── assets/                         # screenshots go here
├── .streamlit/config.toml          # dark navy / cyan theme
├── Dockerfile  .dockerignore  .gitignore
├── requirements.txt  requirements-dev.txt  pytest.ini
└── README.md
```

## Author and contributions

*Author:* **[Your name]** — B.Tech ECE (Data Science), SRM Institute of Science and Technology, Tiruchirappalli
*Guide/Supervisor:* **[Name]** · *Contributions:* **[describe]** *(fill in before submission)*

Dataset: SECOM, UCI Machine Learning Repository, CC BY 4.0 — attribution required when redistributing derived models.
No software licence has been chosen for this code; add a `LICENSE` file before publishing publicly.
