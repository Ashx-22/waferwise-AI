# Data directory

The SECOM dataset is **not** included in this repository (and must never be committed).
Place the files here, then run `python -m src.train` from the project root.

## Option A - UCI files (recommended, no login)

Dataset: *SECOM*, McCann & Johnston (2008), UCI Machine Learning Repository, DOI 10.24432/C54305 (CC BY 4.0).

Page: https://archive.ics.uci.edu/dataset/179/secom

Place these two files directly in `data/`:

```
data/secom.data           # whitespace-separated features, NaN = missing
data/secom_labels.data    # label (-1 = pass, 1 = fail) + quoted timestamp per line
```

Automatic download (needs internet access): `python -m src.data_loader --download-uci`

## Option B - Kaggle CSV

Competition page: https://www.kaggle.com/competitions/secom-defect-detection-challenge/data

1. Sign in to Kaggle and accept the competition rules, then download the data from the page above, **or** use your own
   Kaggle API credentials (`~/.kaggle/kaggle.json`, never stored in this repo):
   `kaggle competitions download -c secom-defect-detection-challenge`
2. Unzip the CSV file(s) into `data/`.
3. Run `python -m src.train`. The loader auto-detects a label column named like `Pass/Fail`, `target`, `label`, `class`...
   If the labels are coded 0/1 the loader stops and asks which value means FAIL:

```
python -m src.train --target-col <column> --fail-value <raw value meaning FAIL>
```

The Kaggle file layout was not inspected when this loader was written, so if your files differ (e.g. a separate
labels file) convert them to one CSV with a label column, or use Option A.

## Target encoding

Internally the target is always `1 = FAIL`, `0 = PASS`. The raw values and the rule used are saved in
`models/metadata.json` and shown in the app.
