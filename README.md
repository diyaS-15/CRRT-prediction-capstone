# CRRT-prediction-capstone
Predicting Continuous Renal Replacement Therapy (CRRT) in Burn Patients Using Machine Learning. ASU Capstone in collaboration with Arizona Burn Center at Valleywise Medical Center, Creighton University School of Medicine

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # add ",dvc" too if you need to pull/push data & models
```

## Data

`data/synthetic_data.csv` (and the trained model artifacts under `reports/`) are versioned with
[DVC](https://dvc.org/), not committed to git directly — only small `.dvc` pointer files are. Real
patient data must never be added to DVC or git (see `.gitignore`).

```bash
dvc pull    # fetch the tracked dataset + model artifacts
dvc push    # after `dvc add`-ing something new
```

The default remote (`.dvc/config`) points at a local directory as a stand-in — swap it for real
shared storage (S3/GCS/Azure) with `dvc remote add -d <name> <url>` before collaborating with others.

## Training

```bash
python -m src.crrt.training.train_xgb        # canonical model, Optuna search + MLflow tracking
python -m src.crrt.training.train_catboost
python -m src.crrt.training.train_lightgbm    # Optuna search + MLflow tracking
```

Each run logs params/metrics/artifacts to MLflow (`sqlite:///mlflow.db`, local by default — set
`MLFLOW_TRACKING_URI` to point at a real tracking server) and registers a new model version. Training
does **not** automatically put a model into production. Inspect runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Promoting a model to Production

A new model version starts in the registry with no stage — `serving/app.py` only ever loads whatever
is tagged **Production**, so "which model is live" has one explicit, auditable answer instead of
being whatever `.joblib` happens to be sitting in `reports/`.

```bash
python -m src.crrt.training.promote_model --model-name crrt-xgb
```

This compares the newest version's test recall against the current Production version and only
promotes it if it's actually better (or if there's no Production model yet); the previous Production
version is archived, not deleted, so you can roll back.

## Serving

```bash
streamlit run src/crrt/serving/app.py
```

Loads the Production-stage `crrt-xgb` model from the MLflow registry, falling back to
`reports/xgb_pipeline.joblib` if nothing has been promoted yet.

## Tests

```bash
pytest
```
