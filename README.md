# CRRT-prediction-capstone
Predicting Continuous Renal Replacement Therapy (CRRT) in Burn Patients Using Machine Learning. ASU Capstone in collaboration with Arizona Burn Center at Valleywise Medical Center, Creighton University School of Medicine

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # add ",dvc" too if you need to pull/push data & models
```

Or via `make install`. Every command below also has a `make` target — run `make` with no
arguments (or open the `Makefile`) to see the full list: `install`, `test`, `lint`, `train`,
`train-xgb`/`train-catboost`/`train-lightgbm`/`train-xgb-local`, `promote`, `serve`, `mlflow-ui`,
`dvc-pull`/`dvc-push`, `docker-build-train`/`docker-build-serve`, `docker-train`/`docker-serve`.

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

## CI

Every push/PR to `main` runs three jobs (`.github/workflows/tests.yml`):
- **lint** — `ruff check .`
- **test** — the pytest suite above
- **smoke-train** — generates a small throwaway dataset (`scripts/make_ci_fixture_data.py`, same
  schema as the real data, since the real dataset lives in a local-only DVC remote the CI runner
  can't reach) and runs the canonical XGBoost pipeline end to end with a reduced Optuna trial count
  (`CRRT_TUNING_N_TRIALS=3`) to prove the whole training path — not just imports — still executes.

## Docker

Two images, one for training and one for serving (`docker/train.Dockerfile`,
`docker/serve.Dockerfile`):

```bash
make docker-train   # builds crrt-train, runs it with data/, reports/, mlflow.db mounted in
make docker-serve   # builds crrt-serve, runs the Streamlit app on http://localhost:8501
```

Or without `make`:

```bash
docker build -f docker/train.Dockerfile -t crrt-train .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports \
  -v $(pwd)/mlflow.db:/app/mlflow.db crrt-train                         # defaults to train_xgb
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports \
  -v $(pwd)/mlflow.db:/app/mlflow.db crrt-train src.crrt.training.train_catboost

docker build -f docker/serve.Dockerfile -t crrt-serve .
docker run --rm -p 8501:8501 -v $(pwd)/reports:/app/reports \
  -v $(pwd)/mlflow.db:/app/mlflow.db crrt-serve
```

## Deploying the demo

Not yet deployed. To put a live link on a resume/portfolio via
[Streamlit Community Cloud](https://share.streamlit.io) (free):

1. Push this repo to GitHub (public, or a private repo Streamlit Cloud has access to).
2. On share.streamlit.io: New app → pick this repo/branch → main file path `src/crrt/serving/app.py`.
3. **The model has to actually be available at deploy time.** `reports/xgb_pipeline.joblib` and
   `mlflow.db` are both gitignored (DVC-tracked / local-only), so a fresh clone on Streamlit Cloud
   starts with neither. Two ways to handle this for a demo deployment:
   - Simplest: after training locally, `git add -f reports/xgb_pipeline.joblib` to commit that one
     file directly as a demo fallback (defeats the DVC point for this one file, but guarantees the
     app has something to load — `serving/app.py`'s local-file fallback path picks it up automatically).
   - More correct: point `MLFLOW_TRACKING_URI` (via Streamlit's "Secrets") at a real hosted MLflow
     tracking server with a model already promoted to Production, and swap the DVC remote
     (`.dvc/config`) from the local placeholder to real cloud storage (S3/GCS) so `dvc pull` can run
     as part of the deploy.
