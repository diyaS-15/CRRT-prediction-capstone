# Training image: run any of the training scripts (Optuna search, MLflow
# logging/registration) in an environment that matches CI/prod exactly.
#
# Build:  docker build -f docker/train.Dockerfile -t crrt-train .
# Run:    docker run --rm \
#           -v $(pwd)/data:/app/data \
#           -v $(pwd)/reports:/app/reports \
#           -v $(pwd)/mlflow.db:/app/mlflow.db \
#           crrt-train                              # defaults to train_xgb
#         docker run --rm ... crrt-train src.crrt.training.train_catboost
FROM python:3.10-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by xgboost/lightgbm/catboost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

ENV BCQP_DATA_PATH=/app/data/synthetic_data.csv \
    MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

ENTRYPOINT ["python", "-m"]
CMD ["src.crrt.training.train_xgb"]
