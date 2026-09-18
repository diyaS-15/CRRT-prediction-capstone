# Serving image: the Streamlit CRRT risk prediction app.
#
# Build:  docker build -f docker/serve.Dockerfile -t crrt-serve .
# Run:    docker run --rm -p 8501:8501 \
#           -v $(pwd)/reports:/app/reports \
#           -v $(pwd)/mlflow.db:/app/mlflow.db \
#           crrt-serve
# Then open http://localhost:8501
FROM python:3.10-slim

WORKDIR /app

# libgomp1: OpenMP runtime required to unpickle the xgboost model pipeline
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/crrt/serving/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
