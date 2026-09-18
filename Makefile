.PHONY: install test lint train train-xgb train-catboost train-lightgbm train-xgb-local \
        promote serve mlflow-ui dvc-pull dvc-push \
        docker-build-train docker-build-serve docker-train docker-serve \
        clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check .

## Training (runs locally, using whatever .venv/interpreter is active)
train: train-xgb train-catboost train-lightgbm  ## all three production models

train-xgb:
	python -m src.crrt.training.train_xgb

train-catboost:
	python -m src.crrt.training.train_catboost

train-lightgbm:
	python -m src.crrt.training.train_lightgbm

train-xgb-local:  ## experimental leakage-sensitivity pipeline, see file header
	python -m src.crrt.training.train_xgb_local

## MODEL is the registered model name, e.g. `make promote MODEL=crrt-xgb`
MODEL ?= crrt-xgb
promote:
	python -m src.crrt.training.promote_model --model-name $(MODEL)

serve:
	streamlit run src/crrt/serving/app.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

## Data & model artifact versioning (see README's DVC section)
dvc-pull:
	dvc pull

dvc-push:
	dvc push

## Docker
docker-build-train:
	docker build -f docker/train.Dockerfile -t crrt-train .

docker-build-serve:
	docker build -f docker/serve.Dockerfile -t crrt-serve .

docker-train: docker-build-train
	docker run --rm \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/reports:/app/reports \
		-v $(CURDIR)/mlflow.db:/app/mlflow.db \
		crrt-train

docker-serve: docker-build-serve
	docker run --rm -p 8501:8501 \
		-v $(CURDIR)/reports:/app/reports \
		-v $(CURDIR)/mlflow.db:/app/mlflow.db \
		crrt-serve

clean:
	rm -rf catboost_info .pytest_cache *.egg-info
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
