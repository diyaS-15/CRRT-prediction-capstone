"""Lightweight model promotion workflow on top of the MLflow Model Registry.

Every training run registers a new model version in stage "None" — training
never automatically becomes "what's live." This script is the one place that
decides that: it compares the newest unstaged version's test-set metric
against whatever is currently in "Production" and only promotes if it's
actually better (or if there's no Production model yet). The previous
Production version is archived, not deleted, so you can always roll back.

serving/app.py reads whatever is tagged "Production" — so "which model is
live" is answered by this registry, not by which .joblib happens to be in
reports/.

Usage:
    python -m src.crrt.training.promote_model --model-name crrt-xgb
    python -m src.crrt.training.promote_model --model-name crrt-xgb --metric test_pr_auc
    python -m src.crrt.training.promote_model --model-name crrt-xgb --force
"""
import argparse
import warnings

from mlflow.tracking import MlflowClient

# MLflow's "stages" concept (None/Staging/Production/Archived) is deprecated
# in favor of model aliases, but stages are still fully functional and are
# the term anyone familiar with MLOps will recognize immediately, so this
# script keeps using them deliberately rather than switching to aliases.
warnings.filterwarnings("ignore", message=".*transition_model_version_stage.*")

from .mlflow_utils import init_mlflow

DEFAULT_METRIC = "test_sensitivity"  # recall — see generate_report.py for why


def _metric_value(client: MlflowClient, run_id: str, metric: str):
    run = client.get_run(run_id)
    return run.data.metrics.get(metric)


def promote(model_name: str, metric: str = DEFAULT_METRIC, force: bool = False) -> None:
    init_mlflow()
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for registered model '{model_name}'")

    candidate = max(versions, key=lambda v: int(v.version))
    candidate_score = _metric_value(client, candidate.run_id, metric)
    if candidate_score is None:
        raise ValueError(
            f"Candidate version {candidate.version} has no metric '{metric}' logged on its run"
        )

    production = next((v for v in versions if v.current_stage == "Production"), None)

    print(f"Model: {model_name} | metric: {metric}")
    print(f"Candidate: version {candidate.version}, {metric}={candidate_score:.4f}")

    if production is None:
        print("No current Production version — promoting candidate by default.")
        should_promote = True
    else:
        production_score = _metric_value(client, production.run_id, metric)
        print(f"Production: version {production.version}, {metric}={production_score}")
        should_promote = force or (production_score is not None and candidate_score > production_score)

    if not should_promote:
        print(f"Candidate does not beat Production on '{metric}' — leaving Production as-is.")
        return

    if production is not None and production.version != candidate.version:
        client.transition_model_version_stage(
            name=model_name, version=production.version, stage="Archived",
        )
        print(f"Archived previous Production version {production.version}.")

    client.transition_model_version_stage(
        name=model_name, version=candidate.version, stage="Production",
    )
    print(f"Promoted version {candidate.version} to Production.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Registered model name, e.g. crrt-xgb")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help=f"Metric to compare (default: {DEFAULT_METRIC})")
    parser.add_argument("--force", action="store_true", help="Promote the newest version regardless of metric comparison")
    args = parser.parse_args()
    promote(args.model_name, metric=args.metric, force=args.force)


if __name__ == "__main__":
    main()
