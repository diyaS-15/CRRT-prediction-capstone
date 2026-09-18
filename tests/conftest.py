import numpy as np
import pandas as pd
import pytest


def make_raw_rows(n: int, seed: int = 0) -> pd.DataFrame:
    """Build a small raw dataframe matching the schema engineer_features
    expects (FEATURE_INPUT_COLS + TARGET_SOURCE_COLS + a group column)."""
    rng = np.random.default_rng(seed)
    injury = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    admission = injury + pd.to_timedelta(rng.integers(0, 48, n), unit="h")

    return pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "age": rng.integers(1, 90, n),
        "tbsa_2nd_3rd": rng.integers(1, 95, n),
        "inhalation_injury": rng.choice(["Yes", "No"], n),
        "injury_datetime": injury,
        "admission_datetime": admission,
        "total_crystalloid_ml_first_24h": rng.uniform(500, 8000, n),
        "total_colloid_ml_first_24h": rng.uniform(0, 500, n),
        "total_urine_output_ml_first_24h": rng.uniform(100, 3000, n),
        "admission_weight_kg": rng.uniform(40, 120, n),
        "carboxyhemoglobin": rng.uniform(0, 30, n),
        "initial_temp_c": rng.uniform(34, 39, n),
        "comorbidity": rng.choice(["Diabetes", "CKD", "Hypertension", "None"], n),
        "crrt_first_24h": rng.choice(["Yes", "No"], n, p=[0.1, 0.9]),
        "crrt_25_48h": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
    })


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_rows(60, seed=1)
