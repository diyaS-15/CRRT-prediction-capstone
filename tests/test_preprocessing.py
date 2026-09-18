import numpy as np
import pytest

from src.crrt.features.preprocessing import (
    FEATURE_COLS,
    FEATURE_INPUT_COLS,
    TARGET_COL,
    TARGET_SOURCE_COLS,
    engineer_features,
)


def test_engineer_features_raises_on_missing_required_columns(raw_df):
    incomplete = raw_df.drop(columns=["age"])
    with pytest.raises(ValueError):
        engineer_features(incomplete)


def test_engineer_features_produces_all_feature_cols(raw_df):
    out = engineer_features(raw_df)
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    assert not missing, f"engineer_features did not produce: {missing}"


def test_engineer_features_computes_target_when_source_cols_present(raw_df):
    out = engineer_features(raw_df)
    assert TARGET_COL in out.columns

    def yn(v):
        return str(v).strip().lower() in {"yes", "y", "true", "1"}

    expected = (raw_df["crrt_first_24h"].map(yn) | raw_df["crrt_25_48h"].map(yn)).astype(int)
    assert (out[TARGET_COL].astype(int) == expected).all()


def test_engineer_features_skips_target_when_source_cols_absent(raw_df):
    """Serving/inference rows won't have an outcome yet — engineer_features
    must not require or fabricate the target column in that case."""
    inference_row = raw_df.drop(columns=TARGET_SOURCE_COLS).iloc[[0]]
    out = engineer_features(inference_row)
    assert TARGET_COL not in out.columns
    # feature columns should still be fully computable without the target
    assert all(c in out.columns for c in FEATURE_COLS)


def test_inhalation_flag_matches_source_column(raw_df):
    out = engineer_features(raw_df)
    expected = (raw_df["inhalation_injury"].str.strip().str.lower() == "yes").astype(int)
    assert (out["inhalation_flag"] == expected).all()


def test_revised_baux_score_formula(raw_df):
    out = engineer_features(raw_df)
    expected = out["age"] + out["tbsa_2nd_3rd"] + (17 * out["inhalation_flag"])
    assert np.allclose(out["revised_baux_score"], expected)


def test_hours_injury_to_admission_matches_datetime_diff(raw_df):
    out = engineer_features(raw_df)
    expected_hours = (raw_df["admission_datetime"] - raw_df["injury_datetime"]).dt.total_seconds() / 3600.0
    assert np.allclose(out["hours_injury_to_admission"], expected_hours)


def test_age_outlier_is_clipped(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "age"] = 999  # clinically impossible
    out = engineer_features(bad)
    assert out.loc[0, "age"] <= 120


def test_fluid_balance_formula(raw_df):
    out = engineer_features(raw_df)
    expected = (
        raw_df["total_crystalloid_ml_first_24h"].fillna(0)
        + raw_df["total_colloid_ml_first_24h"].fillna(0)
        - raw_df["total_urine_output_ml_first_24h"].fillna(0)
    )
    assert np.allclose(out["fluid_balance_24h"], expected)


def test_feature_input_cols_do_not_include_target_sources():
    """FEATURE_INPUT_COLS must stay separate from TARGET_SOURCE_COLS so a
    caller can run engineer_features on a single patient row that has no
    outcome yet (see test above) without a false 'required column' error."""
    assert not (set(FEATURE_INPUT_COLS) & set(TARGET_SOURCE_COLS))
