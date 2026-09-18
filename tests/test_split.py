import numpy as np
import pandas as pd
import pytest

from src.crrt.data.split import get_Xy, make_patient_level_split
from src.crrt.features.preprocessing import FEATURE_COLS, TARGET_COL


def make_grouped_df(n_patients: int = 40, seed: int = 0) -> pd.DataFrame:
    """Multiple rows (visits) per patient, so a real leakage bug (a patient
    split across train/val/test) would actually be exercised."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient_id in range(n_patients):
        n_visits = rng.integers(1, 4)
        target = int(rng.random() < 0.2)
        for _ in range(n_visits):
            row = {"record_id": patient_id, TARGET_COL: target}
            for col in FEATURE_COLS:
                row[col] = rng.random()
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def grouped_df():
    return make_grouped_df()


def test_split_covers_every_row_exactly_once(grouped_df):
    split = make_patient_level_split(grouped_df, group_col="record_id", seed=0)
    all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert sorted(all_idx.tolist()) == list(range(len(grouped_df)))


def test_split_has_no_patient_overlap_across_sets(grouped_df):
    split = make_patient_level_split(grouped_df, group_col="record_id", seed=0)
    train_ids = set(grouped_df.iloc[split.train_idx]["record_id"])
    val_ids = set(grouped_df.iloc[split.val_idx]["record_id"])
    test_ids = set(grouped_df.iloc[split.test_idx]["record_id"])
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_split_raises_on_nan_target(grouped_df):
    bad = grouped_df.copy()
    bad.loc[0, TARGET_COL] = np.nan
    with pytest.raises(ValueError):
        make_patient_level_split(bad, group_col="record_id", seed=0)


def test_split_raises_on_missing_group_col(grouped_df):
    with pytest.raises(ValueError):
        make_patient_level_split(grouped_df, group_col="not_a_real_column", seed=0)


@pytest.mark.parametrize("val_size,test_size", [(0.5, 0.6), (-0.1, 0.2), (0.1, 1.5)])
def test_split_raises_on_invalid_sizes(grouped_df, val_size, test_size):
    with pytest.raises(ValueError):
        make_patient_level_split(grouped_df, group_col="record_id", val_size=val_size, test_size=test_size, seed=0)


def test_split_is_deterministic_given_same_seed(grouped_df):
    split_a = make_patient_level_split(grouped_df, group_col="record_id", seed=7)
    split_b = make_patient_level_split(grouped_df, group_col="record_id", seed=7)
    assert np.array_equal(split_a.train_idx, split_b.train_idx)
    assert np.array_equal(split_a.val_idx, split_b.val_idx)
    assert np.array_equal(split_a.test_idx, split_b.test_idx)


def test_get_xy_rows_align_with_split_indices(grouped_df):
    split = make_patient_level_split(grouped_df, group_col="record_id", seed=0)
    X_train, X_val, X_test, y_train, y_val, y_test = get_Xy(grouped_df, split)

    assert len(X_train) == len(split.train_idx)
    assert len(X_val) == len(split.val_idx)
    assert len(X_test) == len(split.test_idx)
    assert (y_train.to_numpy() == grouped_df.iloc[split.train_idx][TARGET_COL].to_numpy()).all()
    assert list(X_train.columns) == [c for c in FEATURE_COLS if c in grouped_df.columns]
