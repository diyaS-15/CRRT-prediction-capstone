from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

def make_patient_level_split(
    df: pd.DataFrame,
    group_col: str,
    val_size: float = 0.10,
    test_size: float = 0.20,
    seed: int = 42,
) -> SplitResult:

    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' not in dataframe columns")

    if not (0 < test_size < 1) or not (0 < val_size < 1) or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be between 0 and 1 and sum to < 1")

    groups = df[group_col].astype(str).to_numpy()
    idx_all = np.arange(len(df))

    # 1) Split off test groups
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(idx_all, groups=groups))

    # 2) Split train vs val groups from remaining
    groups_trainval = groups[trainval_idx]
    idx_trainval = idx_all[trainval_idx]

    # val_size should be relative to remaining portion
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx_rel, val_idx_rel = next(gss2.split(idx_trainval, groups=groups_trainval))

    train_idx = idx_trainval[train_idx_rel]
    val_idx = idx_trainval[val_idx_rel]

    _assert_no_group_overlap(groups, train_idx, val_idx, test_idx)

    return SplitResult(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def _assert_no_group_overlap(groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_g = set(groups[train_idx])
    val_g = set(groups[val_idx])
    test_g = set(groups[test_idx])

    if train_g & val_g:
        raise RuntimeError("Leakage: train and val share groups")
    if train_g & test_g:
        raise RuntimeError("Leakage: train and test share groups")
    if val_g & test_g:
        raise RuntimeError("Leakage: val and test share groups")