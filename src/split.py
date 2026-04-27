from __future__ import annotations
from dataclasses import dataclass, field 
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from preprocessing import FEATURE_COLS, TARGET_COL, RANDOM_SEED

@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    summary: dict=field(default_factory=dict) 

def make_patient_level_split(
    df: pd.DataFrame,
    group_col: str,
    val_size: float = 0.10,
    test_size: float = 0.20,
    seed: int = RANDOM_SEED
) -> SplitResult:

    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' not in dataframe columns")

    if not (0 < test_size < 1) or not (0 < val_size < 1) or (val_size + test_size) >= 1:
        raise ValueError("val_size and test_size must be between 0 and 1 and sum to < 1")
    # drop rows where target = NaN
    if TARGET_COL in df.columns: 
        df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
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

    # summary log to display split sizes + warn about imbalanced data 
    summary = build_summary(df, train_idx, val_idx, test_idx)
    warn_if_imbalanced(summary)

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

#  returns split (X_train, X_val, X_test, y_train, y_val, y_test
#  use with  from split import get_Xy (X_train, X_val, X_test, y_train, y_val, y_test = get_Xy(df, split) )
def get_Xy( df: pd.DataFrame, split: SplitResult,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series,pd.Series,pd.Series]:
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"warning FEATURE_COLS not in df and will be skipped: {missing}")
    X = df[available]
    y = df[TARGET_COL].astype(int)
    return ( 
        X.iloc[split.train_idx], X.iloc[split.val_idx], X.iloc[split.test_idx],
        y.iloc[split.train_idx], y.iloc[split.val_idx], y.iloc[split.test_idx],
    )

# summary for per-split rows, group counts, CRRT positive rates 
def build_summary(df: pd.DataFrame,train_idx: np.ndarray,val_idx: np.ndarray, test_idx: np.ndarray,) -> dict:
    summary = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subset = df.iloc[idx]
        pos_rate = subset[TARGET_COL].mean() if TARGET_COL in df.columns else None
        summary[name] = {
            "n_rows": len(idx),
            "pos_rate": round(pos_rate, 4) if pos_rate is not None else None,
        }
    return summary

# imbalance, warning if CRRT pos rates vary by more than 5, flags bad randomization
def warn_if_imbalanced(summary: dict, tol: float = 0.05) -> None:
    rates = {k: v["pos_rate"] for k, v in summary.items() if v["pos_rate"] is not None}
    if len(rates) < 2:
        return
    lo, hi = min(rates.values()), max(rates.values())
    if (hi - lo) > tol:
        worst = max(rates, key=rates.get)
        best  = min(rates, key=rates.get)
        print(
            f"warning CRRT positive rate differs by {hi - lo:.1%} across splits "
            f"({best}={rates[best]:.1%}, {worst}={rates[worst]:.1%}). "
        )

# main
if __name__ == "__main__":
    import sys
    from preprocessing import load_and_preprocess
 
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_data.csv"
    df   = load_and_preprocess(csv_path)
    split = make_patient_level_split(df, group_col="record_id")
 
    print("\n--- Split Summary --")
    for name, stats in split.summary.items():
        print(f"  {name:<6}  rows={stats['n_rows']:>5}  CRRT positive rate={stats['pos_rate']:.1%}")
 
    X_train, X_val, X_test, y_train, y_val, y_test = get_Xy(df, split)
    print(f"\n  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    print(f"  Features used : {list(X_train.columns)}")
    print(f"\n  scale_pos_weight for XGBoost: {(y_train == 0).sum() / (y_train == 1).sum():.2f}")