"""Expand the small (n=200) synthetic dataset into a larger one, for testing
whether the model's noisy test-set recall is a sample-size artifact rather
than a modeling problem.

Real PHI data isn't available (see .gitignore's HIPAA note), so this can't
literally "get more real data." Instead, for each CRRT-outcome class
(positive/negative), every column is resampled INDEPENDENTLY with
replacement from that column's within-class empirical distribution, then
continuous columns are jittered. This preserves each feature's marginal
correlation with the target (what the model actually learns from, given the
feature engineering here has little cross-feature interaction) while
avoiding the leakage trap of whole-row bootstrapping: if you resample entire
rows and just jitter them slightly, near-duplicate copies of the same
original patient land in both train and test splits and inflate test
performance into looking artificially perfect. Sampling columns
independently means no synthetic row is a near-copy of any single real row.

Caveat: this does NOT preserve joint multi-column relationships beyond
sharing a class label (e.g. a synthetic row's age and tbsa_2nd_3rd are drawn
from different original patients), so any engineered feature that combines
several raw columns (e.g. revised_baux_score = age + tbsa + inhalation) will
have a more diluted target correlation than the original 200 rows. That's an
expected, disclosed tradeoff of this approach, not a bug.

Usage: python -m src.crrt.data.expand_synthetic_data [n_rows] [seed]
Writes to data/synthetic_data_expanded.csv (left out of the default training
path; pass it via BCQP_DATA_PATH to use it).
"""
import sys
import numpy as np
import pandas as pd

SRC_PATH = "data/synthetic_data.csv"
OUT_PATH = "data/synthetic_data_expanded.csv"

# columns to jitter after resampling, so rows aren't exact duplicates of the
# 200 originals; (relative_jitter_std, min_clip, max_clip)
# column groups that must stay linked to the SAME source row when resampling
# (drawing them independently could break invariants: an independently
# sampled admission_datetime could land before injury_datetime, and
# independently sampling crrt_first_24h/crrt_25_48h could produce a
# "positive" class row where both happen to land on "No")
LINKED_GROUPS = [
    ("injury_datetime", "admission_datetime"),
    ("crrt_first_24h", "crrt_25_48h"),
]

JITTER_NUMERIC = {
    "age": (0.05, 0, 120),
    "carboxyhemoglobin": (0.10, 0, 100),
    "initial_temp_c": (0.01, 25, 45),
    "tbsa_2nd_3rd": (0.08, 0, 100),
    "admission_weight_kg": (0.05, 1, 300),
    "total_crystalloid_ml_first_24h": (0.10, 0, None),
    "total_crystalloid_ml_25_48h": (0.10, 0, None),
    "total_colloid_ml_first_24h": (0.10, 0, None),
    "total_colloid_ml_25_48h": (0.10, 0, None),
    "total_urine_output_ml_first_24h": (0.10, 0, None),
    "total_urine_output_ml_25_48h": (0.10, 0, None),
}


def crrt_within_48h_flag(df: pd.DataFrame) -> pd.Series:
    def yn(v):
        return str(v).strip().lower() in {"yes", "y", "true", "1"}
    return (df["crrt_first_24h"].map(yn) | df["crrt_25_48h"].map(yn)).astype(int)


def jitter_series(vals: pd.Series, rel_std: float, lo, hi, rng: np.random.Generator) -> pd.Series:
    vals = pd.to_numeric(vals, errors="coerce")
    noise = rng.normal(0, rel_std, size=len(vals)) * vals.abs().clip(lower=1)
    vals = vals + noise
    if lo is not None:
        vals = vals.clip(lower=lo)
    if hi is not None:
        vals = vals.clip(upper=hi)
    return vals.round(1)


def sample_class(class_df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build n synthetic rows for one CRRT-outcome class by drawing each
    column independently (with replacement) from that column's within-class
    values, instead of bootstrapping whole rows. Columns in LINKED_GROUPS are
    drawn together from the same source row to preserve their relationship."""
    out = {}
    linked_cols = {c for group in LINKED_GROUPS for c in group}
    for col in class_df.columns:
        if col in linked_cols:
            continue
        idx = rng.integers(0, len(class_df), size=n)
        sampled = class_df[col].to_numpy()[idx]
        if col in JITTER_NUMERIC:
            rel_std, lo, hi = JITTER_NUMERIC[col]
            sampled = jitter_series(pd.Series(sampled), rel_std, lo, hi, rng).to_numpy()
        out[col] = sampled

    for group in LINKED_GROUPS:
        group = [c for c in group if c in class_df.columns]
        if not group:
            continue
        idx = rng.integers(0, len(class_df), size=n)
        for col in group:
            out[col] = class_df[col].to_numpy()[idx]

    return pd.DataFrame(out)


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = np.random.default_rng(seed)

    df = pd.read_csv(SRC_PATH)
    pos_mask = crrt_within_48h_flag(df) == 1
    pos_df, neg_df = df[pos_mask].reset_index(drop=True), df[~pos_mask].reset_index(drop=True)
    pos_rate = pos_mask.mean()
    print(f"Source: {len(df)} rows, positive rate {pos_rate:.1%}")

    n_pos = int(round(n_rows * pos_rate))
    n_neg = n_rows - n_pos

    sampled_pos = sample_class(pos_df, n_pos, rng)
    sampled_neg = sample_class(neg_df, n_neg, rng)
    expanded = pd.concat([sampled_pos, sampled_neg], ignore_index=True)

    expanded = expanded.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    expanded["record_id"] = np.arange(1, len(expanded) + 1)

    expanded.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(expanded)} rows ({n_pos} positive, {n_neg} negative) to {OUT_PATH}")


if __name__ == "__main__":
    main()
