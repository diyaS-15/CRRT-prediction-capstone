"""Generate a small synthetic CSV for CI smoke-testing the training scripts.

The real dataset (data/synthetic_data.csv) is DVC-tracked and lives in a
local-only remote, so a GitHub Actions runner can't pull it without cloud
storage credentials this project doesn't have configured. This script reuses
tests/conftest.py's fixture generator (same schema engineer_features
expects) to produce a same-shaped, throwaway dataset good enough to prove
the training pipeline runs end to end -- it is not meant to produce
meaningful model metrics.

Usage: python scripts/make_ci_fixture_data.py [out_path] [n_rows] [seed]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import make_raw_rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_data.csv"
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    make_raw_rows(n_rows, seed=seed).to_csv(out_path, index=False)
    print(f"Wrote {n_rows}-row CI fixture dataset to {out_path}")


if __name__ == "__main__":
    main()
