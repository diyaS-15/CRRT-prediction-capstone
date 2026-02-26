import pandas as pd
from src.split import make_patient_level_split

def main():
    # IMPORTANT (NDA/HIPAA safety):
    # Do NOT hardcode your real dataset filename/path in this file before pushing to GitHub.
    # Keep the placeholder default, and set the real path locally using an environment variable:
    #   export BCQP_DATA_PATH="data/your_real_file.csv"   (Mac/Linux)
    # Then run:
    #   python main.py
    data_path = os.getenv("BCQP_DATA_PATH", "data/PLACE_YOUR_FILE_HERE.csv")
    df = pd.read_csv(data_path)

    group_col = "record_id"
    splits = make_patient_level_split(df, group_col=group_col, val_size=0.1, test_size=0.2, seed=42)

    train_df = df.iloc[splits.train_idx]
    val_df   = df.iloc[splits.val_idx]
    test_df  = df.iloc[splits.test_idx]

    print("Rows:", len(df))
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))
    print("Group col:", group_col)
    groups = df["record_id"].astype(str)
    print("Unique groups total:", groups.nunique())
    print("Unique groups train/val/test:",
      groups.iloc[splits.train_idx].nunique(),
      groups.iloc[splits.val_idx].nunique(),
      groups.iloc[splits.test_idx].nunique())
if __name__ == "__main__":
    main()