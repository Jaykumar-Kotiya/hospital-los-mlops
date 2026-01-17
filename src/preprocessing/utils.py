from pathlib import Path
import pandas as pd


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw hospital LOS CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    return pd.read_csv(path)


def save_clean_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned/preprocessed data as CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to: {path}")