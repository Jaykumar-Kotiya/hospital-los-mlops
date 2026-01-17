from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocessing.feature_config import (
    TARGET_COLUMN,
    BASE_FEATURES,
    DIAGNOSIS_FEATURES,
)
from src.preprocessing.diagnosis_mapping import add_grouped_diagnosis_columns
from src.preprocessing.utils import load_raw_data, save_clean_data


def build_clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df = df.dropna(subset=[TARGET_COLUMN])

    df = add_grouped_diagnosis_columns(df)

    grouped_diag_cols = [f"{col}_grouped" for col in DIAGNOSIS_FEATURES]
    all_features = BASE_FEATURES + grouped_diag_cols

    existing_cols = [c for c in all_features if c in df.columns]
    missing_cols = set(all_features) - set(existing_cols)
    if missing_cols:
        print(f"Warning: these expected columns were not found in raw data: {missing_cols}")

    cols_to_keep = existing_cols + [TARGET_COLUMN]
    df_clean = df[cols_to_keep].copy()

    print(f"Cleaned dataframe shape: {df_clean.shape}")
    return df_clean


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]  
    raw_path = project_root / "data" / "hospital_los_raw.csv"
    output_path = project_root / "data" / "processed" / "hospital_los_clean.csv"

    print(f"Loading raw data from: {raw_path}")
    df_raw = load_raw_data(str(raw_path))

    print("Cleaning + grouping diagnosis codes...")
    df_clean = build_clean_dataframe(df_raw)

    print("Creating feature matrices (X, y)...")
    build_feature_matrix(df_clean)

    print(f"Saving cleaned data to: {output_path}")
    save_clean_data(df_clean, str(output_path))

    print("Preprocessing step 1 complete.")


def build_feature_matrix(df_clean: pd.DataFrame):
    """Fit preprocessing pipeline and generate X, y matrices + feature names."""

    from src.preprocessing.encoders import (
        build_preprocessing_pipeline,
        fit_transform,
        save_pipeline,
        get_feature_names,
    )
    import numpy as np

    print("⚙️ Building preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline(df_clean)

    print("🏗 Fitting pipeline + transforming dataset...")
    X, y, fitted_pipeline = fit_transform(df_clean, pipeline)

    # Project output folder
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save matrices
    np.save(processed_dir / "X.npy", X)
    np.save(processed_dir / "y.npy", y)

    print(f"💾 Saved X matrix: {X.shape}")
    print(f"💾 Saved y vector: {y.shape}")

    # Save fitted pipeline
    save_pipeline(fitted_pipeline, processed_dir / "preprocessing_pipeline.pkl")

    # ✅ NEW: Save feature names
    feature_names = get_feature_names(fitted_pipeline)

    (processed_dir / "feature_names.txt").write_text("\n".join(feature_names))
    np.save(processed_dir / "feature_names.npy", np.array(feature_names, dtype=object))

    print(f"✅ Saved feature names: {len(feature_names)} columns")
    print(f"📄 feature_names.txt → {processed_dir / 'feature_names.txt'}")



if __name__ == "__main__":
    main()