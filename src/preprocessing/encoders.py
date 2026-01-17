# src/preprocessing/encoders.py

"""
Build + export the preprocessing pipeline AND feature names.

This module provides:
- build_preprocessing_pipeline()
- fit_transform()
- get_feature_names()
- save_pipeline()
- load_pipeline()
"""

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.preprocessing.feature_config import (
    NUMERIC_FEATURES,
    DEMOGRAPHIC_CATEGORICAL_FEATURES,
    ADMISSION_CATEGORICAL_FEATURES,
    DIAGNOSIS_FEATURES,
    MEDICATION_FEATURES,
    TARGET_COLUMN,
)


def build_preprocessing_pipeline(df_clean: pd.DataFrame) -> ColumnTransformer:
    grouped_diag_cols = [f"{col}_grouped" for col in DIAGNOSIS_FEATURES]

    onehot_features = (
        DEMOGRAPHIC_CATEGORICAL_FEATURES
        + ADMISSION_CATEGORICAL_FEATURES
        + grouped_diag_cols
    )

    medication_categories = ["No", "Steady", "Up", "Down"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat_onehot", OneHotEncoder(handle_unknown="ignore"), onehot_features),
            (
                "med_ordinal",
                OrdinalEncoder(categories=[medication_categories] * len(MEDICATION_FEATURES)),
                MEDICATION_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,  # cleaner names
    )

    return preprocessor


def fit_transform(df_clean: pd.DataFrame, pipeline: ColumnTransformer):
    X = pipeline.fit_transform(df_clean)
    y = df_clean[TARGET_COLUMN].values
    return X, y, pipeline


def get_feature_names(pipeline: ColumnTransformer) -> List[str]:
    """
    Return feature names after pipeline is fitted.
    Works with StandardScaler + OneHotEncoder + OrdinalEncoder.
    """

    # ColumnTransformer provides get_feature_names_out (sklearn >= 1.0)
    try:
        names = pipeline.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        pass

    # Fallback: build names manually if needed
    feature_names = []

    # Read fitted transformers
    for name, transformer, cols in pipeline.transformers_:
        if name == "remainder" and transformer == "drop":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            # OneHotEncoder typically supports this
            try:
                out = transformer.get_feature_names_out(cols)
                feature_names.extend([str(x) for x in out])
                continue
            except Exception:
                out = transformer.get_feature_names_out()
                feature_names.extend([str(x) for x in out])
                continue

        # StandardScaler / OrdinalEncoder: one output per input col
        if isinstance(cols, (list, tuple, np.ndarray)):
            feature_names.extend([str(c) for c in cols])
        else:
            feature_names.append(str(cols))

    return feature_names


def save_pipeline(pipeline: ColumnTransformer, output_path: str):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"🔥 Saved preprocessing pipeline → {path}")


def load_pipeline(path: str) -> ColumnTransformer:
    with open(path, "rb") as f:
        return pickle.load(f)
