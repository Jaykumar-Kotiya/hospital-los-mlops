# Utility functions for model training:
# Loading X, y matrices
# Train/validation split
# Regression metrics
# Saving metrics to JSON

from pathlib import Path
import json
from typing import Tuple, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_project_root() -> Path:

    return Path(__file__).resolve().parents[2]


def load_feature_matrices(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed"

    X_path = processed_dir / "X.npy"
    y_path = processed_dir / "y.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError("X.npy or y.npy not found. Run preprocessing first.")

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Loaded X: {X.shape}, y: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def compute_regression_metrics(
    y_true, y_pred
) -> Dict[str, float]:

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
   
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {path}")
