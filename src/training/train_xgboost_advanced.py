import os
import json
import joblib
import numpy as np

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    X = np.load(DATA_PROCESSED / "X.npy")
    y = np.load(DATA_PROCESSED / "y.npy")
    return X, y


def train_advanced_xgb():
    print("Loading data...")
    X, y = load_data()
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Setting up XGBoost + hyperparameter search...")
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    param_dist = {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [400, 800, 1200],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,              
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    search.fit(X_train, y_train)
    best_model: XGBRegressor = search.best_estimator_
    print("Best params:", search.best_params_)

    # Evaluate
    y_val_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = float(np.sqrt(mse))

    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")

    # Persist model
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_joblib_path = MODELS_DIR / f"xgboost_los_model_advanced_{timestamp}.joblib"
    booster_bst_path = MODELS_DIR / "model.bst"  

    print(f"Saving best model to: {model_joblib_path}")
    joblib.dump(best_model, model_joblib_path)

    print(f"Exporting booster to: {booster_bst_path}")
    booster = best_model.get_booster()
    booster.save_model(booster_bst_path)

    # Save metrics + params for documentation
    summary = {
        "timestamp_utc": timestamp,
        "best_params": search.best_params_,
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_val": int(X_val.shape[0]),
        "n_features": int(X.shape[1]),
    }

    metrics_path = REPORTS_DIR / f"training_metrics_{timestamp}.json"
    print(f"Saving training summary to: {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Training finished ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train_advanced_xgb()
