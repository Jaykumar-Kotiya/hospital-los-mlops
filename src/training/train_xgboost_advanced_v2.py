import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # hospital-los-mlops/
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
CLEAN_CSV_PATH = DATA_DIR / "hospital_los_clean.csv"  # optional, for patient split if available


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_data():
    print("Loading data...")
    if not X_PATH.exists() or not Y_PATH.exists():
        raise FileNotFoundError(
            f"Missing X/y files.\nExpected:\n  {X_PATH}\n  {Y_PATH}\n"
            "Run preprocessing first: python -m src.preprocessing.preprocess"
        )

    X = np.load(X_PATH, allow_pickle=False)
    y = np.load(Y_PATH, allow_pickle=False)

    if X.ndim != 2:
        raise ValueError(f"X should be 2D matrix, got shape={X.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def train_val_split(X, y, test_size=0.2, random_state=42):
    """
    If patient_nbr exists in hospital_los_clean.csv, do a grouped split to reduce leakage.
    Otherwise use normal random split.
    """
    if CLEAN_CSV_PATH.exists():
        df = pd.read_csv(CLEAN_CSV_PATH)
        if "patient_nbr" in df.columns and len(df) == len(y):
            print("Found patient_nbr in hospital_los_clean.csv — using grouped split by patient.")
            patient_ids = df["patient_nbr"].astype(str).values

            # Split unique patients
            unique_patients = np.unique(patient_ids)
            train_pat, val_pat = train_test_split(
                unique_patients, test_size=test_size, random_state=random_state
            )

            train_mask = np.isin(patient_ids, train_pat)
            val_mask = np.isin(patient_ids, val_pat)

            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            return X_train, X_val, y_train, y_val

        else:
            print("WARNING: patient_nbr not found in hospital_los_clean.csv. Using normal split.")
    else:
        print("WARNING: hospital_los_clean.csv not found. Using normal split.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


def main():
    safe_mkdir(MODELS_DIR)
    safe_mkdir(REPORTS_DIR)

    X, y = load_data()

    # Basic sanity for LOS target: ensure numeric and positive-ish
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1.0, None)

    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2, random_state=42)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # ---- Log target transform to reduce negative predictions and stabilize training ----
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    print("Setting up XGBoost + hyperparameter search (log target)...")

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    # Keep search small enough to run on a laptop but still meaningful
    param_distributions = {
        "n_estimators": [600, 900, 1200, 1500],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.65, 0.7, 0.8, 0.85, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 2, 5],
        "gamma": [0.0, 0.05, 0.1, 0.2],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    # NOTE: We fit search on log target. No early stopping here (your xgboost build doesn't support it via sklearn fit).
    search.fit(X_train, y_train_log)

    best_model = search.best_estimator_
    best_params = search.best_params_
    print("Best params:", best_params)

    # ---- Train final model using native XGBoost API (supports early stopping) ----
    print("Training final Booster with xgb.train + early stopping...")

    best_params_native = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(best_params["max_depth"]),
        "eta": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "min_child_weight": float(best_params["min_child_weight"]),
        "gamma": float(best_params["gamma"]),
        "reg_alpha": float(best_params.get("reg_alpha", 0.0)),
        "reg_lambda": float(best_params.get("reg_lambda", 1.0)),
    }

    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    # Use many rounds + early stopping to find a good stopping point
    bst = xgb.train(
        params=best_params_native,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    # Predict using best iteration
    y_val_pred_log = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))

    # Invert log transform
    y_val_pred = np.expm1(y_val_pred_log)
    y_val_true = np.expm1(y_val_log)

    # Clip to realistic range (tweak if you want)
    LOS_MIN, LOS_MAX = 1.0, 14.0
    y_val_pred_clip = np.clip(y_val_pred, LOS_MIN, LOS_MAX)
    y_val_true_clip = np.clip(y_val_true, LOS_MIN, LOS_MAX)

    # Metrics
    mae = mean_absolute_error(y_val_true_clip, y_val_pred_clip)
    rmse = np.sqrt(mean_squared_error(y_val_true_clip, y_val_pred_clip))

    print(f"Validation MAE:  {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Pred range (clipped): min={y_val_pred_clip.min():.2f}, max={y_val_pred_clip.max():.2f}, mean={y_val_pred_clip.mean():.2f}")

    ts = utc_timestamp()

    # Save sklearn best estimator (for reference / reproducibility)
    joblib_path = MODELS_DIR / f"xgboost_los_model_advanced_v2_{ts}.joblib"
    print(f"Saving best sklearn model to: {joblib_path}")
    joblib.dump(best_model, joblib_path)

    # Save booster for Vertex built-in XGBoost container
    booster_path = MODELS_DIR / "model.bst"
    print(f"Exporting booster to: {booster_path}")
    bst.save_model(str(booster_path))

    # Save report
    report = {
        "timestamp_utc": ts,
        "log_target": True,
        "clip_range_days": [LOS_MIN, LOS_MAX],
        "best_params_sklearn": best_params,
        "best_params_native": best_params_native,
        "best_iteration": int(bst.best_iteration),
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_val": int(X_val.shape[0]),
        "n_features": int(X.shape[1]),
        "artifacts": {
            "sklearn_joblib": str(joblib_path),
            "vertex_booster_bst": str(booster_path),
        },
    }

    report_path = REPORTS_DIR / f"training_metrics_advanced_v2_{ts}.json"
    print(f"Saving training summary to: {report_path}")
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Training finished ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
