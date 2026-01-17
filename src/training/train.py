from pathlib import Path

import xgboost as xgb
from joblib import dump as joblib_dump

from src.training.utils import (
    get_project_root,
    load_feature_matrices,
    compute_regression_metrics,
    save_metrics,
)


def train_xgboost_regressor():
    
    # Load data
    X_train, X_val, y_train, y_val = load_feature_matrices()

    # Define model
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoostRegressor...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=20,
    )

    print("Training complete. Evaluating on validation set...")

    # Evaluate
    y_val_pred = model.predict(X_val)
    metrics = compute_regression_metrics(y_val, y_val_pred)

    print("Validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model + metrics
    project_root = get_project_root()
    models_dir = project_root / "models"
    artifacts_dir = project_root / "artifacts"

    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "xgboost_los_model.json"
    joblib_path = models_dir / "xgboost_los_model.joblib"
    metrics_path = artifacts_dir / "xgboost_los_metrics.json"

    # Save in XGBoost's native JSON format
    model.save_model(model_path)
    print(f"Saved XGBoost model (JSON) → {model_path}")

    # Save also as joblib (useful for sklearn-style loading)
    joblib_dump(model, joblib_path)
    print(f"Saved XGBoost model (joblib) → {joblib_path}")

    # Save metrics
    save_metrics(metrics, str(metrics_path))

    print("Training step complete.")
    return model, metrics


def main():
    train_xgboost_regressor()


if __name__ == "__main__":
    main()
