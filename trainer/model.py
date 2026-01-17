import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_xgboost(X_train, y_train, X_val, y_val):
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    preds = model.predict(X_val)

    metrics = {
        "mae": float(mean_absolute_error(y_val, preds)),
        "mse": float(mean_squared_error(y_val, preds)),
        "rmse": float(mean_squared_error(y_val, preds, squared=False)),
        "r2": float(r2_score(y_val, preds)),
    }

    return model, metrics