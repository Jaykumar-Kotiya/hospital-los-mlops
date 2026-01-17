from pathlib import Path
import joblib

def main():
    model_path = Path("models/xgboost_los_model.joblib")
    output_path = Path("models/model.bst")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    booster = model.get_booster()
    booster.save_model(output_path)

    print(f"Saved XGBoost booster to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
