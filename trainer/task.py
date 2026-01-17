import argparse
from trainer.utils import load_numpy_from_gcs
from trainer.model import train_xgboost
import joblib
import os
from google.cloud import storage

def upload_to_gcs(bucket_name, local_file, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(local_file)

def main(args):
    bucket = args.bucket

    print("Downloading data from GCS...")
    X = load_numpy_from_gcs(bucket, "data/X.npy", "/tmp/X.npy")
    y = load_numpy_from_gcs(bucket, "data/y.npy", "/tmp/y.npy")

    print("Splitting data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost...")
    model, metrics = train_xgboost(X_train, y_train, X_val, y_val)
    print("Validation metrics:", metrics)

    os.makedirs("/tmp/model", exist_ok=True)
    model_path = "/tmp/model/xgboost_los_model.joblib"
    joblib.dump(model, model_path)

    print("Uploading model to GCS...")
    upload_to_gcs(bucket, model_path, "models/xgboost_los_model.joblib")

    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, required=True)
    args = parser.parse_args()
    main(args)