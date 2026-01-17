import numpy as np
from google.cloud import aiplatform

PROJECT_ID = "hospital-los-mlops"
REGION = "us-central1"
ENDPOINT_ID = "4706735500113739776"  

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )
    print(f"Using endpoint: {endpoint_name}")

    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    print(f"Full dataset shape: X={X.shape}, y={y.shape}")

    n_samples = 512
    if n_samples > len(X):
        n_samples = len(X)

    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(X), size=n_samples, replace=False)

    X_batch = X[indices]
    y_true = y[indices]

    print(f"Sending batch of {n_samples} instances to the endpoint...")

    instances = X_batch.tolist()

    prediction = endpoint.predict(instances=instances)

    raw_preds = np.array(prediction.predictions, dtype=float).reshape(-1)

    print("\nRaw predictions (first 10):", raw_preds[:10])

    los_min, los_max = 1.0, 14.0
    clipped_preds = np.clip(raw_preds, los_min, los_max)

    print("Clipped predictions (first 10):", clipped_preds[:10])
    print("True LOS (first 10):          ", y_true[:10])

    mae = np.mean(np.abs(clipped_preds - y_true))
    mse = np.mean((clipped_preds - y_true) ** 2)
    rmse = np.sqrt(mse)

    print("\n=== Batch Evaluation on Endpoint Predictions ===")
    print(f"Number of samples: {n_samples}")
    print(f"MAE:  {mae:.4f} days")
    print(f"RMSE: {rmse:.4f} days")
    print(f"Min predicted LOS (clipped): {clipped_preds.min():.2f}")
    print(f"Max predicted LOS (clipped): {clipped_preds.max():.2f}")
    print(f"Mean predicted LOS (clipped): {clipped_preds.mean():.2f}")

    print("\nSample rows (true vs predicted, first 10):")
    for i in range(min(10, n_samples)):
        print(
            f"  Patient idx={indices[i]:>6} | true={y_true[i]:>2} | pred={clipped_preds[i]:5.2f}"
        )


if __name__ == "__main__":
    main()
