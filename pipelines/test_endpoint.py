import numpy as np
from google.cloud import aiplatform

PROJECT_ID = "hospital-los-mlops"
REGION = "us-central1"

ENDPOINT_NAME = "projects/53152273814/locations/us-central1/endpoints/4706735500113739776"


MIN_LOS = 1.0
MAX_LOS = 14.0


def main():

    aiplatform.init(project=PROJECT_ID, location=REGION)

    endpoint = aiplatform.Endpoint(ENDPOINT_NAME)

    X = np.load("data/processed/X.npy")

    idx = X.shape[0] // 2
    instance = X[idx].tolist()

    print(f"Sending instance #{idx} with {len(instance)} features to endpoint...")

    prediction = endpoint.predict(instances=[instance])

    raw_pred = float(prediction.predictions[0])
    print(f"\nRaw model prediction: {raw_pred}")

    clipped = max(MIN_LOS, min(MAX_LOS, raw_pred))
    final_los = round(clipped, 1)

    print(f"Clipped to [{MIN_LOS}, {MAX_LOS}] days: {clipped}")
    print(f"\n Final LOS to show in app: {final_los} days")


if __name__ == "__main__":
    main()
