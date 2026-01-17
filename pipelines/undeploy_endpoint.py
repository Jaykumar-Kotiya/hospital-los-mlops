from google.cloud import aiplatform

#IMPORTANT: use YOUR correct values
PROJECT_ID = "hospital-los-mlops"
REGION = "us-central1"

# This is the endpoint that worked for prediction
ENDPOINT_ID = "4706735500113739776"

def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/53152273814/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )

    print(f"Undeploying all models from endpoint: {ENDPOINT_ID} ...")

    endpoint.undeploy_all()

    print("✅ Model undeployed successfully.")
    print("⚠️ Endpoint still exists but is no longer serving traffic (billing reduced).")

if __name__ == "__main__":
    main()
