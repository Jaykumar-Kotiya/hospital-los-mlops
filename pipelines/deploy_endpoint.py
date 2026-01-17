import argparse
from google.cloud import aiplatform


def deploy_model(project: str, region: str, bucket: str) -> None:

    aiplatform.init(
        project=project,
        location=region,
    )

    artifact_uri = f"gs://{bucket}/model"

    serving_image = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"

    print(f"Using artifact_uri={artifact_uri}")
    print(f"Using serving_image={serving_image}")

    model = aiplatform.Model.upload(
        display_name="hospital-los-xgboost-model",
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_image,
    )

    print("Model uploaded.")
    print("Model resource name:", model.resource_name)

    endpoint = aiplatform.Endpoint.create(
        display_name="hospital-los-xgboost-endpoint",
    )

    print("Endpoint created.")
    print("Endpoint resource name:", endpoint.resource_name)

    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        traffic_split={"0": 100},
    )

    print("Deployment complete!")
    print("Deployed model:", model.resource_name)
    print("Endpoint:", endpoint.resource_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="Vertex AI region, e.g. us-central1")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs:// prefix)")
    args = parser.parse_args()

    deploy_model(
        project=args.project,
        region=args.region,
        bucket=args.bucket,
    )
