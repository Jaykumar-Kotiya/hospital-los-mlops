import argparse
from google.cloud import aiplatform


def run_training_job(project: str, region: str, bucket: str) -> None:

    staging_bucket = f"gs://{bucket}"

    aiplatform.init(
        project=project,
        location=region,
        staging_bucket=staging_bucket,
    )

    training_image = "us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-6:latest"

    serving_image = "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-6:latest"

    job = aiplatform.CustomTrainingJob(
        display_name="hospital-los-xgboost-training",
        script_path="trainer/task.py",          
        container_uri=training_image,         
        model_serving_container_image_uri=serving_image,  
    )

    job.run(
        args=[
            "--bucket",
            bucket,
        ],
        replica_count=1,
        machine_type="e2-medium",
        base_output_dir=f"gs://{bucket}/training_output",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", required=True, help="Vertex AI region, e.g. us-central1")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs:// prefix)")

    args = parser.parse_args()

    run_training_job(
        project=args.project,
        region=args.region,
        bucket=args.bucket,
    )
