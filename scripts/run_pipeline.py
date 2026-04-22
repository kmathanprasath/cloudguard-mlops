"""
Submit the CloudGuard pipeline to a running Kubeflow Pipelines instance.
Usage:
    python scripts/run_pipeline.py --host http://localhost:8080
"""

import argparse
import kfp

from pipeline.cloudguard_pipeline import cloudguard_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8080", help="KFP host URL")
    parser.add_argument("--experiment", default="cloudguard", help="KFP experiment name")
    parser.add_argument("--fused-csv", default="s3://your-bucket/data/fused_logs.csv")
    parser.add_argument("--s3-bucket", default="your-bucket")
    parser.add_argument("--s3-prefix", default="cloudguard/models/v1")
    parser.add_argument("--kserve-image", default="your-registry/cloudguard:latest")
    args = parser.parse_args()

    client = kfp.Client(host=args.host)

    run = client.create_run_from_pipeline_func(
        cloudguard_pipeline,
        arguments={
            "fused_csv_uri": args.fused_csv,
            "s3_bucket": args.s3_bucket,
            "s3_prefix": args.s3_prefix,
            "kserve_image": args.kserve_image,
        },
        experiment_name=args.experiment,
        run_name="cloudguard-training-run",
    )
    print(f"Pipeline run submitted: {run.run_id}")
    print(f"Track at: {args.host}/#/runs/details/{run.run_id}")


if __name__ == "__main__":
    main()
