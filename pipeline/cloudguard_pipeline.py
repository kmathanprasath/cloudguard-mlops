"""
CloudGuard Kubeflow Pipeline
Stages: data_prep → train → evaluate → push_model → deploy_kserve
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics
from typing import NamedTuple


# ── 1. Data Preparation ────────────────────────────────────────────────────────
@component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas==2.2.2", "numpy==2.0.2", "scikit-learn==1.6.1"],
)
def data_prep(
    fused_csv_uri: str,
    X_train_out: Output[Dataset],
    X_test_out: Output[Dataset],
    y_train_out: Output[Dataset],
    y_test_out: Output[Dataset],
    scaler_out: Output[Model],
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(fused_csv_uri)

    # Feature engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_offhours"] = (~df["hour"].between(8, 18)).astype(int)

    if "is_failed" not in df.columns:
        df["is_failed"] = 0
    if "is_rare_ip" not in df.columns:
        ip_counts = df["source_ip"].value_counts()
        df["is_rare_ip"] = df["source_ip"].map(ip_counts).lt(3).astype(int)

    df["action_count_1h"] = df.groupby("user").cumcount() + 1

    for col in ["user", "action", "source_type"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))

    FEATURE_COLS = [
        "hour", "day_of_week", "is_weekend", "is_offhours",
        "is_failed", "is_rare_ip", "user_enc", "action_enc",
        "source_type_enc", "action_count_1h",
    ]

    X = df[FEATURE_COLS].fillna(0).values
    y = df["label"].fillna(0).values.astype(int)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(X_train_out.path + ".npy", X_tr)
    np.save(X_test_out.path + ".npy", X_te)
    np.save(y_train_out.path + ".npy", y_tr)
    np.save(y_test_out.path + ".npy", y_te)
    joblib.dump(scaler, scaler_out.path + ".pkl")

    print(f"Data prep done — train: {X_tr.shape}, test: {X_te.shape}")


# ── 2. Train ───────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.12-slim",
    packages_to_install=["scikit-learn==1.6.1", "numpy==2.0.2", "joblib==1.5.3"],
)
def train(
    X_train_in: Input[Dataset],
    model_out: Output[Model],
    n_estimators: int = 200,
    contamination: float = 0.005,
):
    import numpy as np
    import joblib
    from sklearn.ensemble import IsolationForest

    X_train = np.load(X_train_in.path + ".npy")

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train)
    joblib.dump(clf, model_out.path + ".pkl")
    print(f"Isolation Forest trained on {X_train.shape[0]} samples")


# ── 3. Evaluate ────────────────────────────────────────────────────────────────
@component(
    base_image="python:3.12-slim",
    packages_to_install=["scikit-learn==1.6.1", "numpy==2.0.2", "joblib==1.5.3"],
)
def evaluate(
    model_in: Input[Model],
    X_test_in: Input[Dataset],
    y_test_in: Input[Dataset],
    metrics_out: Output[Metrics],
) -> NamedTuple("Outputs", [("auroc", float), ("f1", float)]):
    import numpy as np
    import joblib
    from sklearn.metrics import roc_auc_score, f1_score
    from collections import namedtuple

    model = joblib.load(model_in.path + ".pkl")
    X_test = np.load(X_test_in.path + ".npy")
    y_test = np.load(y_test_in.path + ".npy")

    scores = -model.score_samples(X_test)  # higher = more anomalous
    preds = (scores > np.percentile(scores, 97.5)).astype(int)

    auroc = float(roc_auc_score(y_test, scores))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    metrics_out.log_metric("auroc", auroc)
    metrics_out.log_metric("f1", f1)
    print(f"AUROC: {auroc:.4f}  F1: {f1:.4f}")

    Outputs = namedtuple("Outputs", ["auroc", "f1"])
    return Outputs(auroc=auroc, f1=f1)


# ── 4. Push model artifacts to object storage ──────────────────────────────────
@component(
    base_image="python:3.12-slim",
    packages_to_install=["boto3==1.34.0"],
)
def push_model(
    model_in: Input[Model],
    scaler_in: Input[Model],
    s3_bucket: str,
    s3_prefix: str,
    aws_region: str = "us-east-1",
):
    import boto3, os

    s3 = boto3.client("s3", region_name=aws_region)

    for local_suffix, s3_key in [
        (model_in.path + ".pkl", f"{s3_prefix}/iforest.pkl"),
        (scaler_in.path + ".pkl", f"{s3_prefix}/scaler.pkl"),
    ]:
        s3.upload_file(local_suffix, s3_bucket, s3_key)
        print(f"Uploaded {local_suffix} → s3://{s3_bucket}/{s3_key}")


# ── 5. Deploy to KServe ────────────────────────────────────────────────────────
@component(
    base_image="python:3.12-slim",
    packages_to_install=["kubernetes==29.0.0"],
)
def deploy_kserve(
    model_uri: str,
    namespace: str = "kubeflow-mlops",
    inference_service_name: str = "cloudguard",
    image: str = "mathanm5/cloudguard:v1",
    min_replicas: int = 1,
    max_replicas: int = 3,
):
    """Creates or patches a KServe InferenceService using a custom predictor."""
    from kubernetes import client, config

    config.load_incluster_config()

    custom_api = client.CustomObjectsApi()

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": inference_service_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": "Serverless",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "containers": [
                    {
                        "name": "cloudguard",
                        "image": image,
                        "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                        "env": [
                            {"name": "MODEL_PATH", "value": "/models/iforest.pkl"},
                            {"name": "SCALER_PATH", "value": "/models/scaler.pkl"},
                            {"name": "ANOMALY_THRESHOLD", "value": "0.5"},
                        ],
                        "volumeMounts": [
                            {"name": "model-store", "mountPath": "/models"}
                        ],
                        "resources": {
                            "requests": {"cpu": "500m", "memory": "512Mi"},
                            "limits": {"cpu": "2", "memory": "2Gi"},
                        },
                    }
                ],
                "volumes": [
                    {
                        "name": "model-store",
                        "persistentVolumeClaim": {"claimName": "cloudguard-model-pvc"},
                    }
                ],
            }
        },
    }

    group, version, plural = "serving.kserve.io", "v1beta1", "inferenceservices"
    try:
        custom_api.create_namespaced_custom_object(
            group=group, version=version, namespace=namespace,
            plural=plural, body=inference_service,
        )
        print(f"InferenceService '{inference_service_name}' created in '{namespace}'")
    except client.exceptions.ApiException as e:
        if e.status == 409:  # already exists — patch it
            custom_api.patch_namespaced_custom_object(
                group=group, version=version, namespace=namespace,
                plural=plural, name=inference_service_name, body=inference_service,
            )
            print(f"InferenceService '{inference_service_name}' patched")
        else:
            raise


# ── Pipeline definition ────────────────────────────────────────────────────────
@pipeline(
    name="cloudguard-training-pipeline",
    description="Train, evaluate, and deploy CloudGuard Isolation Forest to KServe",
)
def cloudguard_pipeline(
    fused_csv_uri: str = "s3://mathanm5mlops/data/fused_logs.csv",
    s3_bucket: str = "mathanm5mlops",
    s3_prefix: str = "cloudguard/models/v1",
    aws_region: str = "us-east-1",
    n_estimators: int = 200,
    contamination: float = 0.005,
    kserve_namespace: str = "kubeflow-mlops",
    kserve_image: str = "mathanm5/cloudguard:v1",
    min_auroc: float = 0.80,
):
    prep = data_prep(fused_csv_uri=fused_csv_uri)

    train_op = train(
        X_train_in=prep.outputs["X_train_out"],
        n_estimators=n_estimators,
        contamination=contamination,
    )

    eval_op = evaluate(
        model_in=train_op.outputs["model_out"],
        X_test_in=prep.outputs["X_test_out"],
        y_test_in=prep.outputs["y_test_out"],
    )

    # Only push & deploy when AUROC meets the threshold
    with dsl.Condition(eval_op.outputs["auroc"] >= min_auroc, name="quality-gate"):
        push_op = push_model(
            model_in=train_op.outputs["model_out"],
            scaler_in=prep.outputs["scaler_out"],
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            aws_region=aws_region,
        )

        deploy_kserve(
            model_uri=f"s3://{s3_bucket}/{s3_prefix}",
            namespace=kserve_namespace,
            image=kserve_image,
        ).after(push_op)


# ── Compile ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=cloudguard_pipeline,
        package_path="cloudguard_pipeline.yaml",
    )
    print("Pipeline compiled → cloudguard_pipeline.yaml")
