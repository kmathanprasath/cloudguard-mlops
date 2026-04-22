# CloudGuard

> An AI-powered cloud threat detection system with a full MLOps lifecycle — multi-source log ingestion, unsupervised anomaly detection, MITRE ATT&CK mapping, CVE enrichment, and a Claude-powered SOC analyst, all wired into a Kubeflow pipeline that trains, gates, and deploys to KServe automatically.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-Isolation%20Forest-purple)
![Pipeline](https://img.shields.io/badge/Orchestration-Kubeflow%20Pipelines%20v2-orange)
![Serving](https://img.shields.io/badge/Serving-KServe%20v0.13-blueviolet)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  AWS CloudTrail ──┐                                                 │
│  BETH (K8s)    ───┼──► Feature Engineering ──► Fused Log Dataset    │
│  Linux Auth     ──┘                                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KUBEFLOW PIPELINE (CI/CD for ML)                 │
│                                                                     │
│  data_prep ──► train ──► evaluate ──► [quality gate: AUROC≥0.80]    │
│                                              │                      │
│                                    ┌─────────┴──────────┐           │
│                                    ▼                    ▼           │
│                               push_model          (blocked)         │
│                                    │                                │
│                                    ▼                                │
│                             deploy_kserve                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE SERVICE (KServe)                       │
│                                                                     │
│  POST /predict ──► Isolation Forest ──► MITRE ATT&CK Mapping        │
│                                    └──► NVD CVE Enrichment          │
│                                    └──► Claude SOC Analyst          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MLOps Pipeline Deep Dive

The pipeline is defined in `pipeline/cloudguard_pipeline.py` using **Kubeflow Pipelines v2 SDK** and compiled to a YAML artifact. Each stage runs in an isolated container with pinned dependencies.

### Stage 1 — `data_prep`

- Reads the fused log CSV from S3 (`fused_csv_uri`)
- Engineers temporal features: `hour`, `day_of_week`, `is_weekend`, `is_offhours`
- Derives behavioral signals: `is_rare_ip` (IP seen < 3 times), `action_count_1h` (rolling user action count)
- Label-encodes categorical fields: `user`, `action`, `source_type`
- Fits a `MinMaxScaler` and serialises it alongside train/test splits as pipeline artifacts
- Outputs: `X_train`, `X_test`, `y_train`, `y_test`, `scaler` (all as KFP `Dataset`/`Model` artifacts)

### Stage 2 — `train`

- Trains an `IsolationForest` on the scaled training set
- Hyperparameters are pipeline-level inputs (default: `n_estimators=200`, `contamination=0.005`)
- Serialises the fitted model with `joblib` as a KFP `Model` artifact

### Stage 3 — `evaluate`

- Scores the test set using `score_samples` (negated → higher = more anomalous)
- Computes **AUROC** and **F1** at the 97.5th percentile threshold
- Logs both metrics to KFP's `Metrics` artifact (visible in the Kubeflow UI)
- Returns `auroc` and `f1` as named tuple outputs consumed by the quality gate

### Stage 4 — Quality Gate (`dsl.Condition`)

```python
with dsl.Condition(eval_op.outputs["auroc"] >= min_auroc, name="quality-gate"):
    push_op = push_model(...)
    deploy_kserve(...).after(push_op)
```

If AUROC falls below `min_auroc` (default `0.80`), the `push_model` and `deploy_kserve` stages are **skipped entirely** — the current production model stays live. This is the CD gate for ML.

### Stage 5 — `push_model`

- Uploads `iforest.pkl` and `scaler.pkl` to S3 under a versioned prefix (`s3://<bucket>/cloudguard/models/v1/`)
- Uses `boto3` with in-cluster IAM or injected AWS credentials

### Stage 6 — `deploy_kserve`

- Creates or patches a `KServe InferenceService` via the Kubernetes Python client (`load_incluster_config`)
- Configures serverless deployment mode with autoscaling (`minReplicas=1`, `maxReplicas=3`)
- Mounts the model PVC at `/models` so the FastAPI container can load artifacts at startup
- Handles `409 Conflict` gracefully — patches the existing resource instead of failing

---

## Inference Service

The FastAPI app (`app/main.py`) is the runtime serving layer.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness/readiness probe — returns model load status |
| `GET`  | `/`       | Service metadata |
| `POST` | `/predict` | Single event anomaly scoring + TTP mapping |
| `POST` | `/predict/batch` | Batch scoring for multiple events |

### Prediction Flow

```
LogEvent (JSON) ──► MinMaxScaler.transform ──► IsolationForest.score_samples
                                                        │
                                          normalise to [0, 1] anomaly score
                                                        │
                                          threshold check (default 0.5)
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    ▼                                       ▼
                             MITRE ATT&CK rules                      is_anomaly flag
                             (4 rule-based detectors)
```

### MITRE ATT&CK Detectors

| TTP | Name | Tactic | Trigger Condition |
|-----|------|--------|-------------------|
| T1110 | Brute Force | Credential Access | `is_failed=1` AND `action_count_1h > 20` |
| T1485 | Data Destruction | Impact | `"delete"` in action AND `is_offhours=1` |
| T1530 | Data from Cloud Storage | Collection | `"list"` in action AND `is_rare_ip=1` |
| T1078 | Valid Accounts | Defense Evasion | `is_offhours=1` AND `is_rare_ip=1` AND `is_failed=0` |

### Input Schema

```json
{
  "hour": 3,
  "day_of_week": 1,
  "is_weekend": 0,
  "is_offhours": 1,
  "is_failed": 0,
  "is_rare_ip": 1,
  "user_enc": 5,
  "action_enc": 12,
  "source_type_enc": 0,
  "action_count_1h": 3,
  "action": "DeleteBucket"
}
```

### Response Schema

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.8312,
  "threshold": 0.5,
  "ttp_detections": [
    {"ttp": "T1485", "name": "Data Destruction", "tactic": "Impact"}
  ],
  "message": "THREAT DETECTED"
}
```

---

## Containerisation

The `Dockerfile` builds a minimal inference image:

- Base: `python:3.12-slim` with only `gcc` as a system dep
- Copies `requirements-serve.txt` (inference-only deps — no training libraries)
- Model artifacts are **not baked into the image** — they are mounted at runtime via the PVC at `/models`
- Environment variables control model paths and threshold, making the image reusable across model versions

```
ENV MODEL_PATH=/models/iforest.pkl
ENV SCALER_PATH=/models/scaler.pkl
ENV ANOMALY_THRESHOLD=0.5
```

This separation of image and model artifacts is a core MLOps pattern — you can roll back a model by swapping the PVC contents without rebuilding the container.

---

## Kubernetes Manifests

### `k8s/model-pvc.yaml`

Provisions a 1Gi `PersistentVolumeClaim` in the `kubeflow-mlops` namespace. The Kubeflow pipeline writes model artifacts here (via S3 → PVC sync or direct mount), and the inference container reads from it at startup.

### `k8s/inference-service.yaml`

Defines the `KServe InferenceService` with:

- **Serverless deployment mode** — scales to zero when idle
- **Autoscaling**: 1–3 replicas based on request load
- **Resource requests/limits**: `500m CPU / 512Mi` → `2 CPU / 2Gi`
- **Readiness probe** on `GET /health` — KServe won't route traffic until the model is loaded
- **Volume mount** of `cloudguard-model-pvc` at `/models`

---

## Project Structure

```
CloudGuard.ipynb                # Experimentation notebook (training, EDA, threshold tuning)
requirements.txt                # Full notebook + training dependencies
Dockerfile                      # Inference service container image

app/
  main.py                       # FastAPI inference service (predict, batch, health)

pipeline/
  cloudguard_pipeline.py        # Kubeflow Pipeline v2: data_prep→train→evaluate→push→deploy

k8s/
  inference-service.yaml        # KServe InferenceService manifest
  model-pvc.yaml                # PersistentVolumeClaim for model artifacts

scripts/
  build_push.sh                 # Build & push Docker image to registry
  run_pipeline.py               # Submit compiled pipeline to Kubeflow
```

---

## Datasets

| Source | Dataset | Description |
|--------|---------|-------------|
| AWS CloudTrail | [flaws.cloud logs](http://summitroute.com/downloads/flaws_cloudtrail_logs.tar) | Real-world misconfigured AWS environment logs |
| K8s / Syscalls | [BETH Dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset) | Labelled Linux kernel syscall logs with `evil` column |
| Linux Auth | [LogHub Linux_2k.log](https://github.com/logpai/loghub/tree/master/Linux) | Real SSH authentication logs with brute-force patterns |

---

## Model Performance

| Model | AUROC | Notes |
|-------|-------|-------|
| Isolation Forest | **0.8935** | Production model — `n_estimators=200`, `contamination=0.005` |
| LSTM Autoencoder | ~0.49 | Experimental only, not used in the pipeline |

Threshold tuning is done in the notebook via a grid search over Precision / Recall / F1 to find the optimal operating point before the model is promoted.

---

## Notebook Sections

| Cells | What it does |
|-------|-------------|
| 0 | Install dependencies |
| 1–5 | Download datasets (CloudTrail, BETH, Linux auth) |
| 6–8 | Parse each source into unified schema |
| 9–11 | Fuse sources, engineer features, train/test split |
| 12–20 | Label fixing and data validation |
| 21–23 | Train Isolation Forest, evaluate, threshold tuning |
| 24–28 | LSTM Autoencoder (experimental) |
| 29 | Finalise models |
| 30 | NVD CVE enrichment function |
| 31 | MITRE ATT&CK TTP mapping rules |
| 32–34 | Claude SOC analyst setup and end-to-end test |

---

## Deployment Runbook

### Prerequisites

- Docker + access to a container registry
- A running Kubernetes cluster with Kubeflow Pipelines and KServe installed
- AWS credentials with S3 read/write access
- `kubectl` configured for the target cluster

### 1. Build & push the inference image

```bash
REGISTRY=your-registry TAG=v1 bash scripts/build_push.sh
```

### 2. Provision Kubernetes resources

```bash
kubectl apply -f k8s/model-pvc.yaml
kubectl apply -f k8s/inference-service.yaml
```

### 3. Run the training pipeline

```bash
python scripts/run_pipeline.py \
  --host http://<kfp-host>:8080 \
  --fused-csv s3://your-bucket/data/fused_logs.csv \
  --s3-bucket your-bucket \
  --kserve-image your-registry/cloudguard:v1
```

The pipeline runs `data_prep → train → evaluate → push_model → deploy_kserve`.  
Deployment is blocked automatically if AUROC < 0.80.

Track the run at: `http://<kfp-host>:8080/#/runs/details/<run_id>`

### 4. Compile the pipeline locally (optional)

```bash
python pipeline/cloudguard_pipeline.py
# outputs: cloudguard_pipeline.yaml
```

### 5. Call the inference endpoint

```bash
curl -X POST http://<kserve-ingress>/v1/models/cloudguard:predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 3, "day_of_week": 1, "is_weekend": 0, "is_offhours": 1,
    "is_failed": 0, "is_rare_ip": 1, "user_enc": 5, "action_enc": 12,
    "source_type_enc": 0, "action_count_1h": 3, "action": "DeleteBucket"
  }'
```

---

## Example Output

```
TTP detected: T1485 - Data Destruction
CVEs found: 3

=== Claude SOC Analysis ===
Explanation  : DeleteBucket was called at 3:14 AM from an IP not previously
               seen in this account, targeting the production backup bucket.
               This is a strong indicator of compromised credentials.

Attack Tech  : T1485 - Data Destruction (MITRE ATT&CK)

Action       : Immediately revoke the admin IAM credentials, enable S3
               versioning and MFA delete on all buckets, and review
               CloudTrail for prior reconnaissance from the flagged IP.
```

---

## Colab Quickstart (Experimentation)

The notebook runs on Google Colab with a T4 GPU.  
`Runtime > Change runtime type > T4 GPU`

### Add Colab Secrets

| Secret Name | Where to get it |
|-------------|----------------|
| `KAGGLE_USERNAME` | kaggle.com/settings |
| `KAGGLE_KEY` | kaggle.com/settings > API > Create New Token |
| `ANTHROPIC_API_KEY` | console.anthropic.com > API Keys |

Upload `CloudGuard.ipynb` to [colab.research.google.com](https://colab.research.google.com) and run cells top to bottom.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML | scikit-learn (Isolation Forest), TensorFlow/Keras (LSTM, experimental) |
| Serving | FastAPI + Uvicorn |
| Containerisation | Docker |
| Pipeline Orchestration | Kubeflow Pipelines v2 |
| Model Serving | KServe v0.13 (Serverless mode) |
| Object Storage | AWS S3 (model artifacts) |
| AI Analyst | Anthropic Claude API |
| Threat Intel | mitreattack-python, NIST NVD REST API |
| Data | pandas, numpy |

---

## Security

- No credentials are hardcoded anywhere in this repository
- Colab secrets are loaded via `google.colab.userdata` at runtime
- Production deployments should use Kubernetes Secrets or a secrets manager (e.g. AWS Secrets Manager, Vault) injected as environment variables
- Never commit API keys or AWS credentials to source control

---

## License

MIT License — see [LICENSE](LICENSE) for details.  
Copyright (c) 2026 Mathanprasath K
