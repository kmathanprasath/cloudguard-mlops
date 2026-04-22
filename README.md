# CloudGuard v1

> An AI-powered cloud threat detection system that fuses multi-source security logs, detects anomalies using machine learning, maps threats to MITRE ATT&CK TTPs, enriches alerts with NVD CVEs, and uses Claude as an automated SOC analyst for plain-English alert triage.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![Model](https://img.shields.io/badge/Model-Isolation%20Forest-purple)

---

## How It Works

```
AWS CloudTrail ──┐
BETH (K8s)    ───┼──► Feature Engineering ──► Isolation Forest ──► MITRE ATT&CK Mapping ──► Claude SOC Analyst
Linux Auth    ──┘
```

---

## Features

- Multi-source log fusion — AWS CloudTrail, BETH K8s syscalls, Linux SSH auth logs
- Unsupervised anomaly detection using Isolation Forest (AUROC 0.8935)
- MITRE ATT&CK TTP mapping with 4 rule-based detectors
- NVD CVE enrichment via NIST REST API for flagged events
- Claude-powered SOC analyst returning structured JSON: explanation, attack technique, recommended action
- Threshold tuning grid search to optimize Precision / Recall / F1

---

## Datasets

| Source | Dataset | Description |
|---|---|---|
| AWS CloudTrail | [flaws.cloud logs](http://summitroute.com/downloads/flaws_cloudtrail_logs.tar) | Real-world misconfigured AWS environment logs |
| K8s / Syscalls | [BETH Dataset](https://www.kaggle.com/datasets/katehighnam/beth-dataset) | Labelled Linux kernel syscall logs with evil column |
| Linux Auth | [LogHub Linux_2k.log](https://github.com/logpai/loghub/tree/master/Linux) | Real SSH authentication logs with brute-force patterns |

---

## Model Performance

| Model | AUROC | Notes |
|---|---|---|
| Isolation Forest | 0.8935 | Primary detector, n_estimators=200, contamination=0.005 |
| LSTM Autoencoder | ~0.49 | Experimental only, not used in final pipeline |

---

## Project Structure

```
CloudGuard.ipynb            # ML notebook — training & experimentation
requirements.txt            # Full notebook dependencies
requirements-serve.txt      # Minimal inference service dependencies
Dockerfile                  # Container image for the FastAPI service

app/
  main.py                   # FastAPI inference service

pipeline/
  cloudguard_pipeline.py    # Kubeflow Pipeline (data_prep → train → evaluate → push → deploy)

k8s/
  inference-service.yaml    # KServe InferenceService manifest
  model-pvc.yaml            # PersistentVolumeClaim for model artifacts

scripts/
  build_push.sh             # Build & push Docker image
  run_pipeline.py           # Submit pipeline to Kubeflow
```

### Notebook Sections

| Cells | What it does |
|---|---|
| 0 | Install dependencies |
| 1-5 | Download datasets (CloudTrail, BETH, Linux auth) |
| 6-8 | Parse each source into unified schema |
| 9-11 | Fuse sources, engineer features, train/test split |
| 12-20 | Label fixing and data validation |
| 21-23 | Train Isolation Forest, evaluate, threshold tuning |
| 24-28 | LSTM Autoencoder (experimental) |
| 29 | Finalize models |
| 30 | NVD CVE enrichment function |
| 31 | MITRE ATT&CK TTP mapping rules |
| 32-34 | Claude SOC analyst setup and end-to-end test |

---

## Setup

This project runs on Google Colab with a T4 GPU.
Runtime > Change runtime type > T4 GPU

### 1. Add Colab Secrets

Click the key icon in the Colab left sidebar and add:

| Secret Name | Where to get it |
|---|---|
| KAGGLE_USERNAME | Your Kaggle username from kaggle.com/settings |
| KAGGLE_KEY | kaggle.com/settings > API > Create New Token |
| ANTHROPIC_API_KEY | console.anthropic.com > API Keys |

### 2. Open in Colab

Upload `CloudGuard_v1.ipynb` to colab.research.google.com and run cells top to bottom.

### 3. Dependencies

```bash
pip install anthropic scikit-learn tensorflow keras pandas numpy
pip install fastapi uvicorn requests python-dotenv kaggle mitreattack-python
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

## MLOps Deployment

### 1. Build & push the Docker image
```bash
REGISTRY=your-registry TAG=v1 bash scripts/build_push.sh
```

### 2. Apply Kubernetes manifests
```bash
kubectl apply -f k8s/model-pvc.yaml
kubectl apply -f k8s/inference-service.yaml
```

### 3. Run the Kubeflow training pipeline
```bash
python scripts/run_pipeline.py \
  --host http://<kfp-host>:8080 \
  --fused-csv s3://your-bucket/data/fused_logs.csv \
  --s3-bucket your-bucket \
  --kserve-image your-registry/cloudguard:v1
```

The pipeline runs: `data_prep → train → evaluate → push_model → deploy_kserve`  
A quality gate blocks deployment if AUROC < 0.80.

### 4. Call the inference endpoint
```bash
# Via KServe ingress
curl -X POST http://<kserve-ingress>/v1/models/cloudguard:predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 3, "day_of_week": 1, "is_weekend": 0, "is_offhours": 1,
    "is_failed": 0, "is_rare_ip": 1, "user_enc": 5, "action_enc": 12,
    "source_type_enc": 0, "action_count_1h": 3, "action": "DeleteBucket"
  }'
```

## Tech Stack

- Python 3.12
- scikit-learn (Isolation Forest)
- TensorFlow / Keras (LSTM Autoencoder)
- FastAPI + Uvicorn (inference service)
- Docker (containerisation)
- Kubeflow Pipelines v2 (ML pipeline orchestration)
- KServe v0.13 (model serving on Kubernetes)
- Anthropic Claude API
- mitreattack-python (MITRE ATT&CK)
- NIST NVD REST API (CVE enrichment)
- pandas, numpy

---

## Security

No credentials are hardcoded in this notebook. All secrets are loaded at runtime via `google.colab.userdata`. Never commit API keys to source control.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Mathanprasath K

You are free to use, modify, and distribute this project, but you must include the original copyright notice and give credit to the author.
