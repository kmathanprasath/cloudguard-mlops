FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps — only what the inference service needs
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# App code
COPY app/ ./app/

# Model artifacts are mounted at runtime via /models
# (populated by the Kubeflow pipeline or a PVC)
ENV MODEL_PATH=/models/iforest.pkl
ENV SCALER_PATH=/models/scaler.pkl
ENV ANOMALY_THRESHOLD=0.5

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
