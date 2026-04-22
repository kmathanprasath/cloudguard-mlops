#!/usr/bin/env bash
# Build and push the CloudGuard Docker image
set -euo pipefail

REGISTRY="${REGISTRY:-your-registry}"
IMAGE_NAME="cloudguard"
TAG="${TAG:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building ${FULL_IMAGE}..."
docker build -t "${FULL_IMAGE}" .

echo "Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"

echo "Done: ${FULL_IMAGE}"
