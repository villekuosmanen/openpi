#!/bin/bash
set -e

ARCH="${1:-amd64}"
IMAGE_NAME="${2:-openpi_${ARCH}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$ARCH" = "amd64" ]; then
    docker build -t "$IMAGE_NAME" -f "${SCRIPT_DIR}/Dockerfile" "$REPO_DIR"
else
    docker buildx build \
        --platform "linux/${ARCH}" \
        -t "$IMAGE_NAME" \
        --load \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "$REPO_DIR"
fi

