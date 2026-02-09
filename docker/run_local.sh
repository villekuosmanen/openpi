#!/bin/bash
# Run container locally for training/testing

ARCH="${1:-amd64}"
IMAGE_NAME="${2:-openpi_${ARCH}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

docker run --gpus all --rm -it \
    -v "${REPO_DIR}:/workspace/repo" \
    -v "${REPO_DIR}/.venv:/.venv" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -v "${HOME}/.cache/openpi:/openpi_assets" \
    -e "HF_HOME=/root/.cache/huggingface" \
    -e "WANDB_MODE=offline" \
    -e "WANDB_ENTITY=pravsels" \
    -e "OPENPI_DATA_HOME=/openpi_assets" \
    -e "PYTHONPATH=/workspace/repo:${PYTHONPATH}" \
    "$IMAGE_NAME"

