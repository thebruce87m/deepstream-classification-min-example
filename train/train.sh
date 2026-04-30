#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$REPO_ROOT"

DOCKER_REGISTRY="nvcr.io"
DOCKER_NAME="nvidia/tao/tao-toolkit"
DOCKER_TAG="6.26.3-pyt"
DOCKER_CONTAINER="$DOCKER_REGISTRY/$DOCKER_NAME:$DOCKER_TAG"

docker run \
  -it \
  --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v "$WORKSPACE_DIR":/workspace \
  "$DOCKER_CONTAINER" \
  classification_pyt train \
  -e /workspace/train/train.yaml