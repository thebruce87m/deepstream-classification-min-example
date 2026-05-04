#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$REPO_ROOT"

DOCKER_REGISTRY="nvcr.io"
DOCKER_NAME="nvidia/tao/tao-toolkit"
DOCKER_TAG="6.26.3-deploy"
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
  classification_pyt gen_trt_engine \
    -e /workspace/train/export.yaml \
    gen_trt_engine.onnx_file=/workspace/models/classifier_model.onnx \
    gen_trt_engine.trt_engine=/workspace/models/classifier_model.onnx_b16_gpu0_fp16.engine \
    gen_trt_engine.results_dir=/workspace/results/engine \
    gen_trt_engine.tensorrt.workspace_size=1024 \
    gen_trt_engine.tensorrt.min_batch_size=1 \
    gen_trt_engine.tensorrt.opt_batch_size=1 \
    gen_trt_engine.tensorrt.max_batch_size=1 \
    gen_trt_engine.tensorrt.data_type=FP16