#!/bin/bash

set -e

# Absolute path of the script directory
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_DIR


docker run \
--gpus all \
-it \
--rm \
--net=host \
--privileged \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $(pwd):/code/ \
-e DISPLAY=$DISPLAY \
-e CUDA_VER=13.1 \
-w /opt/nvidia/deepstream/deepstream-9.0/sources/apps/sample_apps/deepstream-test1 \
ds-classification