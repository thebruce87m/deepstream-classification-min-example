#!/bin/bash

set -e

# Absolute path of the script directory
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd $SCRIPT_DIR

docker build . \
-t ds-classification