#!/bin/bash
set -ex

ROOT_DIR=$(pwd)

cd "$ROOT_DIR/nccl" || exit
make -C src "$ROOT_DIR/nccl/build/include/nccl.h"

cd "$ROOT_DIR/fairring" || exit
export CPATH="$ROOT_DIR/nccl/build/include:$CPATH"
USE_TORCH_NCCL=1 pip install --no-deps .
