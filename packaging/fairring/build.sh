ROOT_DIR=$(pwd)

cd "$ROOT_DIR/nccl"
make -j src.build

cd "$ROOT_DIR/fairring"
export CPATH="$ROOT_DIR/nccl/build/include:$CPATH"
export LIBRARY_PATH="$ROOT_DIR/nccl/build/lib:$LIBRARY_PATH"
pip install --no-deps .