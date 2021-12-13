ROOT_DIR=$(pwd)

cd "$ROOT_DIR/nccl"
CXX=$SYSTEM_CXX make -j src.build

cd "$ROOT_DIR/tensorpipe"
export GLIBCXX_USE_CXX11_ABI="$(python -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")"
mkdir build
cd build
mkdir "$ROOT_DIR/tp_install"
cmake ../ -GNinja -DTP_USE_CUDA=1 -DTP_BUILD_LIBUV=1 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=0 "-DCMAKE_INSTALL_PREFIX=$ROOT_DIR/tp_install" "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=$GLIBCXX_USE_CXX11_ABI -fPIC"
ninja
ninja install

cd "$ROOT_DIR/fairring"
export CPATH="$ROOT_DIR/nccl/build/include:$ROOT_DIR/tp_install/include:$CPATH"
export LIBRARY_PATH="$ROOT_DIR/nccl/build/lib:$ROOT_DIR/tp_install/lib:$ROOT_DIR/tp_install/lib64:$LIBRARY_PATH"
pip install --no-deps .