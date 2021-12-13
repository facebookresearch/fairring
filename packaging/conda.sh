#!/bin/bash
set -ex

export PYTORCH_BUILD_PREFIX="py${PYTHON_VERSION}_cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}_cudnn"
export PYTORCH_VERSION=$(conda search --json 'pytorch[channel=pytorch-nightly]' | python -c "import json, sys; d = json.load(sys.stdin); sys.stdout.write(''.join(sorted(p['version'] + '\n' for p in d['pytorch'] if p['build'].startswith('${PYTORCH_BUILD_PREFIX}'))[-1]))")
export DATE=$(echo ${PYTORCH_VERSION} | sed -re 's/^.*\.dev([0-9]+)$/\1/')

export FAIRRING_LATEST_TAG=$(git tag -l --sort=version:refname 'v*' | tail -n 1 | sed -re 's/^v//')
export FAIRRING_VERSION="${FAIRRING_LATEST_TAG}.dev${DATE}"

if conda search --json "fairring==${FAIRRING_VERSION}[channel=fairring,build=py${PYTHON_VERSION}_cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}]"; then
    exit 0
fi

export CUDATOOLKIT_CHANNEL="nvidia"
if [[ $CUDA_MAJOR_VERSION == 11 && $CUDA_MINOR_VERSION == 5 ]]; then
    export CUDATOOLKIT_CHANNEL="conda-forge"
fi

export CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
export PATH="$CUDA_HOME/bin:$PATH"
export SYSTEM_CXX=$(which g++)

conda install -c defaults -c $CUDATOOLKIT_CHANNEL -c pytorch -c pytorch-nightly pytorch==$PYTORCH_VERSION cudatoolkit=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} --download-only --yes

for i in `seq 30`; do sleep 60; echo "Still going..."; done &
SPAMMER_PID=$!
conda build -c defaults -c $CUDATOOLKIT_CHANNEL -c pytorch -c pytorch-nightly -c conda-forge --no-anaconda-upload --python "$PYTHON_VERSION" packaging/fairring
kill $SPAMMER_PID

conda install -yq anaconda-client
anaconda -t "${ANACONDA_TOKEN}" upload -u fairring --label main --force --no-progress /opt/conda/conda-bld/linux-64/fairring-*.tar.bz2