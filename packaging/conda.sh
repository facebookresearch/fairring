#!/bin/bash
set -ex

FAIRRING_BUILD="py${PYTHON_VERSION}_cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}"

PYTORCH_VERSION_AND_MD5=$(conda search --json 'pytorch[channel=pytorch-test]' | python -c "import json, sys; d = json.load(sys.stdin); p = max((p for p in d['pytorch'] if p['build'].startswith('${FAIRRING_BUILD}_cudnn')), key=lambda p: p['timestamp']); sys.stdout.write(p['version'] + ' ' + p['md5'] + '\n')")
PYTORCH_VERSION=$(echo $PYTORCH_VERSION_AND_MD5 | awk '{ print $1 }')
PYTORCH_MD5=$(echo $PYTORCH_VERSION_AND_MD5 | awk '{ print substr($2,1,8) }')

FAIRRING_LATEST_TAG=$(git tag -l --sort=version:refname 'v*' | tail -n 1 | sed -re 's/^v//')
FAIRRING_VERSION="${FAIRRING_LATEST_TAG}.dev${PYTORCH_MD5}"

if conda search --json "fairring==${FAIRRING_VERSION}[channel=fairring,build=${FAIRRING_BUILD}]"; then
    exit 0
fi

CUDATOOLKIT_CHANNEL="nvidia"
if [[ $CUDA_MAJOR_VERSION == 11 && $CUDA_MINOR_VERSION == 5 ]]; then
    CUDATOOLKIT_CHANNEL="conda-forge"
fi

export PYTORCH_VERSION
export FAIRRING_VERSION
export FAIRRING_LATEST_TAG
export CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
export PATH="$CUDA_HOME/bin:$PATH"

conda build -c defaults -c $CUDATOOLKIT_CHANNEL -c pytorch -c pytorch-test --no-anaconda-upload --python "$PYTHON_VERSION" packaging/fairring

conda install -yq anaconda-client
anaconda -t "${ANACONDA_TOKEN}" upload -u fairring --label main --force --no-progress /opt/conda/conda-bld/linux-64/fairring-*.tar.bz2

for version in $(conda search --json "fairring[channel=fairring,build=${FAIRRING_BUILD}]" | python -c "import json, sys; d = json.load(sys.stdin); sys.stdout.write(''.join(p['version'] + '\n' for p in d['fairring']))"); do
  if [[ "$version" != "$FAIRRING_VERSION" ]]; then
    anaconda -t "${ANACONDA_TOKEN}" remove --force "fairring/fairring/${version}/linux-64/fairring-${version}-${FAIRRING_BUILD}.tar.bz2";
  fi
done
