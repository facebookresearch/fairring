#!/bin/bash
set -ex

FAIRRING_BUILD="py${PYTHON_VERSION}_cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}"

PYTORCH_VERSION=$(conda search --json 'pytorch[channel=pytorch]' | python -c "import json, sys; d = json.load(sys.stdin); p = max((p for p in d['pytorch'] if p['build'].startswith('${FAIRRING_BUILD}_cudnn')), key=lambda p: p['timestamp']); sys.stdout.write(p['version'] + '\n')")

FAIRRING_LATEST_TAG=$(git tag -l --sort=version:refname 'v*' | tail -n 1 | sed -re 's/^v//')
FAIRRING_VERSION="${FAIRRING_LATEST_TAG}"

if conda search --json "fairring==${FAIRRING_VERSION}[channel=fairring,build=${FAIRRING_BUILD}]"; then
    exit 0
fi

export PYTORCH_VERSION
export FAIRRING_VERSION
export FAIRRING_LATEST_TAG
export CUDA_HOME=/usr/local/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
export PATH="$CUDA_HOME/bin:$PATH"

conda build -c defaults -c nvidia -c pytorch --no-anaconda-upload --python "$PYTHON_VERSION" packaging/fairring

conda install -yq anaconda-client
anaconda -t "${ANACONDA_TOKEN}" upload -u fairring --label main --force --no-progress /opt/conda/conda-bld/linux-64/fairring-*.tar.bz2

for version in $(conda search --json "fairring[channel=fairring,build=${FAIRRING_BUILD}]" | python -c "import json, sys; d = json.load(sys.stdin); sys.stdout.write(''.join(p['version'] + '\n' for p in d['fairring']))"); do
  if [[ "$version" != "$FAIRRING_VERSION" ]]; then
    anaconda -t "${ANACONDA_TOKEN}" remove --force "fairring/fairring/${version}/linux-64/fairring-${version}-${FAIRRING_BUILD}.tar.bz2";
  fi
done
