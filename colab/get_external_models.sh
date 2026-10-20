#!/usr/bin/env bash
# Clone the two independent Gaussian-plume implementations used as cross-checks by dynamic_qra/gaussian_farfield.py.
# They are not vendored because neither repository states a licence.
set -e
cd "$(dirname "$0")/.."
mkdir -p external
[ -d external/GPM ] || git clone --depth 1 https://github.com/koaandmud/GPM external/GPM
[ -d external/gaussian-plume ] || git clone --depth 1 https://github.com/gerritjandebruin/gaussian-plume external/gaussian-plume
echo "external models ready: external/GPM, external/gaussian-plume"
