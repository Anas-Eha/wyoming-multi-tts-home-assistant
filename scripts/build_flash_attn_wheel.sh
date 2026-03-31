#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-${ROOT_DIR}/wheelhouse}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.6.2-devel-ubuntu22.04}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0+cu126}"
UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

mkdir -p "${WHEELHOUSE_DIR}"

docker run --rm \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${CUDA_IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y build-essential python3 python3-dev python3-venv curl git
    rm -rf /var/lib/apt/lists/*
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:\${PATH}
    mkdir -p /tmp/uv-cache /workspace/wheelhouse
    UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.11 --seed /tmp/flash-attn-wheel
    UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL} UV_CACHE_DIR=/tmp/uv-cache \
      uv pip install --python /tmp/flash-attn-wheel/bin/python \
      'torch==${TORCH_VERSION}' setuptools wheel packaging ninja numpy psutil
    /tmp/flash-attn-wheel/bin/python -m pip wheel \
      --no-deps \
      --no-build-isolation \
      --wheel-dir /workspace/wheelhouse \
      'flash-attn==${FLASH_ATTN_VERSION}'
  "

echo "Built flash-attn wheel into ${WHEELHOUSE_DIR}"
