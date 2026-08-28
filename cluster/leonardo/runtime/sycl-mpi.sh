#!/usr/bin/env bash
set -euo pipefail

# The Open MPI + UCX baseline every MPI-backed runtime shares.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_openmpi.sh"

# UCX transport tuning shared with the other GPU-buffer MPI runtimes.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_ucx-gpu.sh"

export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
export SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER:-cuda}
