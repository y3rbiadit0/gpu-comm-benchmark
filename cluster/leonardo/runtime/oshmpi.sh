#!/usr/bin/env bash
set -euo pipefail

# The Open MPI + UCX baseline every MPI-backed runtime shares.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_openmpi.sh"

# UCX transport tuning shared with the other GPU-buffer MPI runtimes.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_ucx-gpu.sh"

export SHMEM_INFO=${SHMEM_INFO:-0}
export SHMEM_VERSION=${SHMEM_VERSION:-0}

export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}
export OSHMPI_MPI_GPU_FEATURES=${OSHMPI_MPI_GPU_FEATURES:-all}
export OSHMPI_VERBOSE=${OSHMPI_VERBOSE:-0}

# UCC must be on for direct device reductions. Without it, OSHMPI's MPI-backed
# reduction reaches a host operation that cannot consume device pointers.
# Staged memory remains an explicit benchmark fallback.
export OMPI_MCA_osc=${OMPI_MCA_osc:-ucx}
export OMPI_MCA_osc_base_verbose=${OMPI_MCA_osc_base_verbose:-0}
