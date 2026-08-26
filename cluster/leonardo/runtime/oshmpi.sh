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

# UCC must be on: OSHMPI implements reductions over MPI, so with UCC disabled
# shmem_*_to_all reaches the host `ompi_op` and segfaults on device pointers
# (Leonardo jobs 53261883, 53263113). With it on, the device-resident reduction
# validates (job 53263792) and runs 34x faster than host staging at 16 MiB.
# OSHMPI hands device pointers straight to MPI, so CUDA-aware support matters
# more here than anywhere else. This was the only MPI-using runtime leaving it
# to the build default.
# Open MPI 4.x renamed this; the old name still works but prints a
# deprecation banner into every rank's stderr on every run.
export OMPI_MCA_osc=${OMPI_MCA_osc:-ucx}
export OMPI_MCA_osc_base_verbose=${OMPI_MCA_osc_base_verbose:-0}

