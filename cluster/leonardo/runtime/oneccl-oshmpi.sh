#!/usr/bin/env bash
set -euo pipefail

# oneCCL driven by its OSHMPI backend.
#
# Unlike oneccl-nccl.sh this deliberately does not touch Intel MPI. oneCCL's
# native MPI transport is disabled in this build (ENABLE_MPI=OFF): OSHMPI
# supplies the MPI transport, and it is the OpenMPI 4.1.6 that the sycl stack
# loads. Setting I_MPI_*/FI_* here would configure a library that is not in the
# process.

export ONECCL_OSHMPI_ROOT=${ONECCL_OSHMPI_ROOT:-$HOME/opt/oneccl-oshmpi}
export ONECCL_ROOT="$ONECCL_OSHMPI_ROOT"
export CCL_ROOT="$ONECCL_OSHMPI_ROOT"

export OSHMPI_HOME=${OSHMPI_HOME:-$HOME/opt/oshmpi-ee5cf110-oneccl}
export OSHMPI_ROOT="$OSHMPI_HOME"

if [[ ! -f "$OSHMPI_HOME/include/shmem.h" ]]; then
  echo "OSHMPI not found at $OSHMPI_HOME; set OSHMPI_HOME" >&2
  return 1 2>/dev/null || exit 1
fi
# The backend refuses to run against an unpatched OSHMPI, and the failure would
# otherwise surface as an obscure MPI teardown problem at the end of the job.
if ! grep -q '^#define OSHMPI_PRESERVE_EXTERNAL_MPI 1$' "$OSHMPI_HOME/include/shmem.h"; then
  echo "OSHMPI at $OSHMPI_HOME lacks the external-MPI ownership patch" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -n ${CCL_BACKEND:-} && ${CCL_BACKEND:-} != oshmpi ]]; then
  echo "oneccl-oshmpi runtime requires CCL_BACKEND=oshmpi, got '${CCL_BACKEND:-}'" >&2
  return 1 2>/dev/null || exit 1
fi
export CCL_BACKEND=oshmpi
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-warn}

# The binaries are shared with the NCCL-backed oneCCL runs and compile in
# report.name = "sycl_oneccl_<benchmark>". benchscribe derives the backend from
# that name, so without an override both backends would land in the same row.
export CP_REPORT_BACKEND=${CP_REPORT_BACKEND:-sycl_oneccl_oshmpi}

# Symmetric heap must hold the staging arena plus the point-to-point landing
# area, which is world_size * CCL_OSHMPI_PT2PT_SLOT_SIZE.
export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-2G}
# Production sizing. The oneCCL validation jobs deliberately use a tiny arena to
# force chunking; that would be pathological for a benchmark.
export CCL_OSHMPI_STAGING_SIZE=${CCL_OSHMPI_STAGING_SIZE:-256M}
export CCL_OSHMPI_PT2PT_SLOT_SIZE=${CCL_OSHMPI_PT2PT_SLOT_SIZE:-8388608}

# oneCCL is built with CMAKE_SKIP_RPATH, so nothing records where libccl.so.1 or
# liboshmpi.so live.
export LD_LIBRARY_PATH="$ONECCL_OSHMPI_ROOT/lib:$OSHMPI_HOME/lib:${GCC12_LIB:-}:${DPCPP_INSTALL:-}/lib:${CUDA_HOME:-${CUDA_PATH:-}}/lib64:${LD_LIBRARY_PATH:-}"
