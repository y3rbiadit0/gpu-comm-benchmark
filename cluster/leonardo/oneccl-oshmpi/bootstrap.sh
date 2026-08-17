#!/usr/bin/env bash
set -euo pipefail

# Builds everything needed to run the oneCCL/OSHMPI experiments on Leonardo, in
# dependency order: patched OSHMPI, then oneCCL with the OSHMPI backend, then the
# playground binaries.
#
# oneCCL's own scripts load no modules and source nothing - they take a prepared
# environment through documented variables. This file is where the Leonardo
# specifics live: modules, DPC++, CUDA and the MPI everything must agree on.
#
#   ONECCL_SRC=$HOME/opt-src/oneCCL-oshmpi \
#     ./cluster/leonardo/oneccl-oshmpi/bootstrap.sh
#
# Re-running is cheap: OSHMPI is skipped when already installed, and the oneCCL
# and playground builds are incremental.

playground_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
oneccl_src=${ONECCL_SRC:?set to your oneCCL checkout with the OSHMPI backend}

if [[ ! -x "$oneccl_src/contrib/oshmpi/build_oshmpi.sh" ]]; then
    printf 'error: not a oneCCL checkout with the OSHMPI backend: %s\n' "$oneccl_src" >&2
    exit 2
fi

# Modules, DPC++, CUDA, Open MPI. Exports DPCPP_CLANG/CLANGXX, SYCL_FLAGS,
# CUDA_ROOT and puts mpicc on PATH.
source "$playground_root/cluster/leonardo/environment.sh" sycl

: "${SCRATCH:?Leonardo should define SCRATCH; builds are not written to \$HOME}"

# OSHMPI and oneCCL must resolve one libmpi, so both are pointed at the same
# compiler wrappers rather than each discovering their own.
mpi_c_compiler=$(command -v mpicc) || { printf 'error: mpicc not on PATH\n' >&2; exit 2; }
mpi_cxx_compiler=$(command -v mpicxx) || { printf 'error: mpicxx not on PATH\n' >&2; exit 2; }
export MPI_C_COMPILER="$mpi_c_compiler"
export MPI_CXX_COMPILER="$mpi_cxx_compiler"

export OSHMPI_INSTALL_PREFIX=${OSHMPI_INSTALL_PREFIX:-$HOME/opt/oshmpi-ee5cf110-oneccl}
export OSHMPI_CUDA_ROOT=${OSHMPI_CUDA_ROOT:-$CUDA_ROOT}
export OSHMPI_BUILD_ROOT=${OSHMPI_BUILD_ROOT:-$SCRATCH}

# --- 1. patched OSHMPI --------------------------------------------------------
if [[ -f "$OSHMPI_INSTALL_PREFIX/include/shmem.h" ]] &&
   grep -q '^#define OSHMPI_PRESERVE_EXTERNAL_MPI 1$' \
        "$OSHMPI_INSTALL_PREFIX/include/shmem.h"; then
    printf '== patched OSHMPI already installed: %s\n' "$OSHMPI_INSTALL_PREFIX"
else
    printf '== building patched OSHMPI\n'
    "$oneccl_src/contrib/oshmpi/build_oshmpi.sh"
fi

export OSHMPI_HOME="$OSHMPI_INSTALL_PREFIX"
export OSHMPI_ROOT="$OSHMPI_INSTALL_PREFIX"

# --- 2. oneCCL with the OSHMPI backend ---------------------------------------
printf '== building oneCCL\n'
export ONECCL_C_COMPILER="$DPCPP_CLANG"
export ONECCL_CXX_COMPILER="$DPCPP_CLANGXX"
export ONECCL_SYCL_FLAGS="$SYCL_FLAGS"
export ONECCL_BUILD_ROOT=${ONECCL_BUILD_ROOT:-$SCRATCH}
export ONECCL_INSTALL_PREFIX=${ONECCL_INSTALL_PREFIX:-$HOME/opt/oneccl-oshmpi}
"$oneccl_src/contrib/oshmpi/build_oneccl.sh"

# --- 3. playground binaries ---------------------------------------------------
printf '== building comm-playground\n'
export ONECCL_OSHMPI_ROOT="$ONECCL_INSTALL_PREFIX"
cmake --preset leonardo-sycl-oneccl-oshmpi
cmake --build --preset leonardo-sycl-oneccl-oshmpi

printf '\n== done\n'
printf 'OSHMPI:  %s\n' "$OSHMPI_INSTALL_PREFIX"
printf 'oneCCL:  %s\n' "$ONECCL_INSTALL_PREFIX"
printf 'run an experiment with:\n'
printf '  sbatch cluster/leonardo/experiments/allreduce/sycl_oneccl_oshmpi/1n2g.sh\n'
