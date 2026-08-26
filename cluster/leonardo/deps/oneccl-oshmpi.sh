#!/usr/bin/env bash
set -euo pipefail

# oneCCL with the OSHMPI backend, plus the patched OSHMPI it sits on.
#
# The oneCCL fork carries the build scripts and the OSHMPI ownership patch under
# contrib/oshmpi/. Those scripts source nothing and load no modules - they take a
# prepared environment - so this file is where the Leonardo specifics are applied.

GPU_BENCH_BUILD_STACK=sycl
GPU_BENCH_BUILD_REQUIRES=""

# Only run standalone; bootstrap.sh sources this file for its metadata first.
[[ "${BASH_SOURCE[0]}" == "$0" || -n "${GPU_BENCH_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/_lib.sh"

repo=${GPU_BENCH_ONECCL_REPO:-https://github.com/y3rbiadit0/oneCCL}
ref=${GPU_BENCH_ONECCL_OSHMPI_REF:-feat/oshmpi}
src=$GPU_BENCH_SRC_DIR/oneCCL-oshmpi

gpu_bench_build_log "oneCCL source ($ref)"
gpu_bench_clone_at "$repo" "$ref" "$src"

if [[ ! -x "$src/contrib/oshmpi/build_oshmpi.sh" ]]; then
    printf 'error: %s has no contrib/oshmpi build scripts; wrong ref?\n' "$src" >&2
    exit 2
fi

# OSHMPI and oneCCL must resolve one libmpi, so both are handed the same wrappers
# rather than each discovering its own.
export MPI_C_COMPILER=$(command -v mpicc)
export MPI_CXX_COMPILER=$(command -v mpicxx)
: "${MPI_C_COMPILER:?mpicc not on PATH; load an MPI module}"

export OSHMPI_INSTALL_PREFIX=$OSHMPI_HOME
export OSHMPI_BASE_SOURCE_DIR=$GPU_BENCH_SRC_DIR/oshmpi-upstream
export OSHMPI_SOURCE_DIR=$GPU_BENCH_SRC_DIR/oshmpi
export OSHMPI_BUILD_DIR=$GPU_BENCH_BUILD_DIR/oshmpi
export OSHMPI_CUDA_ROOT=${OSHMPI_CUDA_ROOT:-${CUDA_ROOT:-}}

if gpu_bench_build_done "$OSHMPI_INSTALL_PREFIX/include/shmem.h"; then
    gpu_bench_build_log "patched OSHMPI already at $OSHMPI_INSTALL_PREFIX"
else
    gpu_bench_build_log "patched OSHMPI"
    "$src/contrib/oshmpi/build_oshmpi.sh"
    gpu_bench_strip_debug "$OSHMPI_INSTALL_PREFIX"/lib/liboshmpi.so*
fi

gpu_bench_build_log "oneCCL with the OSHMPI backend"
export OSHMPI_ROOT="$OSHMPI_INSTALL_PREFIX"
export ONECCL_C_COMPILER="${DPCPP_CLANG:?env/sycl.sh must define DPCPP_CLANG}"
export ONECCL_CXX_COMPILER="${DPCPP_CLANGXX:?env/sycl.sh must define DPCPP_CLANGXX}"
export ONECCL_SYCL_FLAGS="${SYCL_FLAGS:?env/sycl.sh must define SYCL_FLAGS}"
export ONECCL_BUILD_DIR=$GPU_BENCH_BUILD_DIR/oneCCL-oshmpi
export ONECCL_INSTALL_PREFIX=$ONECCL_OSHMPI_ROOT

# build_oneccl.sh aborts on a build directory configured with another compiler.
# Clear it here rather than making the user find and delete it.
gpu_bench_reset_cmake_dir "$ONECCL_BUILD_DIR" "$ONECCL_CXX_COMPILER"

"$src/contrib/oshmpi/build_oneccl.sh"

printf '\nOSHMPI:  %s\noneCCL:  %s\n' "$OSHMPI_HOME" "$ONECCL_OSHMPI_ROOT"
