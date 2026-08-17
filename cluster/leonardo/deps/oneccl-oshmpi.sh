#!/usr/bin/env bash
set -euo pipefail

# oneCCL with the OSHMPI backend, plus the patched OSHMPI it sits on.
#
# The oneCCL fork carries the build scripts and the OSHMPI ownership patch under
# contrib/oshmpi/. Those scripts source nothing and load no modules - they take a
# prepared environment - so this file is where the Leonardo specifics are applied.

CP_BUILD_STACK=sycl
CP_BUILD_REQUIRES=""
CP_BUILD_PROVIDES=${ONECCL_OSHMPI_ROOT:-$HOME/opt/oneccl-oshmpi}

# Only run standalone; bootstrap.sh sources this file for its metadata first.
[[ "${BASH_SOURCE[0]}" == "$0" || -n "${CP_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/_lib.sh"

build_root=$(cp_build_root)
repo=${CP_ONECCL_REPO:-https://github.com/y3rbiadit0/oneCCL}
ref=${CP_ONECCL_OSHMPI_REF:-feat/oshmpi}
src=${CP_ONECCL_OSHMPI_SRC:-$build_root/oneCCL-oshmpi}

cp_build_log "oneCCL source ($ref)"
cp_clone_at "$repo" "$ref" "$src"

if [[ ! -x "$src/contrib/oshmpi/build_oshmpi.sh" ]]; then
    printf 'error: %s has no contrib/oshmpi build scripts; wrong ref?\n' "$src" >&2
    exit 2
fi

# OSHMPI and oneCCL must resolve one libmpi, so both are handed the same wrappers
# rather than each discovering its own.
export MPI_C_COMPILER=$(command -v mpicc)
export MPI_CXX_COMPILER=$(command -v mpicxx)
: "${MPI_C_COMPILER:?mpicc not on PATH; load an MPI module}"

export OSHMPI_INSTALL_PREFIX=${OSHMPI_HOME:-$HOME/opt/oshmpi-ee5cf110-oneccl}
export OSHMPI_BUILD_ROOT=$build_root
export OSHMPI_CUDA_ROOT=${OSHMPI_CUDA_ROOT:-${CUDA_ROOT:-}}

if cp_build_done "$OSHMPI_INSTALL_PREFIX/include/shmem.h"; then
    cp_build_log "patched OSHMPI already at $OSHMPI_INSTALL_PREFIX"
else
    cp_build_log "patched OSHMPI"
    "$src/contrib/oshmpi/build_oshmpi.sh"
fi

cp_build_log "oneCCL with the OSHMPI backend"
export OSHMPI_ROOT="$OSHMPI_INSTALL_PREFIX"
export ONECCL_C_COMPILER="${DPCPP_CLANG:?env/sycl.sh must define DPCPP_CLANG}"
export ONECCL_CXX_COMPILER="${DPCPP_CLANGXX:?env/sycl.sh must define DPCPP_CLANGXX}"
export ONECCL_SYCL_FLAGS="${SYCL_FLAGS:?env/sycl.sh must define SYCL_FLAGS}"
export ONECCL_BUILD_ROOT=$build_root
export ONECCL_INSTALL_PREFIX=$CP_BUILD_PROVIDES
"$src/contrib/oshmpi/build_oneccl.sh"

printf '\nOSHMPI:  %s\noneCCL:  %s\n' "$OSHMPI_INSTALL_PREFIX" "$ONECCL_INSTALL_PREFIX"
