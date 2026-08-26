#!/usr/bin/env bash
set -euo pipefail

# oneCCL with the NCCL backend, from the fork that adds NCCL group support.
#
# Unlike the OSHMPI fork, this branch carries no contrib/ build scripts -- it is
# a plain CMake project -- so the Leonardo build recipe lives beside this file in
# _build-oneccl-nccl.sh, which this target clones a source tree for and drives.
#
# The install layout matters: oneCCL bundles the Intel MPI from its deps/mpi into
# <prefix>/opt/mpi, and cluster/leonardo/runtime/oneccl-nccl.sh expects to find it
# there together with <prefix>/env/vars.sh. The sycl_oneccl executables link that
# same bundled MPI (GPU_BENCH_ONECCL_USE_BUNDLED_MPI=ON in the preset), because an
# Open MPI executable driving a oneCCL that dlopens Intel MPI fails in transport
# setup.

GPU_BENCH_BUILD_STACK=sycl
GPU_BENCH_BUILD_REQUIRES=""

# Only run standalone; bootstrap.sh sources this file for its metadata first.
[[ "${BASH_SOURCE[0]}" == "$0" || -n "${GPU_BENCH_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/_lib.sh"

repo=${GPU_BENCH_ONECCL_REPO:-https://github.com/y3rbiadit0/oneCCL}
ref=${GPU_BENCH_ONECCL_NCCL_REF:-fix/nccl_group_support}
src=$GPU_BENCH_SRC_DIR/oneCCL-nccl

gpu_bench_build_log "oneCCL source ($ref)"
gpu_bench_clone_at "$repo" "$ref" "$src"

build_dir=$GPU_BENCH_BUILD_DIR/oneCCL-nccl

if gpu_bench_build_done "$ONECCL_NCCL_ROOT/env/vars.sh"; then
    gpu_bench_build_log "oneCCL (NCCL) already at $ONECCL_NCCL_ROOT"
else
    gpu_bench_build_log "oneCCL with the NCCL backend"
    # The recipe reconfigures a clean directory by default; clear it explicitly
    # so a run after a compiler change cannot trip CMake's cached-compiler guard.
    gpu_bench_reset_cmake_dir "$build_dir" "${DPCPP_CLANGXX:?env/sycl.sh must define DPCPP_CLANGXX}"

    # --skip-env: bootstrap.sh has already loaded the sycl stack for this target,
    # and letting the recipe re-source environment.sh would load it twice.
    # NCCL is not passed explicitly: env/sycl.sh loads the nccl module, and the
    # recipe already resolves NCCL_ROOT then NCCL_HOME and fails with a clear
    # message if neither is set. Naming one here would guess at which the module
    # exports.
    # --no-examples: we link libccl.so and build our own benchmarks against it;
    # oneCCL's example binaries are never run here. They are also the only part
    # of the tree that fails to compile with the DPC++ build in use -- an LLVM
    # assertion ("VPlan cost model and legacy cost model disagreed") firing while
    # vectorizing a SYCL kernel in examples/benchmark. Skipping them avoids a
    # compiler bug in code we do not need, rather than working around it.
    "$script_dir/_build-oneccl-nccl.sh" \
        --skip-env \
        --no-examples \
        --source-dir "$src" \
        --build-dir "$build_dir" \
        --install-prefix "$ONECCL_NCCL_ROOT"
fi

printf '\noneCCL (NCCL): %s\n' "$ONECCL_NCCL_ROOT"
