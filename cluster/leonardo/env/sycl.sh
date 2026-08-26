#!/usr/bin/env bash
set -euo pipefail

# Validated for SYCL-on-NVIDIA experiments on Leonardo.
module purge
module load gcc/12.2.0
module load cmake/4.1.2
module load python
module load ninja
module load curl
module load cuda/12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22

# Which MPI the SYCL stack links.
#
# hpcx (default) is the same HPC-X 2.19 the CUDA stack links, so cuda_mpi and
# sycl_mpi differ only in programming model. It is also sycl_mpi's fastest
# configuration by a wide margin.
#
# openmpi selects the cluster module. It is kept for one experiment and should
# not be used to produce headline numbers: that build has no UCC, and Open MPI's
# `tuned` collectives reduce with host ompi_op functions, so an allreduce on
# device buffers stages through host memory at a flat 0.42 GB/s whatever the size
# -- 176x slower than hpcx at 16 MiB. Collectives that only move bytes
# (alltoall) are unaffected and match cuda_mpi either way, which is what
# identifies the device-side reduction as the cause. Reproduce with:
#
#   GPU_BENCH_SYCL_MPI=openmpi GPU_BENCH_RESULTS_ROOT=results-stock-mpi \
#     GPU_BENCH_ONLY_BACKENDS=sycl_mpi cluster/harness/launch.sh --all allreduce
#
# Never into the default results tree: benchscribe keys on
# (benchmark, backend, topology) and would average the two together.
case "${GPU_BENCH_SYCL_MPI:-hpcx}" in
  openmpi)
    module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
    ;;
  hpcx)
    # The prefix comes from the hpcx-mpi module, which cannot be loaded here --
    # it pulls in nvhpc and would replace DPC++ and CUDA 12.2. Ask a login shell
    # for it instead, so nothing from that module leaks into this environment.
    GPU_BENCH_HPCX_OMPI_HOME=${GPU_BENCH_HPCX_OMPI_HOME:-$(
      bash -lc 'module load nvhpc/24.5 hpcx-mpi/2.19 >/dev/null 2>&1; echo "$HPCX_MPI_HOME"' 2>/dev/null
    )}
    : "${GPU_BENCH_HPCX_OMPI_HOME:?could not resolve HPC-X; set it to the ompi prefix}"
    [[ -x "$GPU_BENCH_HPCX_OMPI_HOME/bin/mpicxx" ]] \
      || { echo "no mpicxx under GPU_BENCH_HPCX_OMPI_HOME=$GPU_BENCH_HPCX_OMPI_HOME" >&2; exit 2; }
    export MPI_HOME="$GPU_BENCH_HPCX_OMPI_HOME"
    export HPCX_MPI_HOME="$MPI_HOME"
    # Open MPI locates its MCA components relative to the prefix it was *built*
    # with, not the one it is run from. Using HPC-X outside the module that set
    # it up means saying where it lives, or it starts with no btl/pml/coll
    # components at all and fails in MPI_Init.
    export OPAL_PREFIX="$MPI_HOME"
    # HPC-X ships inside nvhpc, so its mpicc/mpicxx wrappers invoke nvc/nvc++ by
    # default. This stack deliberately does not load nvhpc -- doing so would
    # replace DPC++ and CUDA 12.2 -- so the wrappers would call a compiler that
    # is not on PATH and configure fails with "C compiler cannot create
    # executables". OMPI_CC/OMPI_CXX are Open MPI's supported override; gcc 12.2
    # is what the rest of this stack uses, and what stock Open MPI's wrappers
    # called before the switch.
    export OMPI_CC=${OMPI_CC:-gcc}
    export OMPI_CXX=${OMPI_CXX:-g++}
    # HPC-X's ompi is one component of a bundle; UCX and UCC are siblings of it.
    # Without them on the library path the loader falls back to the system UCX in
    # /lib64, which is too old:
    #
    #   UCX WARN UCP API version is incompatible: required >= 1.17,
    #            actual 1.15.0 (loaded from /lib64/libucp.so.0)
    #
    # An incompatible UCX means the ucx PML cannot initialise, and UCC sits on
    # UCX, so the collective component never engages either -- the allreduce
    # silently falls back to a host-staged path and looks exactly like stock
    # Open MPI. Derived from the ompi prefix rather than asked for, because a
    # variable nobody sets is a variable that is wrong.
    GPU_BENCH_HPCX_ROOT=${GPU_BENCH_HPCX_ROOT:-$(dirname "$MPI_HOME")}
    for _hpcx_component in ucx ucc sharp hcoll; do
      _hpcx_lib="$GPU_BENCH_HPCX_ROOT/$_hpcx_component/lib"
      [[ -d "$_hpcx_lib" ]] && LD_LIBRARY_PATH="$_hpcx_lib:${LD_LIBRARY_PATH:-}"
    done
    unset _hpcx_component _hpcx_lib
    export GPU_BENCH_HPCX_ROOT

    if [[ ! -d "$GPU_BENCH_HPCX_ROOT/ucx/lib" ]]; then
      echo "warning: no ucx/lib under $GPU_BENCH_HPCX_ROOT -- HPC-X will load the" >&2
      echo "         system UCX and its collectives will not engage" >&2
    fi

    export PATH="$MPI_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
    ;;
  *)
    echo "unknown GPU_BENCH_SYCL_MPI='${GPU_BENCH_SYCL_MPI}' (expected openmpi or hpcx)" >&2
    exit 2
    ;;
esac
export GPU_BENCH_SYCL_MPI=${GPU_BENCH_SYCL_MPI:-hpcx}

export DPCPP_HOME=${DPCPP_HOME:-$HOME/opt/dpcpp_6.3}
export DPCPP_INSTALL=${DPCPP_INSTALL:-$DPCPP_HOME/llvm/build/install}
export DPCPP_CLANG=${DPCPP_CLANG:-$DPCPP_INSTALL/bin/clang}
export DPCPP_CLANGXX=${DPCPP_CLANGXX:-$DPCPP_INSTALL/bin/clang++}

export CC="$DPCPP_CLANG"
export CXX="$DPCPP_CLANGXX"

export GCC12_ROOT=${GCC12_ROOT:-${GCC_HOME:?gcc/12.2.0 module must define GCC_HOME}}
export GCC12_LIB=${GCC12_LIB:-$GCC12_ROOT/lib64}

export CUDA_ROOT="$CUDA_HOME"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"

export SYCL_TARGET=${SYCL_TARGET:-nvptx64-nvidia-cuda}
export NVIDIA_GPU_ARCH=${NVIDIA_GPU_ARCH:-sm_80}
export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
export SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER:-cuda}

export GPU_BENCH_SYCL_FLAGS=${GPU_BENCH_SYCL_FLAGS:--fsycl --gcc-toolchain=${GCC12_ROOT} -fsycl-targets=${SYCL_TARGET} -Xsycl-target-backend=${SYCL_TARGET} --cuda-gpu-arch=${NVIDIA_GPU_ARCH}}
export SYCL_FLAGS="$GPU_BENCH_SYCL_FLAGS"

export PATH="$DPCPP_INSTALL/bin:$PATH"
export LD_LIBRARY_PATH="$DPCPP_INSTALL/lib:$GCC12_LIB:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC12_LIB:${LIBRARY_PATH:-}"

# DPC++'s CUDA adapter depends on hwloc symbols newer than Leonardo's module stack provides.
export HWLOC_ROOT=${HWLOC_ROOT:-$HOME/opt/hwloc}
export PATH="$HWLOC_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HWLOC_ROOT/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$HWLOC_ROOT/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

export GPU_BENCH_LEONARDO_STACK=sycl
export GPU_BENCH_CUDA_ARCH=${GPU_BENCH_CUDA_ARCH:-80}
