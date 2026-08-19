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
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2

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
