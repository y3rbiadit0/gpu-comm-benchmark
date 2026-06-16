#!/usr/bin/env bash
set -euo pipefail

# Validated for native CUDA experiments on Leonardo.
module purge
module load nvhpc/24.5 hpcx-mpi/2.19

export NVHPC_CUDA=${NVHPC_CUDA:-${NVHPC_HOME:?nvhpc/24.5 module must define NVHPC_HOME}/Linux_x86_64/24.5/cuda/12.4}
export CUDA_ROOT="$NVHPC_CUDA"
export CUDA_PATH="$NVHPC_CUDA"
export CUDACXX="$NVHPC_CUDA/bin/nvcc"

export MATH_LIBS=${MATH_LIBS:-${NVHPC_HOME}/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux}
export NCCL_HOME=${NCCL_HOME:-${NVHPC_HOME}/Linux_x86_64/24.5/comm_libs/nccl}
export NVSHMEM_HOME=${NVSHMEM_HOME:-${NVHPC_HOME}/Linux_x86_64/24.5/comm_libs/nvshmem}

export PATH="$NVHPC_CUDA/bin:$PATH"
export LD_LIBRARY_PATH="$MATH_LIBS/lib:$NCCL_HOME/lib:$NVSHMEM_HOME/lib:${LD_LIBRARY_PATH:-}"

export COMM_PLAYGROUND_LEONARDO_STACK=cuda
export COMM_PLAYGROUND_CUDA_ARCH=${COMM_PLAYGROUND_CUDA_ARCH:-80}
