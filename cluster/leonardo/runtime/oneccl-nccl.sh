#!/usr/bin/env bash
set -euo pipefail

export ONECCL_NCCL_ROOT=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}

if [[ -f "$ONECCL_NCCL_ROOT/env/vars.sh" ]]; then
  set +u
  source "$ONECCL_NCCL_ROOT/env/vars.sh"
  set -u
fi

export CCL_BACKEND=${CCL_BACKEND:-nccl}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}
export CCL_MPI_LIBRARY_PATH=${CCL_MPI_LIBRARY_PATH:-$HOME/oneCCL-nccl/deps/mpi/lib/libmpi.so.12}
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-warn}
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1}
export CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY:-auto}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

if [[ ${COMM_PLAYGROUND_JOB_NODES:-1} -gt 1 ]]; then
  export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-slurm}
  export I_MPI_FABRICS=${I_MPI_FABRICS:-shm:ofi}
  export I_MPI_OFI_PROVIDER=${I_MPI_OFI_PROVIDER:-tcp}
  export FI_PROVIDER=${FI_PROVIDER:-tcp}
  export FI_LOG_LEVEL=${FI_LOG_LEVEL:-error}
else
  export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-slurm}
  export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}
fi

mpi_lib_dir=$(dirname -- "$CCL_MPI_LIBRARY_PATH")
export LD_LIBRARY_PATH="$mpi_lib_dir:${GCC12_LIB:-}:${DPCPP_INSTALL:-}/lib:${CUDA_HOME:-${CUDA_PATH:-}}/lib64:${LD_LIBRARY_PATH:-}"
