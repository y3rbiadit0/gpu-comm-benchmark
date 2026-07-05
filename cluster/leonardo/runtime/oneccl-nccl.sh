#!/usr/bin/env bash
set -euo pipefail

export ONECCL_NCCL_ROOT=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}
export ONECCL_BUNDLED_MPI_ROOT=${ONECCL_BUNDLED_MPI_ROOT:-$HOME/oneCCL-nccl/deps/mpi}
user_ccl_mpi_library_path=${CCL_MPI_LIBRARY_PATH:-}
libfabric_dir=${ONECCL_LIBFABRIC_DIR:-$ONECCL_NCCL_ROOT/opt/mpi/libfabric/lib}
libfabric_provider_dir=${ONECCL_LIBFABRIC_PROVIDER_DIR:-$libfabric_dir/prov-tcp-only}

if [[ -f "$ONECCL_NCCL_ROOT/env/vars.sh" ]]; then
  set +u
  source "$ONECCL_NCCL_ROOT/env/vars.sh"
  set -u
fi

export CCL_BACKEND=${CCL_BACKEND:-nccl}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-mpi}

if [[ -n "$user_ccl_mpi_library_path" ]]; then
  export CCL_MPI_LIBRARY_PATH="$user_ccl_mpi_library_path"
else
  unset CCL_MPI_LIBRARY_PATH
  ccl_mpi_candidates=(
    "$ONECCL_NCCL_ROOT/opt/mpi/lib/release/libmpi.so"
    "$ONECCL_NCCL_ROOT"/opt/mpi/lib/release/libmpi.so.*
    "$ONECCL_NCCL_ROOT/opt/mpi/lib/libmpi.so"
    "$ONECCL_NCCL_ROOT"/opt/mpi/lib/libmpi.so.*
    "$ONECCL_BUNDLED_MPI_ROOT/lib/libmpi.so.12"
    "$ONECCL_BUNDLED_MPI_ROOT/lib/libmpi.so"
    "$ONECCL_BUNDLED_MPI_ROOT"/lib/libmpi.so.*
  )
  for ccl_mpi_candidate in "${ccl_mpi_candidates[@]}"; do
    if [[ -e "$ccl_mpi_candidate" ]]; then
      export CCL_MPI_LIBRARY_PATH="$ccl_mpi_candidate"
      break
    fi
  done
  if [[ -z ${CCL_MPI_LIBRARY_PATH:-} ]]; then
    echo "could not find bundled oneCCL MPI libmpi.so under $ONECCL_NCCL_ROOT or $ONECCL_BUNDLED_MPI_ROOT" >&2
    return 1 2>/dev/null || exit 1
  fi
fi
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-warn}
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1}
export CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY:-auto}

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}
# Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
export NCCL_IB_SL=${NCCL_IB_SL:-1}

if [[ ${COMM_PLAYGROUND_JOB_NODES:-1} -gt 1 ]]; then
  export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-slurm}
  export I_MPI_FABRICS=${I_MPI_FABRICS:-shm:ofi}
  export I_MPI_DEBUG=${I_MPI_DEBUG:-0}
  export I_MPI_OFI_PROVIDER=${I_MPI_OFI_PROVIDER:-tcp}
  export FI_PROVIDER=${FI_PROVIDER:-tcp}
  export FI_PROVIDER_PATH=${FI_PROVIDER_PATH:-$libfabric_provider_dir}
  export FI_LOG_LEVEL=${FI_LOG_LEVEL:-error}
  if [[ ! -e "$libfabric_provider_dir/libtcp-fi.so" && -e "$libfabric_dir/prov/libtcp-fi.so" ]]; then
    mkdir -p "$libfabric_provider_dir"
    ln -sf "$libfabric_dir/prov/libtcp-fi.so" "$libfabric_provider_dir/libtcp-fi.so"
  fi
else
  export I_MPI_HYDRA_BOOTSTRAP=${I_MPI_HYDRA_BOOTSTRAP:-slurm}
  export I_MPI_FABRICS=${I_MPI_FABRICS:-shm}
fi

ccl_mpi_lib_dir=$(dirname -- "$CCL_MPI_LIBRARY_PATH")
if [[ -d "$ONECCL_NCCL_ROOT/opt/mpi/bin" ]]; then
  export PATH="$ONECCL_NCCL_ROOT/opt/mpi/bin:$PATH"
fi
if [[ -d "$ONECCL_BUNDLED_MPI_ROOT/bin" ]]; then
  export PATH="$ONECCL_BUNDLED_MPI_ROOT/bin:$PATH"
fi
export LD_LIBRARY_PATH="$libfabric_dir:$ccl_mpi_lib_dir:$ONECCL_NCCL_ROOT/opt/mpi/lib/release:$ONECCL_NCCL_ROOT/opt/mpi/lib:${GCC12_LIB:-}:${DPCPP_INSTALL:-}/lib:${CUDA_HOME:-${CUDA_PATH:-}}/lib64:${LD_LIBRARY_PATH:-}"
