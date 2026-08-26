#!/usr/bin/env bash
# Pretty-print the communication-relevant environment as a "Leonardo ENV" block.
# Sourced by cluster/leonardo/environment.sh; call gpu_bench_leonardo_print_env from
# experiment summary functions so every stdout log is self-describing.

gpu_bench_leonardo_print_env() {
  local vars=(
    # UCX (Open MPI / OSHMPI data plane)
    UCX_IB_SL
    UCX_MAX_RNDV_RAILS
    UCX_TLS
    UCX_RNDV_SCHEME
    UCX_RNDV_THRESH
    # Open MPI MCA
    OMPI_MCA_pml
    OMPI_MCA_btl
    OMPI_MCA_osc
    OMPI_MCA_coll_hcoll_enable
    OMPI_MCA_coll_ucc_enable
    OMPI_MCA_opal_cuda_support
    # NCCL
    NCCL_IB_SL
    NCCL_ALGO
    NCCL_PROTO
    NCCL_SOCKET_IFNAME
    NCCL_DEBUG
    # NVSHMEM / SHMEM
    NVSHMEM_IB_SL
    NVSHMEM_BOOTSTRAP
    NVSHMEM_REMOTE_TRANSPORT
    NVSHMEM_IB_ENABLE_IBGDA
    SHMEM_SYMMETRIC_SIZE
    OSHMPI_MPI_GPU_FEATURES
    # oneCCL / Intel MPI / libfabric
    CCL_BACKEND
    CCL_ATL_TRANSPORT
    CCL_MPI_LIBRARY_PATH
    CCL_WORKER_COUNT
    I_MPI_FABRICS
    I_MPI_OFI_PROVIDER
    FI_PROVIDER
    FI_PROVIDER_PATH
    FI_LOG_LEVEL
    # Host-side execution
    OMP_NUM_THREADS
    SLURM_CPU_BIND
  )

  echo "Leonardo ENV : {"
  local var origin
  for var in "${vars[@]}"; do
    # With GPU_BENCH_VERBOSE_ENV=1 each set value carries the file that wrote it.
    # Four of this project's measurement errors were inherited-flag bugs that a
    # value alone could not have caught -- 0 and 1 look equally plausible until
    # you can see which layer chose.
    if [[ "${GPU_BENCH_VERBOSE_ENV:-0}" == "1" && -n "${!var:-}" ]] \
       && declare -F gpu_bench_origin_of >/dev/null; then
      origin=$(gpu_bench_origin_of "$var")
      if [[ -n "$origin" ]]; then
        printf '  %s: %s    <- %s\n' "$var" "${!var}" "$origin"
        continue
      fi
    fi
    echo "  ${var}: ${!var:-unset}"
  done
  echo "}"
}
