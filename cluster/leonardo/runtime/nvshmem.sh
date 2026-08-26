#!/usr/bin/env bash
set -euo pipefail

# The Open MPI + UCX baseline every MPI-backed runtime shares.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_openmpi.sh"

# Aligned with the other MPI-using runtimes. NVSHMEM uses MPI only to bootstrap
# and for the harness's own validation/statistics reductions, both outside every
# timed region, so this does not affect the measurement - it is set so the only
# difference between runtime files is the transport actually under test.
# Open MPI 4.x renamed this; the old name still works but prints a
# deprecation banner into every rank's stderr on every run.

export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibrc}
export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-0}
# NVSHMEM dispatches its collectives to NCCL when it is available. Disabling
# that leaves its own reduction path, which does not scale:
#
#   allreduce 1n4g, 16 MiB   DISABLE_NCCL=1   0.63 GB/s
#                            DISABLE_NCCL=0  85.13 GB/s   (135x)
#
# With NCCL enabled the result is within 0.2% of cuda_nccl at every size, which
# is the point: NVSHMEM's allreduce *is* NCCL's. That equivalence is a result to
# report, not something to hide by turning NCCL off -- and off is not a
# defensible alternative, since 0.63 GB/s is a non-scaling fallback rather than
# a measurement of anything users would run.
#
# NVSHMEM's native strength is RMA, which halo_1d and pingpong measure and which
# this flag does not affect.
export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-0}
export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}

if [[ ${GPU_BENCH_JOB_NODES:-1} -gt 1 ]]; then
  export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}
  export UCX_TLS=${GPU_BENCH_NVSHMEM_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
  # Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
  # UCX covers the MPI bootstrap; the ibrc data plane needs NVSHMEM's own knob.
  export UCX_IB_SL=${UCX_IB_SL:-1}
  export NVSHMEM_IB_SL=${NVSHMEM_IB_SL:-1}
else
  export UCX_TLS=${GPU_BENCH_NVSHMEM_UCX_TLS:-sm,cuda_copy,cuda_ipc,self}
fi
