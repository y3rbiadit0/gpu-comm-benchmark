#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=${LC_ALL:-C}
export OMP_DISPLAY_ENV=${OMP_DISPLAY_ENV:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}

export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_mpi_cuda_support=${OMPI_MCA_mpi_cuda_support:-1}

export NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-MPI}
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibrc}
export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-0}
export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-1}
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
