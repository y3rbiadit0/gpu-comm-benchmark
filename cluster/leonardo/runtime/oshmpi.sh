#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=${LC_ALL:-C}
export OMP_DISPLAY_ENV=${OMP_DISPLAY_ENV:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}
export SHMEM_INFO=${SHMEM_INFO:-0}
export SHMEM_VERSION=${SHMEM_VERSION:-0}

export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}
export OSHMPI_MPI_GPU_FEATURES=${OSHMPI_MPI_GPU_FEATURES:-all}
export OSHMPI_VERBOSE=${OSHMPI_VERBOSE:-0}

export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_opal_cuda_support=${OMPI_MCA_opal_cuda_support:-1}
export OMPI_MCA_osc=${OMPI_MCA_osc:-ucx}
export OMPI_MCA_osc_base_verbose=${OMPI_MCA_osc_base_verbose:-0}
export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}

if [[ ${COMM_PLAYGROUND_JOB_NODES:-1} -gt 1 ]]; then
  export UCX_TLS=${COMM_PLAYGROUND_OSHMPI_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
  export UCX_RNDV_SCHEME=${COMM_PLAYGROUND_OSHMPI_UCX_RNDV_SCHEME:-get_zcopy}
  export UCX_RNDV_THRESH=${COMM_PLAYGROUND_OSHMPI_UCX_RNDV_THRESH:-16384}
  # Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
  export UCX_IB_SL=${UCX_IB_SL:-1}
  # Pin the rail count (UCX default is 2) so multi-rail behavior is explicit.
  export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
else
  export UCX_TLS=${COMM_PLAYGROUND_OSHMPI_UCX_TLS:-sm,cuda_copy,cuda_ipc,self}
fi
