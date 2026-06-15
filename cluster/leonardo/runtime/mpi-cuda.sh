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

if [[ ${COMM_PLAYGROUND_JOB_NODES:-1} -gt 1 ]]; then
  export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}
  export UCX_TLS=${COMM_PLAYGROUND_CUDA_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
  export UCX_RNDV_SCHEME=${COMM_PLAYGROUND_CUDA_UCX_RNDV_SCHEME:-get_zcopy}
  export UCX_RNDV_THRESH=${COMM_PLAYGROUND_CUDA_UCX_RNDV_THRESH:-16384}
else
  export UCX_TLS=${COMM_PLAYGROUND_CUDA_UCX_TLS:-sm,cuda_copy,cuda_ipc,self}
fi
