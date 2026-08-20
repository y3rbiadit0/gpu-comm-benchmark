#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=${LC_ALL:-C}
export OMP_DISPLAY_ENV=${OMP_DISPLAY_ENV:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export SLURM_CPU_BIND=${SLURM_CPU_BIND:-none}

export ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-cuda:*}
export SYCL_DEVICE_FILTER=${SYCL_DEVICE_FILTER:-cuda}

# Open MPI's accelerated collective components.
#
# hcoll stays off. UCC is ON by default because leaving it off silently
# cripples every MPI collective on device buffers: without it, `tuned`/libnbc
# reduce with the host `ompi_op` (ompi_op_avx_2buff_add_float_avx2), so Open MPI
# has to stage device->host->device inside the collective.
#
# Measured on Leonardo, allreduce 1n4g, cuda_mpi (2026-08-20):
#
#     message     UCC off      UCC on    speedup
#      16 MiB   48655 us      351 us      138.7x
#       4 MiB   11783 us      134 us       87.9x
#     512 KiB     432 us       68 us        6.4x
#      64 KiB      83 us       59 us        1.4x
#       4 KiB      33 us       57 us        0.6x   <- UCC loses here
#         64 B      24 us       37 us        0.7x   <- and here
#
# The crossover is around 32-64 KiB: UCC has a higher latency floor for small
# messages and a bad step at 4-16 KiB, but everything from 64 KiB up is
# transformative, and with it off the large-message curve *falls* with size,
# which is a fallback, not a bandwidth curve. Since cuda_mpi is the suite's
# baseline, running it crippled distorted every normalized number.
#
# Set OMPI_MCA_coll_ucc_enable=0 to reproduce the old behaviour.
export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-1}
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_mpi_cuda_support=${OMPI_MCA_mpi_cuda_support:-1}
export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}

if [[ ${GPU_BENCH_JOB_NODES:-1} -gt 1 ]]; then
  export UCX_TLS=${GPU_BENCH_SYCL_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
  # Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
  export UCX_IB_SL=${UCX_IB_SL:-1}
  # Pin the rail count (UCX default is 2) so multi-rail behavior is explicit.
  export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
else
  export UCX_TLS=${GPU_BENCH_SYCL_UCX_TLS:-sm,cuda_copy,cuda_ipc,self}
fi
