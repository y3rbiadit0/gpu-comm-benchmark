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
# Measured on Leonardo, allreduce, cuda_mpi (2026-08-20). UCC on / UCC off:
#
#     message        1n4g          2n4g
#      16 MiB      138.7x         35.7x
#       4 MiB       87.9x         27.0x
#       1 MiB        9.7x          5.7x
#     256 KiB        3.8x          2.7x
#      64 KiB        1.4x          1.0x   <- crossover
#      16 KiB        0.7x          0.4x   <- UCC loses below here
#         4 B        0.6x          0.3x
#
# Two regimes, and the trade-off is sharper inter-node. Above ~64 KiB UCC is
# transformative: without it the large-message curve *falls* with size (0.42
# GB/s at 16 MiB on 2n4g), which is a host-staged fallback, not a bandwidth
# curve. Below ~64 KiB UCC costs a higher latency floor - 3.1x at 4 B on 2n4g
# (25.8 -> 78.6 us), 1.7x on 1n4g.
#
# On balance UCC is on: a broken large-message baseline distorted every
# normalized number in the suite, and cuda_mpi is what everything is compared
# against. Note this affects COLLECTIVES only - halo_1d and pingpong are
# point-to-point and unchanged, so the latency-floor cost lands on allreduce,
# alltoall, cg_step and moe.
#
# The crossover is itself a result worth reporting rather than a setting to
# hide: sweeping OMPI_MCA_coll_ucc_enable=0/1 gives two alpha-beta curves for
# the same transport. Set it to 0 to reproduce the old behaviour.
export OMPI_MCA_coll_hcoll_enable=${OMPI_MCA_coll_hcoll_enable:-0}
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-1}
export OMPI_MCA_btl=${OMPI_MCA_btl:-^openib}
export OMPI_MCA_mpi_cuda_support=${OMPI_MCA_mpi_cuda_support:-1}
export OMPI_MCA_pml=${OMPI_MCA_pml:-ucx}

if [[ ${GPU_BENCH_JOB_NODES:-1} -gt 1 ]]; then
  export UCX_TLS=${GPU_BENCH_SYCL_UCX_TLS:-sm,cuda_copy,cuda_ipc,rc,self}
  # Match mpi-cuda.sh. These were set there and not here, which is not a stack
  # difference but a tuning difference: pingpong 2n1g measured cuda_mpi at
  # 21.02 GB/s against sycl_mpi's 12.18 on the same fabric, and the point of
  # running both is to compare the stacks, not their UCX settings.
  export UCX_RNDV_SCHEME=${GPU_BENCH_SYCL_UCX_RNDV_SCHEME:-get_zcopy}
  export UCX_RNDV_THRESH=${GPU_BENCH_SYCL_UCX_RNDV_THRESH:-16384}
  # Service level 1 enables adaptive routing on Leonardo's Dragonfly+ fabric.
  export UCX_IB_SL=${UCX_IB_SL:-1}
  # Pin the rail count (UCX default is 2) so multi-rail behavior is explicit.
  export UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
else
  export UCX_TLS=${GPU_BENCH_SYCL_UCX_TLS:-sm,cuda_copy,cuda_ipc,self}
fi
