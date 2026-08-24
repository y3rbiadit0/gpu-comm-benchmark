#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=alltoall
GPU_BENCH_N_LABEL="count per peer"

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

# UCC is enabled globally in runtime/mpi-cuda.sh because it is transformative
# for allreduce (139x at 16 MiB). It is the opposite for the all-to-all family:
#
#   cuda_mpi moe 1n4g, GB/s      UCC=1    UCC=0     gain
#     uniform                     21.8    167.9     7.7x
#     locality80                   5.5    258.3    47.0x
#     hotspot80                    9.2     80.2     8.7x
#
# With UCC off, cuda_mpi matches sycl_mpi to within 1% on every routing case --
# stock Open MPI has no working UCC, so it was running `tuned` all along, and
# both paths converge once cuda_mpi does too. Measured 2026-08-24, job 53993204.
#
# Each backend is measured in its best configuration for the operation under
# test; print-env.sh records the setting in every job log.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}


# alltoall sends GPU_BENCH_N elements to every peer (send/recv buffers are ranks*GPU_BENCH_N).
gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-65536}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  # alltoall binaries accept: <count_per_peer> [iterations] [warmup]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
}

gpu_bench_alltoall_main() { gpu_bench_experiment_main; }
