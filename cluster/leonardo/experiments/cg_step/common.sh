#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=cg_step
GPU_BENCH_N_LABEL="grid side"

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

# cg_step's two reductions are on a single double -- 8 bytes -- which is deep in
# the regime where UCC loses. The global default in runtime/mpi-cuda.sh is on,
# because UCC is transformative for large allreduce; here it is the opposite:
#
#   cuda_mpi cg_step 2n4g   UCC=1  245.6 us   (last of six backends)
#                           UCC=0  129.0 us   (third)  -- 1.90x, job 54050170
#
# The rule is message size, not benchmark: UCC wins above ~64 KiB and loses
# below. allreduce sweeps across that boundary and keeps the global default;
# cg_step, alltoall and moe each have one characteristic size and set the
# setting that suits it. print-env.sh records the choice in every job log.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}


# cg_step works on a square GPU_BENCH_N x GPU_BENCH_N grid; GPU_BENCH_N is the side length.
# Kept small so the halo exchange + two reductions dominate over stencil compute.
gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-512}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-50}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-10}
  # cg_step binaries accept: <side> [iterations] [warmup]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
}

gpu_bench_cg_step_main() { gpu_bench_experiment_main; }
