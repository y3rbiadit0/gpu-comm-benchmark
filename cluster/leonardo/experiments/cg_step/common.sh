#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=cg_step
GPU_BENCH_N_LABEL="grid side"

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

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
