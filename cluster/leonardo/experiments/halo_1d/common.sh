#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=halo_1d
GPU_BENCH_N_LABEL="problem size"

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-1048576}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  # halo_1d binaries accept: <max_halo_elems> [iterations] [warmup] [halo_sizes]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
  if [[ -n "${GPU_BENCH_MSG_SIZES:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_MSG_SIZES"
  fi
}

gpu_bench_experiment_extra_summary() {
  echo "message sizes: ${GPU_BENCH_MSG_SIZES:-powers-of-two}"
}

gpu_bench_halo_1d_main() { gpu_bench_experiment_main; }
