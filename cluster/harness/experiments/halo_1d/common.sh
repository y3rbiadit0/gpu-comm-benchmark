#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=halo_1d
GPU_BENCH_N_LABEL="problem size"

source "$GPU_BENCH_PROJECT_ROOT/cluster/harness/experiments/common.sh"

gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-1048576}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  export GPU_BENCH_BATCH_SAMPLES=${GPU_BENCH_BATCH_SAMPLES:-10}
  # The isolated case averages nothing inside a sample, so it needs a longer
  # series than the steady case to pin down the latency intercept.
  export GPU_BENCH_ISOLATED_SAMPLES=${GPU_BENCH_ISOLATED_SAMPLES:-100}
  # halo_1d binaries accept: <max_halo_elems> [iterations] [warmup] [halo_sizes]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
  if [[ -n "${GPU_BENCH_MSG_SIZES:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_MSG_SIZES"
  fi
}

gpu_bench_experiment_extra_summary() {
  echo "message sizes: ${GPU_BENCH_MSG_SIZES:-powers-of-two}"
  echo "batch samples (steady): $GPU_BENCH_BATCH_SAMPLES"
  echo "batch samples (isolated): $GPU_BENCH_ISOLATED_SAMPLES"
}

gpu_bench_halo_1d_main() { gpu_bench_experiment_main; }
