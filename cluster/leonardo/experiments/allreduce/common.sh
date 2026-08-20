#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=allreduce
GPU_BENCH_N_LABEL="max message elements"

source "$GPU_BENCH_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

# allreduce sweeps powers of two from 1 element up to GPU_BENCH_N by default;
# GPU_BENCH_MSG_SIZES can override that with a comma-separated list.
gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-4194304}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  # allreduce binaries accept: <max_elements> [iterations] [warmup] [message_sizes]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
  if [[ -n "${GPU_BENCH_MSG_SIZES:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_MSG_SIZES"
  fi
}

gpu_bench_experiment_extra_summary() {
  echo "message sizes: ${GPU_BENCH_MSG_SIZES:-powers-of-two}"
  # OSHMPI-only knob, but recorded for every run so the job log always says
  # which reduction memory path the results came from.
  echo "oshmpi allreduce memory: ${GPU_BENCH_OSHMPI_ALLREDUCE_MEM:-staged}"
}

gpu_bench_allreduce_main() { gpu_bench_experiment_main; }
