#!/usr/bin/env bash
set -euo pipefail

GPU_BENCH_EXPERIMENT=alltoall
GPU_BENCH_N_LABEL="count per peer"

source "$GPU_BENCH_PROJECT_ROOT/cluster/harness/experiments/common.sh"

# UCC regresses personalized exchanges on the validated Leonardo stack.
export OMPI_MCA_coll_ucc_enable=${OMPI_MCA_coll_ucc_enable:-0}

# alltoall sends GPU_BENCH_N elements to every peer (send/recv buffers are ranks*GPU_BENCH_N).
gpu_bench_experiment_defaults() {
  GPU_BENCH_N=${GPU_BENCH_N:-65536}
  GPU_BENCH_ITERS=${GPU_BENCH_ITERS:-100}
  GPU_BENCH_WARMUP=${GPU_BENCH_WARMUP:-20}
  # alltoall binaries accept: <max_count_per_peer> [iterations] [warmup] [sizes]
  GPU_BENCH_EXTRA_ARGS=${GPU_BENCH_EXTRA_ARGS:-"$GPU_BENCH_ITERS $GPU_BENCH_WARMUP"}
  if [[ -n "${GPU_BENCH_MSG_SIZES:-}" ]]; then
    GPU_BENCH_EXTRA_ARGS="$GPU_BENCH_EXTRA_ARGS $GPU_BENCH_MSG_SIZES"
  fi
}

gpu_bench_alltoall_main() { gpu_bench_experiment_main; }

gpu_bench_experiment_extra_summary() {
  echo "message sizes: ${GPU_BENCH_MSG_SIZES:-powers-of-two}"
}
