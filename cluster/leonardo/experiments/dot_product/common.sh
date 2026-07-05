#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=dot_product
CP_N_LABEL="problem size"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

cp_experiment_defaults() {
  CP_N=${CP_N:-1048576}
  CP_ITERS=${CP_ITERS:-100}
  CP_WARMUP=${CP_WARMUP:-20}
  # dot_product binaries accept: <global_size> [iterations] [warmup]
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-"$CP_ITERS $CP_WARMUP"}
}

cp_dot_product_main() { cp_experiment_main; }
