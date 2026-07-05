#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=vector_add
CP_N_LABEL="problem size"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

cp_experiment_defaults() {
  CP_N=${CP_N:-1048576}
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-}
}

cp_vector_add_main() { cp_experiment_main; }
