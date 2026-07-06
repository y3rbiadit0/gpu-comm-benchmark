#!/usr/bin/env bash
set -euo pipefail

CP_EXPERIMENT=halo_1d
CP_N_LABEL="problem size"

source "$CP_PROJECT_ROOT/cluster/leonardo/experiments/common.sh"

cp_experiment_defaults() {
  CP_N=${CP_N:-1048576}
  CP_EXTRA_ARGS=${CP_EXTRA_ARGS:-}
}

cp_halo_1d_main() { cp_experiment_main; }
