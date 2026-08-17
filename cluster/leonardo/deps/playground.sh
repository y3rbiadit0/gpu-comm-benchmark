#!/usr/bin/env bash
set -euo pipefail

# The comm-playground binaries for the SYCL/oneCCL-OSHMPI preset.
#
# Separate from the dependency targets so `bootstrap.sh playground` can rebuild
# just the benchmarks after a source change, without re-checking oneCCL.

CP_BUILD_STACK=sycl
CP_BUILD_REQUIRES="oneccl-oshmpi"
CP_BUILD_PROVIDES=""

[[ "${BASH_SOURCE[0]}" == "$0" || -n "${CP_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../../.." && pwd)
source "$script_dir/_lib.sh"

preset=${CP_PRESET:-leonardo-sycl-oneccl-oshmpi}

# The preset reads these through $env{}, so they must be exported before
# configure rather than passed on the command line.
export ONECCL_OSHMPI_ROOT=${ONECCL_OSHMPI_ROOT:-$HOME/opt/oneccl-oshmpi}
export OSHMPI_HOME=${OSHMPI_HOME:-$HOME/opt/oshmpi-ee5cf110-oneccl}

cp_build_log "comm-playground ($preset)"
cd "$project_root"
cmake --preset "$preset"
cmake --build --preset "$preset"
