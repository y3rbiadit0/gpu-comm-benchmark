#!/usr/bin/env bash
set -euo pipefail

# The gpu-comm-bench binaries for the SYCL/oneCCL-OSHMPI preset.
#
# Separate from the dependency targets so `bootstrap.sh benchmarks` can rebuild
# just the benchmarks after a source change, without re-checking oneCCL.

GPU_BENCH_BUILD_STACK=sycl
GPU_BENCH_BUILD_REQUIRES="oneccl-oshmpi"

[[ "${BASH_SOURCE[0]}" == "$0" || -n "${GPU_BENCH_BUILD_RUN:-}" ]] || return 0

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../../.." && pwd)
source "$script_dir/_lib.sh"

preset=${GPU_BENCH_PRESET:-leonardo-sycl-oneccl-oshmpi}

# The preset reads ONECCL_OSHMPI_ROOT and OSHMPI_HOME through $env{}; _lib.sh
# has already exported them from layout.sh.

gpu_bench_build_log "gpu-comm-bench ($preset)"
cd "$project_root"
cmake --preset "$preset"
cmake --build --preset "$preset"
