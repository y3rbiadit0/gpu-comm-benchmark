#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stack=${1:-${GPU_BENCH_STACK:-}}

if [[ -z "$stack" ]]; then
  echo "usage: source cluster/leonardo/environment.sh <cuda|sycl>" >&2
  echo "or set GPU_BENCH_STACK=<cuda|sycl> before sourcing" >&2
  return 2 2>/dev/null || exit 2
fi

# Records which file wrote each environment variable, so a job log can answer
# "where is this set?". Must come first: everything below is sourced through it.
source "$script_dir/provenance.sh"
: "${_GPU_BENCH_ENV_BEFORE:=$(_gpu_bench_env_snapshot)}"

gpu_bench_source_tracked "$script_dir/slurm.sh"
# Install prefixes for everything the bootstrap builds. Sourced here as well as by
# the bootstrap targets, so a preset resolving $env{OSHMPI_HOME} gets the same
# answer whether it was reached through bootstrap.sh or `make leonardo`.
gpu_bench_source_tracked "$script_dir/layout.sh"
source "$script_dir/print-env.sh"   # defines a function, sets nothing

case "$stack" in
  cuda)
    gpu_bench_source_tracked "$script_dir/env/cuda.sh"
    ;;
  sycl)
    gpu_bench_source_tracked "$script_dir/env/sycl.sh"
    ;;
  *)
    echo "unknown Leonardo stack '$stack' (expected cuda or sycl)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
