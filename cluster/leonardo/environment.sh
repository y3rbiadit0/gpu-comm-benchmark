#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
stack=${1:-${GPU_BENCH_STACK:-}}

if [[ -z "$stack" ]]; then
  echo "usage: source cluster/leonardo/environment.sh <cuda|sycl>" >&2
  echo "or set GPU_BENCH_STACK=<cuda|sycl> before sourcing" >&2
  return 2 2>/dev/null || exit 2
fi

source "$script_dir/slurm.sh"
# Install prefixes for everything the bootstrap builds. Sourced here as well as by
# the bootstrap targets, so a preset resolving $env{OSHMPI_HOME} gets the same
# answer whether it was reached through bootstrap.sh or `make leonardo`.
source "$script_dir/utils/layout.sh"
source "$script_dir/utils/print-env.sh"

case "$stack" in
  cuda)
    source "$script_dir/env/cuda.sh"
    ;;
  sycl)
    source "$script_dir/env/sycl.sh"
    ;;
  *)
    echo "unknown Leonardo stack '$stack' (expected cuda or sycl)" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
