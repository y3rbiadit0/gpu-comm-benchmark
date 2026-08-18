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
source "$script_dir/print-env.sh"

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
