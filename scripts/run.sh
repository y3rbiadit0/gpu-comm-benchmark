#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <np> <binary> [args...]" >&2
  exit 2
fi

np="$1"
binary="$2"
shift 2

mpirun -np "${np}" "${binary}" "$@"
