#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <preset> [extra cmake args...]" >&2
  exit 2
fi

preset="$1"
shift

cmake --preset "${preset}" "$@"
