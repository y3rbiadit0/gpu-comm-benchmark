#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <preset>" >&2
  exit 2
fi

cmake --build --preset "$1"
