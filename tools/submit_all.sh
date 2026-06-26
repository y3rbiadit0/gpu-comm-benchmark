#!/usr/bin/env bash
# Submit comm-playground experiment jobs on Leonardo.
#
# Usage:
#   tools/submit_all.sh                      # submit every benchmark, every backend, every topology
#   tools/submit_all.sh dot_product pingpong # only these benchmarks
#
# Filters (env, space-separated globs):
#   CP_ONLY_BACKENDS="cuda_mpi cuda_nccl"    # restrict backends
#   CP_ONLY_TOPOS="1n4g 2n1g"                # restrict topologies
#   CP_DRYRUN=1                              # print sbatch commands instead of running them
#   CP_MSG_SIZES="1,8,64,1024"              # pingpong only: explicit message sizes
#
# Pass-through overrides (CP_N, CP_ITERS, CP_WARMUP, CP_NTRIALS, CP_MSG_SIZES, ...) are exported
# here and inherited by every job, e.g.:
#   CP_NTRIALS=5 tools/submit_all.sh dot_product
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/cluster/leonardo/experiments"
export CP_PROJECT_ROOT="$ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "error: sbatch not found -- run this on Leonardo (a login node)." >&2
  exit 1
fi

# Benchmarks: explicit args, or every experiment directory that has a common.sh.
benchmarks=("$@")
if [[ ${#benchmarks[@]} -eq 0 ]]; then
  for d in "$EXP"/*/common.sh; do benchmarks+=("$(basename "$(dirname "$d")")"); done
fi

matches() {  # matches <value> <space-separated-globs-or-empty>
  local value="$1" filters="$2" f
  [[ -z "$filters" ]] && return 0
  for f in $filters; do [[ "$value" == $f ]] && return 0; done
  return 1
}

submitted=0
skipped=0
for bench in "${benchmarks[@]}"; do
  bench_dir="$EXP/$bench"
  if [[ ! -d "$bench_dir" ]]; then
    echo "warning: no such benchmark: $bench" >&2
    continue
  fi
  for script in "$bench_dir"/*/*.sh; do
    [[ -e "$script" ]] || continue
    topo="$(basename "$script" .sh)"
    backend="$(basename "$(dirname "$script")")"
    matches "$backend" "${CP_ONLY_BACKENDS:-}" || { skipped=$((skipped+1)); continue; }
    matches "$topo" "${CP_ONLY_TOPOS:-}" || { skipped=$((skipped+1)); continue; }
    if [[ "${CP_DRYRUN:-0}" == "1" ]]; then
      echo "would submit: $bench/$backend/$topo"
    else
      echo "submitting: $bench/$backend/$topo"
      sbatch "$script"
    fi
    submitted=$((submitted+1))
  done
done

echo "---"
echo "submitted: $submitted   skipped (filtered): $skipped"
echo "watch queue: squeue -u \$USER"
echo "when drained, parse: python3 tools/benchscribe > RESULTS.md"
