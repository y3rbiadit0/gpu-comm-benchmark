#!/usr/bin/env bash
# Submit gpu-comm-bench experiment jobs on Leonardo.
#
# Usage:
#   tools/submit_all.sh                      # submit every benchmark, every backend, every topology
#   tools/submit_all.sh allreduce pingpong    # only these benchmarks
#
# Filters (env, space-separated globs):
#   GPU_BENCH_ONLY_BACKENDS="cuda_mpi cuda_nccl"    # restrict backends
#   GPU_BENCH_ONLY_TOPOS="1n4g 2n1g"                # restrict topologies
#   GPU_BENCH_DRYRUN=1                              # print sbatch commands instead of running them
#   GPU_BENCH_REPEATS=5                      # submit each experiment as N independent jobs
#   GPU_BENCH_SLURM_ACCOUNT=IscrC_OTHER             # override the default allocation
#   GPU_BENCH_SLURM_PARTITION=...                   # override the default partition
#   GPU_BENCH_MSG_SIZES="1,8,64,1024"              # pingpong only: explicit message sizes
#
# Pass-through overrides (GPU_BENCH_N, GPU_BENCH_ITERS, GPU_BENCH_WARMUP, GPU_BENCH_NTRIALS, GPU_BENCH_MSG_SIZES, ...) are exported
# here and inherited by every job, e.g.:
#   GPU_BENCH_NTRIALS=5 tools/submit_all.sh allreduce
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/cluster/leonardo/experiments"
export GPU_BENCH_PROJECT_ROOT="$ROOT"

# Account and partition come from here, not from -A/-p in every job script.
# Override with GPU_BENCH_SLURM_ACCOUNT / GPU_BENCH_SLURM_PARTITION.
source "$ROOT/cluster/leonardo/slurm.sh"

# A dry run only prints what it would do, so it stays useful off-cluster.
if [[ "${GPU_BENCH_DRYRUN:-0}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
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

# Repetition that matters is at the job level, not inside one. Trials within a
# job share an allocation, so they measure the same nodes and GPUs; the spread
# that shows up between allocations is several times larger. Each repeat is a
# separate job, and results key on $SLURM_JOB_ID so they accumulate side by side.
repeats=${GPU_BENCH_REPEATS:-1}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_BENCH_REPEATS must be a positive integer" >&2; exit 2; }

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
    matches "$backend" "${GPU_BENCH_ONLY_BACKENDS:-}" || { skipped=$((skipped+1)); continue; }
    matches "$topo" "${GPU_BENCH_ONLY_TOPOS:-}" || { skipped=$((skipped+1)); continue; }
    for repeat in $(seq "$repeats"); do
      label="$bench/$backend/$topo"
      if [[ "$repeats" -gt 1 ]]; then
        label="$label (repeat $repeat/$repeats)"
      fi
      if [[ "${GPU_BENCH_DRYRUN:-0}" == "1" ]]; then
        echo "would submit: $label"
      else
        echo "submitting: $label"
        sbatch "$script"
      fi
      submitted=$((submitted+1))
    done
  done
done

echo "---"
echo "submitted: $submitted   skipped (filtered): $skipped"
echo "watch queue: squeue -u \$USER"
echo "when drained, parse: python3 tools/benchscribe > RESULTS.md"
