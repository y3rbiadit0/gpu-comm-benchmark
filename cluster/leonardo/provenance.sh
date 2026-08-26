#!/usr/bin/env bash
# Where did this environment variable get its value?
#
# The communication-relevant environment is assembled in layers, each using
# ${VAR:-default} so an earlier layer wins:
#
#   1. the submitting shell / sbatch --export      (GPU_BENCH_N=17 tools/launch.sh ...)
#   2. cluster/leonardo/experiments/<bench>/common.sh   (per-benchmark overrides)
#   3. cluster/leonardo/slurm.sh, env/<stack>.sh        (account, modules, paths)
#   4. cluster/leonardo/runtime/<runtime>.sh            (UCC, UCX, NCCL, NVSHMEM)
#
# That is the right design -- a benchmark that needs UCC off says so once -- but
# it makes "why is UCC 0 here and 1 there?" hard to answer from a job log, and
# four of this project's measurement errors were inherited-flag bugs. This file
# records which layer last wrote each variable so the log can say so.
#
# Deliberately not using associative arrays: Leonardo has bash 4, but macOS ships
# 3.2 and this must stay debuggable off-cluster.

GPU_BENCH_ORIGINS=""   # newline-separated "NAME<TAB>origin" records

# Snapshot NAME=VALUE for every exported variable, sorted.
_gpu_bench_env_snapshot() {
  export -p | sed -E 's/^declare -x //; s/^export //' | sort
}

# gpu_bench_record_origin <origin-label> -- attribute every variable that
# differs from the previous snapshot to <origin-label>.
gpu_bench_record_origin() {
  local origin="$1" now name
  now=$(_gpu_bench_env_snapshot)
  while IFS= read -r line; do
    name="${line%%=*}"
    [[ -n "$name" ]] || continue
    GPU_BENCH_ORIGINS="$GPU_BENCH_ORIGINS$name	$origin
"
  done < <(comm -13 <(printf '%s\n' "${_GPU_BENCH_ENV_BEFORE:-}") <(printf '%s\n' "$now"))
  _GPU_BENCH_ENV_BEFORE="$now"
}

# gpu_bench_source_tracked <file> [args...] -- source it, then attribute whatever
# it changed. The path is shortened to be readable in a log.
gpu_bench_source_tracked() {
  local file="$1"; shift
  : "${_GPU_BENCH_ENV_BEFORE:=$(_gpu_bench_env_snapshot)}"
  # shellcheck disable=SC1090
  source "$file" "$@"
  gpu_bench_record_origin "${file#"${GPU_BENCH_PROJECT_ROOT:-}"/}"
}

# Everything already exported when tracking starts came from outside the job.
gpu_bench_origin_baseline() {
  _GPU_BENCH_ENV_BEFORE=$(_gpu_bench_env_snapshot)
  local name
  while IFS= read -r line; do
    name="${line%%=*}"
    [[ -n "$name" ]] || continue
    GPU_BENCH_ORIGINS="$GPU_BENCH_ORIGINS$name	<inherited from submitting shell>
"
  done < <(printf '%s\n' "$_GPU_BENCH_ENV_BEFORE")
}

# gpu_bench_origin_of <NAME> -- the last layer that wrote NAME, or "" if unknown.
gpu_bench_origin_of() {
  printf '%s' "$GPU_BENCH_ORIGINS" | awk -F'\t' -v n="$1" '$1 == n { o = $2 } END { print o }'
}
