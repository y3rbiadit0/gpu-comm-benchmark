#!/usr/bin/env bash
# Launch benchmark jobs on Leonardo.
#
# One cell:
#   tools/launch.sh halo_1d cuda_mpi 1n2g
#   tools/launch.sh cg_step cuda_mpi 2n4g --qos=boost_qos_dbg   # extra sbatch args
#
# Many cells:
#   tools/launch.sh --all                        every benchmark, backend, topology
#   tools/launch.sh --all halo_1d moe            only these benchmarks
#
# Inspect without submitting:
#   tools/launch.sh --dry-run --all              what would be submitted
#   tools/launch.sh --explain halo_1d cuda_mpi 1n2g
#                                                how every value is resolved, and
#                                                which file writes each env var
#
# Filters (env, space-separated globs; --all only):
#   GPU_BENCH_ONLY_BACKENDS="cuda_mpi cuda_nccl"
#   GPU_BENCH_ONLY_TOPOS="1n4g 2n1g"
#   GPU_BENCH_REPEATS=5            submit each cell as N independent jobs
#
# Repetition that matters is at the job level, not inside one. Trials within a
# job share an allocation, so they measure the same nodes and GPUs; the spread
# between allocations is several times larger. Each repeat is a separate job, and
# results key on $SLURM_JOB_ID so they accumulate side by side.
#
# Pass-through overrides (GPU_BENCH_N, GPU_BENCH_ITERS, GPU_BENCH_WARMUP,
# GPU_BENCH_NTRIALS, GPU_BENCH_MSG_SIZES, ...) are exported here and inherited by
# every job.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/cluster/leonardo/experiments"
export GPU_BENCH_PROJECT_ROOT="$ROOT"

source "$ROOT/cluster/leonardo/slurm.sh"
source "$EXP/backends.sh"
source "$EXP/matrix.sh"

mode=single
dry_run=${GPU_BENCH_DRYRUN:-0}
explain=0
args=()
for arg in "$@"; do
  case "$arg" in
    --all)     mode=all ;;
    --dry-run) dry_run=1 ;;
    --explain) explain=1; dry_run=1 ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         args+=("$arg") ;;
  esac
done

require_sbatch() {
  if [[ "$dry_run" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch not found -- run this on Leonardo (a login node)." >&2
    exit 1
  fi
}

# submit <benchmark> <backend> <topology> [extra sbatch args...]
submit() {
  local bench="$1" backend="$2" topo="$3"; shift 3
  gpu_bench_backend_fields "$backend"
  gpu_bench_topology_fields "$topo"

  if [[ "$explain" == "1" ]]; then
    explain_cell "$bench" "$backend" "$topo"
    return 0
  fi
  if [[ "$dry_run" == "1" ]]; then
    echo "would submit: $bench/$backend/$topo"
    return 0
  fi

  # The allocation shape is passed here, not baked into a script: sbatch
  # command-line options override #SBATCH directives, so one job.sh serves every
  # cell. The three GPU_BENCH_* variables reach the job through the environment,
  # which sbatch propagates by default.
  GPU_BENCH_BENCHMARK="$bench" \
  GPU_BENCH_BACKEND="$backend" \
  GPU_BENCH_TOPOLOGY="$topo" \
  sbatch \
    --job-name="${bench}_${backend}_${topo}" \
    --nodes="$GPU_BENCH_NODES" \
    --ntasks-per-node="$GPU_BENCH_TASKS_PER_NODE" \
    --gres=gpu:"$GPU_BENCH_TASKS_PER_NODE" \
    --time="$(gpu_bench_walltime_for "$GPU_BENCH_NODES")" \
    "$@" \
    "$EXP/job.sh"
}

# Print how one cell resolves, and which file writes each environment variable.
# This runs the real sourcing chain, so it reports what a job would actually see
# rather than a description of what it should see.
explain_cell() {
  local bench="$1" backend="$2" topo="$3"
  cat <<EOF
cell        : $bench / $backend / $topo
job script  : cluster/leonardo/experiments/job.sh
benchmark   : cluster/leonardo/experiments/$bench/common.sh
backend row : cluster/leonardo/experiments/backends.sh
  stack     : $GPU_BENCH_STACK
  runtime   : cluster/leonardo/runtime/$GPU_BENCH_RUNTIME.sh
  launcher  : $GPU_BENCH_LAUNCHER
  preset    : $GPU_BENCH_PRESET
  binary    : build/$GPU_BENCH_PRESET/$GPU_BENCH_BINDIR/${GPU_BENCH_BINARY_PREFIX}_${bench}
topology    : $GPU_BENCH_NODES node(s) x $GPU_BENCH_TASKS_PER_NODE GPU(s) = $((GPU_BENCH_NODES * GPU_BENCH_TASKS_PER_NODE)) ranks
sbatch      : --nodes=$GPU_BENCH_NODES --ntasks-per-node=$GPU_BENCH_TASKS_PER_NODE \
--gres=gpu:$GPU_BENCH_TASKS_PER_NODE --time=$(gpu_bench_walltime_for "$GPU_BENCH_NODES")
results     : results/${bench//_/-}-${backend//_/-}-$topo/$bench
account     : $SBATCH_ACCOUNT   partition: $SBATCH_PARTITION

environment (each value, and the file that wrote it):
EOF
  # Resolve in a subshell so the caller's environment is untouched. This runs the
  # real sourcing chain, which needs the cluster's module system and $SCRATCH --
  # off Leonardo it fails, and saying so beats printing a plausible-looking
  # environment that no job would ever see.
  local out status=0
  # `out=$(...)` alone would trip `set -e` before the status could be read, and
  # 2>&1 must sit *inside* the substitution to be captured rather than printed.
  out=$({
    export GPU_BENCH_BENCHMARK="$bench" GPU_BENCH_BACKEND="$backend" GPU_BENCH_TOPOLOGY="$topo"
    export GPU_BENCH_STACK GPU_BENCH_RUNTIME GPU_BENCH_LAUNCHER
    export GPU_BENCH_NODES GPU_BENCH_TASKS_PER_NODE
    export GPU_BENCH_VERBOSE_ENV=1
    source "$ROOT/cluster/leonardo/provenance.sh"
    gpu_bench_origin_baseline
    source "$EXP/$bench/common.sh"
    gpu_bench_record_origin "cluster/leonardo/experiments/$bench/common.sh"
    source "$ROOT/cluster/leonardo/environment.sh" "$GPU_BENCH_STACK"
    gpu_bench_source_tracked "$ROOT/cluster/leonardo/runtime/$GPU_BENCH_RUNTIME.sh"
    gpu_bench_leonardo_print_env
  } 2>&1) || status=$?
  if [[ $status -ne 0 ]]; then
    echo "  unavailable here -- resolving the environment needs a Leonardo login node."
    echo "  everything above is resolved from the repository and is correct anywhere."
    echo "  reason: $(printf '%s' "$out" | tail -1)"
    return 0
  fi
  printf '%s\n' "$out" | sed 's/^/  /'
}

if [[ "$mode" == "single" ]]; then
  if [[ ${#args[@]} -lt 3 ]]; then
    echo "usage: tools/launch.sh <benchmark> <backend> <topology> [sbatch args...]" >&2
    echo "       tools/launch.sh --all [benchmark...]" >&2
    echo "       tools/launch.sh --explain <benchmark> <backend> <topology>" >&2
    exit 2
  fi
  [[ -f "$EXP/${args[0]}/common.sh" ]] \
    || { echo "error: no such benchmark: ${args[0]}" >&2; exit 2; }
  require_sbatch
  submit "${args[@]}"
  exit 0
fi

benchmarks=(${args[@]+"${args[@]}"})
if [[ ${#benchmarks[@]} -eq 0 ]]; then
  read -r -a benchmarks <<<"$GPU_BENCH_ALL_BENCHMARKS"
fi

matches() {  # matches <value> <space-separated-globs-or-empty>
  local value="$1" filters="$2" f
  [[ -z "$filters" ]] && return 0
  for f in $filters; do [[ "$value" == $f ]] && return 0; done
  return 1
}

require_sbatch

repeats=${GPU_BENCH_REPEATS:-1}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_BENCH_REPEATS must be a positive integer" >&2; exit 2; }

submitted=0
skipped=0
for bench in "${benchmarks[@]}"; do
  if [[ ! -f "$EXP/$bench/common.sh" ]]; then
    echo "warning: no such benchmark: $bench" >&2
    continue
  fi
  for backend in $(gpu_bench_matrix_backends "$bench"); do
    matches "$backend" "${GPU_BENCH_ONLY_BACKENDS:-}" || { skipped=$((skipped+1)); continue; }
    for topo in $(gpu_bench_matrix_topologies "$bench"); do
      matches "$topo" "${GPU_BENCH_ONLY_TOPOS:-}" || { skipped=$((skipped+1)); continue; }
      for repeat in $(seq "$repeats"); do
        if [[ "$dry_run" != "1" ]]; then
          label="$bench/$backend/$topo"
          [[ "$repeats" -gt 1 ]] && label="$label (repeat $repeat/$repeats)"
          echo "submitting: $label"
        fi
        submit "$bench" "$backend" "$topo"
        submitted=$((submitted+1))
      done
    done
  done
done

echo "---"
echo "submitted: $submitted   skipped (filtered): $skipped"
echo "watch queue: squeue -u \$USER"
echo "when drained, parse: python3 tools/benchscribe > RESULTS.md"
