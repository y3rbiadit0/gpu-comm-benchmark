#!/usr/bin/env bash
# Which file sets this environment variable, and which one wins?
#
# The environment is layered, every layer using ${VAR:-default}, so the FIRST
# file to run wins and later ones leave it alone. That order is fixed and known,
# which means the answer can be read straight out of the files -- no running, no
# cluster, no snapshotting the environment before and after each source.
#
# gpu_bench_where_set <benchmark> <stack> <runtime>
#
# Prints one block per variable, definitions in execution order, winner first.

gpu_bench_where_set() {
  local bench="$1" stack="$2" runtime="$3"
  local root="${GPU_BENCH_PROJECT_ROOT:?}" files=() f

  # Execution order, earliest first. The harness contributes the benchmark shim
  # and the shared run loop; everything after comes from the cluster, which is
  # the only thing that knows its own module and runtime files.
  files+=("$root/cluster/harness/experiments/$bench/common.sh")
  files+=("$root/cluster/harness/experiments/common.sh")
  while IFS= read -r f; do files+=("$f"); done \
    < <(gpu_bench_cluster_env_files "$stack" "$runtime")

  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    awk -v file="${f#"$root/"}" '
      # export NAME=${NAME:-VALUE}   -- the overridable form
      match($0, /^[[:space:]]*export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=\$\{[A-Za-z_][A-Za-z0-9_]*:-/) {
        split($0, a, "="); name = a[1]; sub(/^[[:space:]]*export[[:space:]]+/, "", name)
        value = $0; sub(/^[^=]*=\$\{[^:]*:-/, "", value); sub(/\}[[:space:]]*$/, "", value)
        printf "%s\t%s\t%s\n", name, value, file; next
      }
      # export NAME=VALUE            -- unconditional, wins outright
      match($0, /^[[:space:]]*export[[:space:]]+[A-Za-z_][A-Za-z0-9_]*=[^$]/) {
        split($0, a, "="); name = a[1]; sub(/^[[:space:]]*export[[:space:]]+/, "", name)
        value = $0; sub(/^[^=]*=/, "", value)
        printf "%s\t%s\t%s (unconditional)\n", name, value, file; next
      }
    ' "$f"
  done | awk -F'\t' '
    { n = ++count[$1]; val[$1, n] = $2; src[$1, n] = $3; if (n == 1) order[++k] = $1 }
    END {
      for (i = 1; i <= k; i++) {
        name = order[i]
        for (j = 1; j <= count[name]; j++) {
          tag = (j == 1) ? "WINS" : "    "
          printf "  %-34s %-14s %s  %s\n", (j == 1 ? name : ""), val[name, j], tag, src[name, j]
        }
      }
    }'
}
