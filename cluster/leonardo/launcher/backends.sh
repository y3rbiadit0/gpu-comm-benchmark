#!/usr/bin/env bash
# Backend registry.
#
# Every field here was a hand-copied line in each of 230 near-identical job
# scripts. It is a backend constant, so it belongs in one table.
#
# Fields, colon-separated: stack:runtime:launcher:preset:bindir:binary_prefix
#
#   stack           cuda | sycl -- selects the toolchain module set
#   runtime         cluster/leonardo/runtime/<runtime>.sh, the env for this backend
#   launcher        srun | mpirun -- oneCCL needs mpirun; see runtime/oneccl-*.sh
#   preset          CMake preset, so build/<preset>/... locates the binary
#   bindir          where CMake puts that preset's binaries, mirroring src/
#   binary_prefix   target name prefix; the binary is <prefix>_<benchmark>
#
# sycl_oneccl and sycl_oneccl_oshmpi build the *same* sources into the same
# target names from two different presets -- the difference is which transport
# oneCCL was configured against -- so they share a binary_prefix and differ in
# preset and runtime.
GPU_BENCH_BACKENDS=(
  "cuda_mpi:cuda:mpi-cuda:srun:leonardo-cuda-mpi:src/mpi/cuda:cuda_mpi"
  "cuda_nccl:cuda:mpi-cuda:srun:leonardo-cuda-nccl:src/xccl/cuda:cuda_nccl"
  "cuda_nvshmem:cuda:nvshmem:srun:leonardo-cuda-nvshmem:src/shmem/nvshmem:cuda_nvshmem"
  "oshmpi:cuda:oshmpi:srun:leonardo-oshmpi:src/shmem/oshmpi:oshmpi"
  "sycl_mpi:sycl:sycl-mpi:srun:leonardo-sycl-mpi:src/mpi/sycl:sycl_mpi"
  "sycl_oneccl:sycl:oneccl-nccl:mpirun:leonardo-sycl-oneccl:src/xccl/sycl:sycl_oneccl"
  "sycl_oneccl_oshmpi:sycl:oneccl-oshmpi:mpirun:leonardo-sycl-oneccl-oshmpi:src/xccl/sycl:sycl_oneccl"
)

# gpu_bench_backend_fields <backend> -> sets GPU_BENCH_STACK, _RUNTIME,
# _LAUNCHER, _PRESET, _BINARY_PREFIX. Fails loudly on an unknown backend rather
# than submitting a job that would die after the allocation is granted.
gpu_bench_backend_fields() {
  local want="$1" entry name
  for entry in "${GPU_BENCH_BACKENDS[@]}"; do
    name="${entry%%:*}"
    if [[ "$name" == "$want" ]]; then
      IFS=: read -r _ GPU_BENCH_STACK GPU_BENCH_RUNTIME GPU_BENCH_LAUNCHER \
                     GPU_BENCH_PRESET GPU_BENCH_BINDIR GPU_BENCH_BINARY_PREFIX <<<"$entry"
      return 0
    fi
  done
  echo "error: unknown backend '$want'; known: $(gpu_bench_backend_names)" >&2
  return 1
}

gpu_bench_backend_names() {
  local entry
  for entry in "${GPU_BENCH_BACKENDS[@]}"; do printf '%s ' "${entry%%:*}"; done
}

# gpu_bench_topology_fields <topology> -> sets GPU_BENCH_NODES and
# GPU_BENCH_TASKS_PER_NODE. "2n4g" is 2 nodes of 4 GPUs; one rank drives one GPU.
gpu_bench_topology_fields() {
  local topo="$1"
  if [[ ! "$topo" =~ ^([1-9][0-9]*)n([1-9][0-9]*)g$ ]]; then
    echo "error: malformed topology '$topo' (expected <nodes>n<gpus_per_node>g)" >&2
    return 1
  fi
  GPU_BENCH_NODES="${BASH_REMATCH[1]}"
  GPU_BENCH_TASKS_PER_NODE="${BASH_REMATCH[2]}"
}

# Walltime scales with node count: more ranks means more collective work per
# sample, and a job that times out wastes the whole allocation. Short limits are
# also better backfill candidates, so this stays as tight as it safely can.
gpu_bench_walltime_for() {
  local nodes="$1"
  if   (( nodes >= 8 )); then echo "00:20:00"
  elif (( nodes >= 4 )); then echo "00:15:00"
  else                        echo "00:10:00"
  fi
}
