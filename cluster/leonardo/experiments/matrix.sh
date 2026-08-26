#!/usr/bin/env bash
# Which (benchmark, backend, topology) combinations are valid.
#
# This used to live implicitly in which directories happened to exist, where it
# was invisible to review: that pingpong has no 4-rank case, or that halo_1d has
# no single-rank case, could only be discovered by listing the tree. Declared
# here, adding a topology is a one-line edit instead of generating 64 files.
#
# Both lists are ordered: smallest topology first, so submissions go out in an
# order where a cheap job fails before an expensive one does.

GPU_BENCH_ALL_BACKENDS="cuda_mpi cuda_nccl cuda_nvshmem oshmpi sycl_mpi sycl_oneccl"

# pingpong measures one-way point-to-point latency between exactly two ranks, so
# it has no multi-GPU-per-node or single-rank case.
pingpong_TOPOLOGIES="1n2g 2n1g"
pingpong_BACKENDS="$GPU_BENCH_ALL_BACKENDS"

# halo_1d is a periodic ring: it needs at least two ranks, so no 1n1g.
halo_1d_TOPOLOGIES="1n2g 1n4g 2n1g 2n4g 4n4g 8n4g"
halo_1d_BACKENDS="$GPU_BENCH_ALL_BACKENDS"

# The collectives keep 1n1g as a control: at one rank nothing crosses a link, so
# it is the floor the multi-rank numbers are read against. It is excluded from
# the figures by default (see tools/plot).
allreduce_TOPOLOGIES="1n1g 1n2g 1n4g 2n1g 2n4g 4n4g 8n4g"
allreduce_BACKENDS="$GPU_BENCH_ALL_BACKENDS sycl_oneccl_oshmpi"

alltoall_TOPOLOGIES="1n1g 1n2g 1n4g 2n1g 2n4g 4n4g 8n4g"
alltoall_BACKENDS="$GPU_BENCH_ALL_BACKENDS sycl_oneccl_oshmpi"

moe_TOPOLOGIES="1n1g 1n2g 1n4g 2n1g 2n4g 4n4g 8n4g"
moe_BACKENDS="$GPU_BENCH_ALL_BACKENDS"

# cg_step's 1n1g case is the compute baseline the communication cost is measured
# against, so it is a genuine data point here rather than only a control.
cg_step_TOPOLOGIES="1n1g 1n2g 1n4g 2n1g 2n4g 4n4g 8n4g"
cg_step_BACKENDS="$GPU_BENCH_ALL_BACKENDS sycl_oneccl_oshmpi"

# Order matters only for readability of `tools/launch.sh --all` output.
GPU_BENCH_ALL_BENCHMARKS="pingpong halo_1d allreduce alltoall moe cg_step"

gpu_bench_matrix_topologies() {
  local bench="$1" var="${1}_TOPOLOGIES"
  [[ -n "${!var:-}" ]] || { echo "error: no topologies declared for '$bench'" >&2; return 1; }
  echo "${!var}"
}

gpu_bench_matrix_backends() {
  local bench="$1" var="${1}_BACKENDS"
  [[ -n "${!var:-}" ]] || { echo "error: no backends declared for '$bench'" >&2; return 1; }
  echo "${!var}"
}
