# Conjugate-Gradient Step

`cg_step` extracts the communication shape of one conjugate-gradient iteration:
a five-point stencil over a column-slab decomposition, one neighbor halo
exchange, and two scalar global sums. It is an application pattern, not a full
solver.

The complete operation, timing, validation, and metric contract is in
[`docs/benchmarks/cg-step.md`](../../../../docs/benchmarks/cg-step.md).

## Configuration

```text
<grid_side> [iterations] [warmup]
```

The harness defaults to a `512 x 512` global grid, 50 timed iterations, and 10
warmup iterations. The small grid keeps communication visible relative to the
stencil kernel.

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`. At one rank, the benchmark provides a compute and local-reduction
baseline.

## Run

```bash
cluster/harness/launch.sh cg_step cuda_mpi 1n4g
GPU_BENCH_N=1024 GPU_BENCH_NTRIALS=1 \
  cluster/harness/launch.sh cg_step cuda_nvshmem 2n4g
```

The reductions contain one double each. Leonardo disables UCC for this
small-message operation; OSHMPI consequently defaults to its staged scalar path.
Set both `OMPI_MCA_coll_ucc_enable=1` and
`GPU_BENCH_OSHMPI_CG_REDUCE_MEM=device` to measure its device path explicitly.
