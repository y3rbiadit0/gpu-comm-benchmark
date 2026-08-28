# Conjugate-Gradient Step Benchmark

`cg_step` measures the communication skeleton of one conjugate-gradient
iteration: a halo-dependent stencil followed by two scalar global sums. It is an
application pattern, not a complete solver.

## Operation

An `S x S` global grid is block-distributed by columns. Each rank stores its
interior columns plus west and east ghost columns. The decomposition is
nonperiodic. With multiple ranks, the first and last ranks have one neighbor
rather than two; the single-rank control has no halo neighbors.

Each step:

1. Packs and exchanges the boundary columns.
2. Updates ghost columns.
3. Computes the four-neighbor stencil
   `q(i,j) = 0.25 * (north + south + west + east)`, with zero outside the grid.
4. Computes local double-precision partials for `dot(p,q)` and `dot(q,q)`.
5. Performs two separate global sum reductions, each carrying one double.

The input field remains fixed between iterations. The benchmark does not perform
CG vector updates, residual calculation, or convergence testing.

## Command-line contract

```text
<grid_side> [iterations] [warmup]
```

All arguments must be positive. The validated experiment configurations use a
grid side at least as large as the rank count so every rank owns grid columns.

Harness defaults, valid topologies, and environment overrides are documented in
the [experiment operations](../../cluster/harness/experiments/cg_step/README.md).

## Timing

Each sample includes packing, halo exchange and completion, unpacking, stencil
execution, both local dot products, and both completed global reductions.
Initialization, the initial rank barrier, validation, timing reduction, and
reporting are outside timing.

Local samples are reduced element-wise with the maximum across ranks, and all
summary statistics are calculated from that globally reduced series:

```text
global_sample[i] = max(local_sample[rank][i])
```

## Validation

After timing, every rank validates its local stencil slab against a host
reference and checks the two global dot products. Float stencil values and
double reductions use the suite's corresponding relative tolerances. Rank-local
verdicts are combined into one global result.

## Metrics

The report uses a nominal two-column halo byte count:

```text
bytes = 2 * S * sizeof(float) = 8 * S
gbytes_per_s = bytes / time_per_iteration
```

This count excludes the two scalar reductions, does not adjust boundary ranks
with one neighbor, and remains nonzero for the single-rank control. It is an
application-level normalization, not exact link traffic.

## Completion boundaries

| Harness backend | Step model | Completion boundary |
| --- | --- | --- |
| `cuda_mpi` | `MPI_Sendrecv` halo and two `MPI_Allreduce` calls | Blocking communication plus CUDA synchronization |
| `sycl_mpi` | MPI halo and reductions over USM | Blocking communication plus SYCL event completion |
| `cuda_nccl` | Grouped halo and two NCCL allreduces | CUDA stream synchronization |
| `cuda_nvshmem` | Put-based halo, barrier, and two team reductions | CUDA synchronization and SHMEM completion |
| `oshmpi` | Put-based halo and direct or staged reductions | SHMEM completion, including staging when selected |
| `sycl_oneccl` | Grouped halo and two oneCCL allreduces | oneCCL event completion |
| `sycl_oneccl_oshmpi` | Same oneCCL operations over OSHMPI | oneCCL event completion |

OSHMPI reports whether the two reductions used host staging or symmetric device
memory. Staging copies are included in the timed step. oneCCL point-to-point
support depends on the configured transport.

## Related guides

- [Experiment operations](../../cluster/harness/experiments/cg_step/README.md):
  defaults, supported topologies, overrides, and launch examples
- [Backend implementations](../../src/README.md): library operations and build
  instructions
- [Support matrix](../reference/support-matrix.md): declared backend coverage
- [Output schema](../reference/output-schema.md): common fields and timing
  reduction rules
