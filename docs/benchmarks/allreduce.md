# Allreduce Benchmark

`allreduce` measures an element-wise float32 sum over GPU-resident buffers. Every
rank contributes the same element count and receives the complete result.

## Operation

For `P` ranks and `N` elements per rank, rank `r` initializes every source
element to `r + 1`. The expected value in every result element is:

```text
sum(r + 1, r = 0 .. P - 1) = P * (P + 1) / 2
```

Separate source and result buffers are allocated for the largest requested
message. `N` describes the logical per-rank payload, independent of the
collective algorithm selected by a library.

## Command-line contract

```text
<max_elements> [iterations] [warmup] [comma-separated message sizes]
```

Without an explicit size list, implementations sweep powers of two from one
element through `max_elements`. Arguments and explicit sizes must be positive,
and an explicit size cannot exceed the maximum.

Harness defaults, valid topologies, and environment overrides are documented in
the [experiment operations](../../cluster/harness/experiments/allreduce/README.md).

## Timing

Each sample contains one completed allreduce. Source initialization, the initial
rank barrier, validation, and reporting are outside timing.

Local samples are reduced element-wise with the maximum across ranks because a
collective completes when its slowest participant completes. Summary statistics
are calculated from that globally reduced series:

```text
global_sample[i] = max(local_sample[rank][i])
```

## Validation

Every rank copies the final result to the host and checks all elements against
`P * (P + 1) / 2` using the suite's float32 tolerance. Rank-local verdicts are
combined into one global validation result outside the timed interval.

## Bandwidth

The generic bandwidth uses the logical per-rank payload:

```text
bytes = N * sizeof(float) = 4 * N
gbytes_per_s = bytes / time_per_iteration
```

The report also includes conventional ring-equivalent bus bandwidth:

```text
bus_gbytes_per_s = gbytes_per_s * 2 * (P - 1) / P
```

The factor of two represents reduce-scatter plus allgather traffic. It is a
normalization convention, not a claim about the algorithm selected by the
communication library. Bus bandwidth is zero for the single-rank control.

## Completion boundaries

| Harness backend | Reduction model | Completion boundary |
| --- | --- | --- |
| `cuda_mpi` | `MPI_Allreduce` on CUDA buffers | Return of the blocking collective |
| `sycl_mpi` | `MPI_Allreduce` on USM buffers | Return of the blocking collective |
| `cuda_nccl` | `ncclAllReduce` | CUDA stream synchronization |
| `cuda_nvshmem` | NVSHMEM team float sum | Return of the host collective |
| `oshmpi` | Direct or host-staged SHMEM float sum | Collective return, including staging when selected |
| `sycl_oneccl` | oneCCL float sum | oneCCL event completion |
| `sycl_oneccl_oshmpi` | oneCCL float sum over OSHMPI | oneCCL event completion |

The OSHMPI direct path uses symmetric device buffers. Its explicit staged path
includes device-to-host and host-to-device copies in every timed operation and
reports the selected memory mode. These are distinct measured configurations,
not silent fallbacks.

## Related guides

- [Experiment operations](../../cluster/harness/experiments/allreduce/README.md):
  defaults, supported topologies, overrides, and launch examples
- [Backend implementations](../../src/README.md): library operations and build
  instructions
- [Support matrix](../reference/support-matrix.md): declared backend coverage
- [Output schema](../reference/output-schema.md): common fields and timing
  reduction rules
