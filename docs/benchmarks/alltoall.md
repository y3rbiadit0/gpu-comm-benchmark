# All-To-All Benchmark

`alltoall` measures a personalized exchange in which every rank sends a distinct
GPU-resident float32 block to every rank and receives one block from every rank.

## Operation

For `P` ranks and a per-peer count `C`, each send and receive buffer contains
`P * C` float32 elements arranged as `P` contiguous peer blocks. Source rank `r`
fills the block for destination `d` with `r * P + d`. Destination rank `d`
therefore expects block `r` to contain `r * P + d`.

The exchange includes the self block. `C`, rather than the complete buffer
length, is reported as the benchmark size.

## Command-line contract

```text
<max_count_per_peer> [iterations] [warmup] [comma-separated counts]
```

Without an explicit count list, implementations sweep powers of two from one
element through `max_count_per_peer`. Arguments and explicit counts must be
positive, and an explicit count cannot exceed the maximum.

Harness defaults, valid topologies, and environment overrides are documented in
the [experiment operations](../../cluster/harness/experiments/alltoall/README.md).

## Timing

Each sample contains one completed personalized exchange. Buffer generation,
host-to-device initialization, the initial rank barrier, validation, and
reporting are outside timing.

Local samples are reduced element-wise with the maximum across ranks, and all
summary statistics are calculated from that globally reduced series:

```text
global_sample[i] = max(local_sample[rank][i])
```

## Validation

Every rank copies the complete receive buffer to the host and validates every
source block against the expected permutation. Rank-local verdicts are combined
into one global result outside the timed interval.

## Bandwidth

Raw bandwidth counts the entire per-rank send buffer, including the self block:

```text
bytes = P * C * sizeof(float) = 4 * P * C
gbytes_per_s = bytes / time_per_iteration
```

Because only `P - 1` of the `P` blocks cross a communication link, the report
also includes:

```text
bus_gbytes_per_s = gbytes_per_s * (P - 1) / P
```

Bus bandwidth is zero for the single-rank control. Unlike allreduce, this
convention does not apply an additional factor of two.

## Completion boundaries

| Harness backend | Exchange model | Completion boundary |
| --- | --- | --- |
| `cuda_mpi` | `MPI_Alltoall` on CUDA buffers | Return of the blocking collective |
| `sycl_mpi` | `MPI_Alltoall` on USM buffers | Return of the blocking collective |
| `cuda_nccl` | Grouped send and receive to every peer | CUDA stream synchronization |
| `cuda_nvshmem` | NVSHMEM team all-to-all | Return of the host collective |
| `oshmpi` | Per-peer puts and local self copy | Quiet, device synchronization, and global barrier |
| `sycl_oneccl` | Native oneCCL all-to-all | oneCCL event completion |
| `sycl_oneccl_oshmpi` | oneCCL all-to-all over OSHMPI | oneCCL event completion |

NCCL and OSHMPI express the same personalized exchange through native
point-to-point or one-sided primitives. Their completion costs remain inside the
timed operation.

## Related guides

- [Experiment operations](../../cluster/harness/experiments/alltoall/README.md):
  defaults, supported topologies, overrides, and launch examples
- [Backend implementations](../../src/README.md): library operations and build
  instructions
- [Support matrix](../reference/support-matrix.md): declared backend coverage
- [Output schema](../reference/output-schema.md): common fields and timing
  reduction rules
