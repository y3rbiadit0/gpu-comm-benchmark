# Halo 1D Benchmark

`halo_1d` isolates neighbor communication in a periodic one-dimensional ring.
It compares two-sided, collective-library point-to-point, and one-sided models
without including stencil computation.

## Operation

For rank `r` among `P` ranks:

```text
left  = (r - 1 + P) % P
right = (r + 1) % P
```

Each iteration sends one halo of `H` float32 elements to each neighbor and
receives one from each neighbor. Buffers are GPU resident and allocated for the
largest requested halo, not for a global application domain.

The two outgoing regions contain deterministic rank- and boundary-dependent markers.
After a measured batch, each rank validates both received regions locally. No
validation gather is included in the timed interval.

## Interface

```text
<max_halo_elements> [iterations] [warmup] [comma-separated halo sizes]
```

Without an explicit size list, implementations sweep powers of two from one
element through `max_halo_elements`. The benchmark requires at least two ranks.

## Timing Cases

Every halo size produces two cases:

| Case | Batch length | Purpose |
| --- | ---: | --- |
| `isolated` | 1 exchange | Expose submission and completion latency |
| `steady` | Requested iterations | Measure amortized queued or persistent work |

Each rank records completed-batch samples. Samples are reduced element-wise with
the maximum across ranks, because an exchange completes when its slowest
participant completes. Reported per-iteration times divide each batch sample by
its batch length before summary statistics are calculated.

The harness defaults to 100 isolated samples and 10 steady samples. Override
them with `GPU_BENCH_ISOLATED_SAMPLES` and `GPU_BENCH_BATCH_SAMPLES`.

## Bandwidth

The report counts aggregate bus traffic: two sends and two receives per rank.

```text
bytes_per_iteration = 4 * H * sizeof(float) = 16 * H
gbytes_per_s = bytes_per_iteration / time_per_iteration
```

This convention is intentionally different from ping-pong's one-way bandwidth.
Do not compare the two as if they represented the same byte count.

## Implementations

| Harness backend | Exchange model | Completion boundary |
| --- | --- | --- |
| `cuda_mpi` | Persistent CUDA-aware MPI sends and receives | `MPI_Waitall` per exchange |
| `sycl_mpi` | Persistent MPI sends and receives over USM | `MPI_Waitall` per exchange |
| `cuda_nccl` | Grouped `ncclSend`/`ncclRecv` | Stream completion per batch |
| `cuda_nvshmem` | Cooperative multi-block puts and neighbor signals | Device-side signal dependency |
| `oshmpi` | Nonblocking one-sided puts | Quiet, device sync, and global barrier |
| `sycl_oneccl` | Grouped `ccl::send`/`ccl::recv` | oneCCL event completion per batch |

These completion models are part of each measured implementation. In particular,
the OSHMPI global barrier is not equivalent to neighbor signaling and must remain
visible when interpreting comparisons.

oneCCL point-to-point support depends on the configured transport. Unexpected
transport failures are benchmark errors rather than timing results; check the
[support matrix](../reference/support-matrix.md) before a large sweep.

## Build And Run

Build instructions are cluster specific. On Leonardo:

```bash
make leonardo-cuda
cluster/harness/launch.sh halo_1d cuda_mpi 1n4g
```

Run a targeted sweep with:

```bash
GPU_BENCH_MSG_SIZES=1,16,256,4096 GPU_BENCH_NTRIALS=1 \
  cluster/harness/launch.sh halo_1d cuda_nvshmem 2n4g
```

See the [experiment guide](../../cluster/harness/experiments/halo_1d/README.md)
for defaults and valid topologies, and the
[analysis methodology](../analysis/halo-1d-methodology.md) for comparison and
profiling guidance.
