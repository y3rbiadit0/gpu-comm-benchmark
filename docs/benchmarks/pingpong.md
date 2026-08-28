# Ping-Pong Benchmark

`pingpong` measures one-way latency and bandwidth between exactly two
GPU-resident endpoints. Rank 0 sends a payload to rank 1, which returns it
unchanged.

## Operation

For a message of `N` float32 elements, rank 0 sends its buffer to rank 1 and
waits for the reply. Rank 1 receives the message and sends the same payload
back. The benchmark requires exactly two ranks, and rank 0 is always the timing
initiator.

The send buffer contains deterministic values derived from the element index.
Separate send and receive buffers are allocated for the largest requested
message.

## Command-line contract

```text
<max_elements> [iterations] [warmup] [comma-separated message sizes]
```

Without an explicit size list, implementations sweep powers of two from one
element through `max_elements`. Arguments and explicit sizes must be positive,
and an explicit size cannot exceed the maximum.

Harness defaults, valid topologies, and environment overrides are documented in
the [experiment operations](../../cluster/harness/experiments/pingpong/README.md).

## Timing

Rank 0 records the duration of each completed round trip. Rank 1 participates in
the exchange but does not contribute a timing sample. The reported one-way time
and every distribution statistic are half of the corresponding round-trip
value:

```text
time_one_way = time_round_trip / 2
```

Initialization, the initial rank synchronization, validation, and reporting are
outside the timed interval. The NVSHMEM implementation is an exception to the
per-operation sampling model: it measures one persistent kernel containing all
timed exchanges and amortizes that duration over the iteration count.

## Validation

After the timed exchanges for a size, rank 0 copies the returned buffer to the
host and compares it with the original payload using the suite's float32
tolerance. Validation is not included in timing.

## Bandwidth

`bytes` is the payload in one direction, not the sum of both legs:

```text
bytes = N * sizeof(float) = 4 * N
gbytes_per_s = bytes / time_one_way
```

The result is decimal GB/s. No bus-bandwidth correction is reported.

## Completion boundaries

| Harness backend | Exchange model | Completion boundary |
| --- | --- | --- |
| `cuda_mpi` | Blocking CUDA-aware MPI send and receive | Return of the blocking calls |
| `sycl_mpi` | Blocking MPI send and receive over USM | Return of the blocking calls |
| `cuda_nccl` | NCCL point-to-point send and receive | Stream synchronization after each leg |
| `cuda_nvshmem` | Persistent device-initiated puts and signals | Completion of the timed cooperative kernel |
| `oshmpi` | One-sided puts with handshake barriers | Quiet, device synchronization, and barriers |
| `sycl_oneccl` | oneCCL point-to-point send and receive | oneCCL event completion |

These completion costs are part of the measured implementation. In particular,
the OSHMPI barriers are not equivalent to direct point-to-point signaling, and
the NVSHMEM persistent-kernel result has different sampling granularity from the
other backends.

oneCCL point-to-point support depends on the configured transport. Check the
[support matrix](../reference/support-matrix.md) before a large sweep.

## Related guides

- [Experiment operations](../../cluster/harness/experiments/pingpong/README.md):
  defaults, supported topologies, overrides, and launch examples
- [Backend implementations](../../src/README.md): library operations and build
  instructions
- [Support matrix](../reference/support-matrix.md): declared backend coverage
- [Output schema](../reference/output-schema.md): common fields and timing
  reduction rules
