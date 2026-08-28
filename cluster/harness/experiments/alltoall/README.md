# All-To-All

`alltoall` measures a personalized exchange: every rank sends a distinct
GPU-resident block of float32 elements to every rank and validates the complete
permutation locally.

## Interface

```text
<max_count_per_peer> [iterations] [warmup] [comma-separated counts]
```

The harness defaults to 65,536 maximum elements per peer, 100 timed iterations,
and 20 warmup iterations. Without `GPU_BENCH_MSG_SIZES`, counts sweep powers of
two. Buffers contain `ranks * count_per_peer` elements.

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`.

## Submit

```bash
cluster/harness/launch.sh alltoall cuda_mpi 1n4g
GPU_BENCH_MSG_SIZES=1,256,65536 \
  cluster/harness/launch.sh alltoall cuda_nccl 2n4g
```

Each rank includes a self block that never crosses a link. Reports therefore
include bus bandwidth corrected by `(ranks - 1) / ranks` alongside the raw
per-rank bandwidth. NCCL expresses the exchange as grouped point-to-point
operations; OSHMPI uses one-sided puts.
