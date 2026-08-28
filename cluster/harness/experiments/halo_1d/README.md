# Halo 1D

`halo_1d` measures a communication-only periodic ring. Each rank exchanges a
GPU-resident halo with both neighbors over a swept halo width.

The complete buffer, timing, bandwidth, validation, and backend contract is in
[`docs/benchmarks/halo-1d.md`](../../../../docs/benchmarks/halo-1d.md).

## Interface

```text
<max_halo_elements> [iterations] [warmup] [comma-separated halo sizes]
```

The harness defaults to 1,048,576 maximum float elements, 100 timed iterations,
20 warmup iterations, powers-of-two halo sizes, 10 steady-state batch samples,
and 100 isolated samples.

Valid topologies are `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and `8n4g`.
The single-rank control is excluded because a periodic ring needs at least two
ranks.

## Submit

```bash
cluster/harness/launch.sh halo_1d cuda_mpi 1n4g
GPU_BENCH_MSG_SIZES=1,256,65536 GPU_BENCH_NTRIALS=1 \
  cluster/harness/launch.sh halo_1d cuda_nvshmem 2n4g
```

`GPU_BENCH_BATCH_SAMPLES` and `GPU_BENCH_ISOLATED_SAMPLES` override the sample
counts. Profiling instructions are in the [harness guide](../../README.md).
