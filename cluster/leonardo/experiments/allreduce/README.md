# Leonardo Allreduce Experiments

These jobs benchmark a float32 sum allreduce over device-resident buffers (host-symmetric
buffers for OSHMPI). Each rank contributes `n` elements and receives the elementwise sum.
By default each binary sweeps powers of two from 1 element through `GPU_BENCH_N`; `GPU_BENCH_MSG_SIZES`
selects explicit comma-separated sizes. Build setup is in
[`cluster/leonardo/README.md`](../../README.md).

## Topologies

| Script | Nodes | GPUs/node | Path |
| --- | ---: | ---: | --- |
| `1n1g.sh` | 1 | 1 | single-GPU baseline |
| `1n2g.sh` | 1 | 2 | intra-node NVLink |
| `1n4g.sh` | 1 | 4 | intra-node NVLink |
| `2n1g.sh` | 2 | 1 | inter-node InfiniBand |
| `2n4g.sh` | 2 | 4 | mixed intra- and inter-node |

## Submit

```bash
sbatch cluster/leonardo/experiments/allreduce/cuda_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/cuda_nccl/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/cuda_nvshmem/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/oshmpi/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/sycl_mpi/1n4g.sh
sbatch cluster/leonardo/experiments/allreduce/sycl_oneccl/1n4g.sh
```

## Overrides

```bash
GPU_BENCH_N=4194304                  # maximum element count; default sweep is powers of two
GPU_BENCH_MSG_SIZES=1,1024,1048576  # optional explicit element counts
GPU_BENCH_ITERS=100                 # timed iterations per size
GPU_BENCH_WARMUP=20                 # untimed iterations per size
GPU_BENCH_NTRIALS=3                 # job-level repeats
```

The binaries accept `<max_elements> [iterations] [warmup] [message_sizes]`. oneCCL jobs
use `mpirun`; the other launchers match the existing all-to-all conventions.

## Bandwidth

Each size emits one standardized result line. `gbytes_per_s` is algorithm bandwidth,
`bytes / time`; `bus_gbytes_per_s` normalizes allreduce traffic as
`gbytes_per_s * 2 * (ranks - 1) / ranks`, making collective transport efficiency easier
to compare across rank counts. Use `usec` for small-message latency.
