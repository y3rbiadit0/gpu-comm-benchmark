# Ping-Pong

`pingpong` measures one-way latency and bandwidth between exactly two GPUs. The
initiator sends a GPU-resident buffer and waits for the reply; reports divide the
measured round trip by two.

The complete operation, timing, validation, and bandwidth contract is in
[`docs/benchmarks/pingpong.md`](../../../../docs/benchmarks/pingpong.md).

## Configuration

```text
<max_elements> [iterations] [warmup] [comma-separated message sizes]
```

The default run uses 4,194,304 maximum float elements, 100 timed iterations, and
20 warmup iterations. Without an explicit list, each binary sweeps powers of two
from one element to the maximum.

Only `1n2g` and `2n1g` are valid because the benchmark requires two endpoints.
All standard backends are declared; see the
[support matrix](../../../../docs/reference/support-matrix.md).

## Run

```bash
cluster/harness/launch.sh pingpong cuda_mpi 1n2g
GPU_BENCH_MSG_SIZES=1,1024,1048576 \
  cluster/harness/launch.sh pingpong cuda_nccl 2n1g
```

Each message size emits one record using the
[standard output schema](../../../../docs/reference/output-schema.md).
