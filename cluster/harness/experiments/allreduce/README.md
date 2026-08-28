# Allreduce

`allreduce` measures an element-wise float32 sum over GPU-resident buffers. Every
rank contributes the same element count and receives the result.

The complete operation, timing, validation, and bandwidth contract is in
[`docs/benchmarks/allreduce.md`](../../../../docs/benchmarks/allreduce.md).

## Configuration

```text
<max_elements> [iterations] [warmup] [comma-separated message sizes]
```

The harness defaults to 4,194,304 maximum elements, 100 timed iterations, and 20
warmup iterations. Without `GPU_BENCH_MSG_SIZES`, each implementation sweeps
powers of two.

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`. The single-rank case is a control rather than a network measurement.

## Run

```bash
cluster/harness/launch.sh allreduce cuda_mpi 1n4g
GPU_BENCH_MSG_SIZES=1,1024,4194304 \
  cluster/harness/launch.sh allreduce cuda_nccl 2n4g
```

On Leonardo, OSHMPI defaults to direct device-buffer reduction and requires UCC.
Set `GPU_BENCH_OSHMPI_ALLREDUCE_MEM=staged` to measure its explicit host-staging
fallback; the selected path is included in the result.
