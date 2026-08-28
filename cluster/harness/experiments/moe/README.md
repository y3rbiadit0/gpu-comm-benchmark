# Mixture Of Experts

`moe` models one top-1 mixture-of-experts communication step. Tokens are routed
to owner ranks with variable counts, then returned through the inverse exchange.
It measures dispatch and combine together.

The complete operation, timing, validation, and metric contract is in
[`docs/benchmarks/moe.md`](../../../../docs/benchmarks/moe.md).

## Configuration

```text
<tokens_per_rank> [hidden] [iterations] [warmup] [comma-separated routing cases]
```

The harness defaults to 16,384 tokens per rank, hidden width 256, 100 timed
iterations, and 20 warmup iterations. It runs three deterministic distributions:

| Case | Distribution |
| --- | --- |
| `uniform` | Tokens are assigned by a uniform hash across ranks |
| `locality80` | Most tokens remain on the source rank |
| `hotspot80` | Most tokens target one expert rank |

Declared topologies are `1n1g`, `1n2g`, `1n4g`, `2n1g`, `2n4g`, `4n4g`, and
`8n4g`.

## Run

```bash
cluster/harness/launch.sh moe cuda_mpi 1n4g
GPU_BENCH_HIDDEN=512 GPU_BENCH_ROUTINGS=uniform,hotspot80 \
  cluster/harness/launch.sh moe cuda_nccl 2n4g
```

The routing case and hidden width are part of the analysis grouping key. A
recognized missing oneCCL point-to-point capability is reported as
`NOT_IMPLEMENTED` rather than as timing data.
