# Leonardo MoE Experiments

These jobs benchmark a top-1 mixture-of-experts exchange over float32 token payloads. Each
rank starts with `GPU_BENCH_N` tokens of width `GPU_BENCH_HIDDEN`; one expert is assigned to each rank.
Every timed iteration dispatches variable-sized token blocks to their expert ranks and then
combines them back to their source ranks. This is a skew-sensitive application pattern, not
the dense equal-count exchange measured by `alltoall`.

The default run reports all three deterministic routing cases:

| Case | Routing |
| --- | --- |
| `uniform` | Tokens are hashed uniformly across expert ranks. |
| `locality80` | 80% of each rank's tokens remain local; the rest are hashed across other experts. |
| `hotspot80` | 80% of all tokens target expert rank 0; the rest target other experts. |

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
tools/launch.sh moe cuda_mpi 1n4g
tools/launch.sh moe cuda_nccl 1n4g
tools/launch.sh moe cuda_nvshmem 1n4g
tools/launch.sh moe oshmpi 1n4g
tools/launch.sh moe sycl_mpi 1n4g
tools/launch.sh moe sycl_oneccl 1n4g
```

## Overrides

```bash
GPU_BENCH_N=16384                         # tokens per rank
GPU_BENCH_HIDDEN=256                      # float32 values per token
GPU_BENCH_ITERS=100                       # timed dispatch+combine iterations per case
GPU_BENCH_WARMUP=20                       # untimed iterations per case
GPU_BENCH_ROUTINGS=uniform,hotspot80      # optional subset; omitted runs all internal cases
GPU_BENCH_NTRIALS=3                       # job-level repeats
```

The binaries accept `<tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]`.
`routing_cases` is an optional comma-separated list drawn from `uniform`, `locality80`, and
`hotspot80`; omitting it lets the binaries run all three cases internally. oneCCL jobs use
`mpirun`. If the NCCL-backed oneCCL fork does not implement point-to-point operations, each
remaining case emits `status=NOT_IMPLEMENTED reason=point_to_point validation=SKIP` instead
of benchmark timings.

The Leonardo fork dispatches public groups through `group_impl`, uses native
NCCL groups, and defers operation events until the outermost group ends. Grouped
point-to-point is enabled by default. The runtime capability probe still reports
`NOT_IMPLEMENTED` if the backend does not provide `ccl::send`/`ccl::recv`.

Each result reports expert imbalance and useful throughput for both variable-count phases.
Useful bytes per rank per iteration are `2 * tokens * hidden * sizeof(float)` for dispatch
plus combine.
