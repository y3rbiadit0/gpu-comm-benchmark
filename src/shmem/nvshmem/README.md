# CUDA + NVSHMEM

This backend uses NVSHMEM symmetric GPU buffers and one-sided or team operations.

## Implemented operations

| Benchmark | NVSHMEM operation |
| --- | --- |
| `pingpong` | Persistent device-initiated puts and completion signals |
| `halo_1d` | Cooperative multi-block puts with neighbor signals |
| `allreduce` | `nvshmem_float_sum_reduce` |
| `alltoall` | `nvshmem_float_alltoall` |
| `cg_step` | Cooperative halo puts with neighbour signals, one fused double reduction |
| `moe` | Multi-block puts with per-peer dispatch/combine signals, split per phase |

The ping-pong, halo, and CG-step implementations use cooperative kernels so
communication can be device initiated. On proxy-based inter-node transports
their grid size is capped by default, because each block there sends to a
*different* peer and every block's remote op is separately proxied.

`moe` is capped too, but at 64 rather than 8: its blocks split one peer's
payload, so the limit is how many concurrent proxied block-puts the host proxy
absorbs, not how many peers are in flight. A 2n1g sweep put the optimum at 64
(1656 / 877 / 2439 usec for uniform / locality80 / hotspot80) and measured the
full 992-block grid at roughly 1.9x slower with 50x the run-to-run variance.
`GPU_BENCH_NVSHMEM_MAX_BLOCKS` sets the cap for any of them.

## Build

Leonardo provides NVSHMEM through NVHPC and uses the
`leonardo-cuda-nvshmem` preset:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-nvshmem
cmake --build --preset leonardo-cuda-nvshmem
```

The [halo analysis](../../../docs/analysis/halo-1d-methodology.md) explains the
device-initiated timing and transport considerations.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
