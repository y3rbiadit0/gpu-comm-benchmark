# CUDA + NVSHMEM

This backend uses NVSHMEM symmetric GPU buffers and one-sided or team operations.

## Implemented operations

| Benchmark | NVSHMEM operation |
| --- | --- |
| `pingpong` | Persistent device-initiated puts and completion signals |
| `halo_1d` | Cooperative multi-block puts with neighbor signals |
| `allreduce` | `nvshmem_float_sum_reduce` |
| `alltoall` | `nvshmem_float_alltoall` |
| `cg_step` | Host-driven puts, barrier, and two double reductions |
| `moe` | Variable-count puts with quiet and barrier completion |

The ping-pong and halo implementations use cooperative kernels so communication
can be device initiated. On proxy-based inter-node transports, their grid size
is capped by default; `GPU_BENCH_NVSHMEM_MAX_BLOCKS` overrides the cap.

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
