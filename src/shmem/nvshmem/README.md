# CUDA + NVSHMEM

This backend uses NVSHMEM symmetric GPU buffers and one-sided or team operations.

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

## Build And Run

Leonardo provides NVSHMEM through NVHPC and uses the
`leonardo-cuda-nvshmem` preset:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-nvshmem
cmake --build --preset leonardo-cuda-nvshmem
cluster/harness/launch.sh halo_1d cuda_nvshmem 1n4g
```

The [halo analysis](../../../docs/analysis/halo-1d-methodology.md) explains the
device-initiated timing and transport considerations.
