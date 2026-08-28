# CUDA + NCCL

This backend uses NCCL for GPU-resident communication and MPI only for process
launch, NCCL bootstrap, and result collection.

## Implemented operations

| Benchmark | NCCL operation |
| --- | --- |
| `pingpong` | Matched point-to-point send and receive |
| `halo_1d` | Grouped neighbor sends and receives |
| `allreduce` | `ncclAllReduce` |
| `alltoall` | Grouped send and receive to every peer |
| `cg_step` | Grouped halo exchange and two allreduces |
| `moe` | Variable-count grouped dispatch and combine |

NCCL has no dedicated all-to-all or halo collective; grouped point-to-point
operations are its native expression of those patterns.

## Build

Leonardo provides NCCL through NVHPC and uses the `leonardo-cuda-nccl` preset:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-nccl
cmake --build --preset leonardo-cuda-nccl
```

For another cluster, provide `NCCL_INCLUDE_DIR` and `NCCL_LIBRARY` in a CMake
preset and register the resulting backend with the harness.

For benchmark semantics, see the
[benchmark contracts](../../../docs/README.md#benchmark-contracts). For
arguments, defaults, topologies, and launch examples, see the
[experiment operations](../../../cluster/harness/README.md#experiment-operations).
