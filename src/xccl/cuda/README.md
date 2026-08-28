# CUDA + NCCL

This backend uses NCCL for GPU-resident communication and MPI only for process
launch, NCCL bootstrap, and result collection.

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

## Build And Run

Leonardo provides NCCL through NVHPC and uses the `leonardo-cuda-nccl` preset:

```bash
source cluster/leonardo/environment.sh cuda
cmake --preset leonardo-cuda-nccl
cmake --build --preset leonardo-cuda-nccl
cluster/harness/launch.sh allreduce cuda_nccl 1n4g
```

For another cluster, provide `NCCL_INCLUDE_DIR` and `NCCL_LIBRARY` in a CMake
preset and register the resulting backend with the harness.
