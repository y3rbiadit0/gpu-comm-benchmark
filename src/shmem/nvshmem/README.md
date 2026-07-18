# CUDA + NVSHMEM

CUDA examples using NVSHMEM symmetric buffers.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_nvshmem_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | Device-initiated `nvshmemx_float_put_signal_nbi_block` writes halos into symmetric neighbor buffers; point-to-point `nvshmem_signal_wait_until` sync (no barrier_all in the timed loop). Single-block put + per-iteration launch (the naive baseline). |
| `cuda_nvshmem_halo_1d_optimized` | Same comm-only 1D halo exchange. | A persistent cooperative kernel (`nvshmemx_collective_launch` + `grid.sync()`) runs the whole loop in one launch. Multiple blocks move halo chunks, then block 0 calls `nvshmem_quiet` and emits one completion signal per direction. Inter-node the grid is capped (8 blocks by default; `CP_NVSHMEM_MAX_BLOCKS` overrides) to avoid flooding the host-proxy path when IBGDA is disabled. See [`docs/analysis/halo_1d-crossover.md`](../../../docs/analysis/halo_1d-crossover.md). |
| `cuda_nvshmem_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | Device-initiated kernel: `nvshmemx_float_put_block` + `nvshmemx_signal_op`/`nvshmem_signal_wait_until` round-trip between 2 PEs (NVSHMEM p2p sync is device-only). |
| `cuda_nvshmem_allreduce` | Float32 sum allreduce latency/bandwidth, internal size sweep. | `nvshmem_float_sum_reduce` over device-symmetric buffers. |
| `cuda_nvshmem_alltoall` | All-to-all personalized exchange (bisection bandwidth). | Native `nvshmem_float_alltoall` team collective over symmetric buffers. |
| `cuda_nvshmem_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | Host-driven `nvshmem_float_put`+barrier halo + two `nvshmem_double_sum_reduce`. |
| `cuda_nvshmem_moe` | Top-1 MoE dispatch + combine with variable expert loads. | Variable-count `nvshmem_float_put` loops over device-symmetric buffers, with `quiet` + `barrier_all` after dispatch and inverse combine. |

## Run

```bash
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_halo_1d 1048576 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_halo_1d_optimized 1048576 100 20
mpirun -np 2 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_pingpong 4194304 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_allreduce 4194304 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_alltoall 65536 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_cg_step 512 50 10
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_moe 16384 256 100 20 uniform,locality80,hotspot80
```
