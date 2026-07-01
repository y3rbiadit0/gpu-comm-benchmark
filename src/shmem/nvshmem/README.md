# CUDA + NVSHMEM

CUDA examples using NVSHMEM symmetric buffers.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_nvshmem_vector_add` | Each PE computes `c[i] = a[i] + b[i]` for a contiguous global slice. | Host-side NVSHMEM puts distribute inputs and collect local results. |
| `cuda_nvshmem_vector_add_device` | Same vector-add problem. | Device-side NVSHMEM calls move communication into CUDA kernels. |
| `cuda_nvshmem_halo_1d` | Comm-only 1D halo exchange, periodic ring, swept halo width. | Device-initiated `nvshmemx_float_put_signal_nbi_block` writes halos into symmetric neighbor buffers; point-to-point `nvshmem_signal_wait_until` sync (no barrier_all in the timed loop). Single-block put + per-iteration launch (the naive baseline). |
| `cuda_nvshmem_halo_1d_optimized` | Same comm-only 1D halo exchange. | Optimized variant that fixes the baseline's two handicaps: a **persistent cooperative kernel** (`nvshmemx_collective_launch` + `grid.sync()`) runs the whole loop in one launch, and **many blocks** each move a halo chunk (`SIGNAL_ADD` count as an all-blocks-done barrier) to scale copy bandwidth past one SM. See [`docs/analysis/halo_1d-crossover.md`](../../../docs/analysis/halo_1d-crossover.md). |
| `cuda_nvshmem_dot_product` | Double-precision global dot product (CG inner-product). | `nvshmem_double_sum_reduce` (NVSHMEM_TEAM_WORLD) reduces a device-resident symmetric scalar across PEs. |
| `cuda_nvshmem_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | Device-initiated kernel: `nvshmemx_float_put_block` + `nvshmemx_signal_op`/`nvshmem_signal_wait_until` round-trip between 2 PEs (NVSHMEM p2p sync is device-only). |
| `cuda_nvshmem_halo_2d` | 2D 5-point Jacobi stencil, column-slab decomposition. | Host-driven `nvshmem_float_put` + `barrier_all` write packed (strided) halo columns into symmetric neighbor buffers. |
| `cuda_nvshmem_alltoall` | All-to-all personalized exchange (bisection bandwidth). | Native `nvshmem_float_alltoall` team collective over symmetric buffers. |
| `cuda_nvshmem_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | Host-driven `nvshmem_float_put`+barrier halo + two `nvshmem_double_sum_reduce`. |

## Run

```bash
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_vector_add 1048576
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_halo_1d 1048576 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_dot_product 1048576 100 20
mpirun -np 2 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_pingpong 4194304 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_halo_2d 4096 50 10
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_alltoall 65536 100 20
mpirun -np 4 ./build/leonardo-cuda-nvshmem/src/shmem/nvshmem/cuda_nvshmem_cg_step 512 50 10
```
