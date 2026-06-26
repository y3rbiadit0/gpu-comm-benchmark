# CUDA + NCCL

CUDA examples using MPI for process launch and NCCL for GPU-resident communication.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_nccl_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | NCCL point-to-point distributes inputs and gathers local results. |
| `cuda_nccl_halo_1d` | One-step 1D stencil over contiguous rank-owned segments. | NCCL `ncclSend`/`ncclRecv` exchanges device-buffer ghost cells with neighbors. |
| `cuda_nccl_dot_product` | Double-precision global dot product (CG inner-product). | `ncclAllReduce` reduces a device-resident scalar across ranks on a stream. |
| `cuda_nccl_pingpong` | Two-endpoint one-way latency/bandwidth, internal size sweep. | Matched `ncclSend`/`ncclRecv` round-trip device buffers between 2 ranks. |
| `cuda_nccl_halo_2d` | 2D 5-point Jacobi stencil, column-slab decomposition. | Grouped `ncclSend`/`ncclRecv` exchange packed (strided) halo columns with neighbors. |
| `cuda_nccl_alltoall` | All-to-all personalized exchange (bisection bandwidth). | Grouped `ncclSend`/`ncclRecv` to every peer (NCCL has no native all-to-all). |
| `cuda_nccl_cg_step` | CG iteration skeleton (SpMV halo + two reductions). | Grouped `ncclSend`/`ncclRecv` halo + two `ncclAllReduce`. |

## Run

```bash
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_vector_add 1048576
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_halo_1d 1048576
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_dot_product 1048576 100 20
mpirun -np 2 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_pingpong 4194304 100 20
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_halo_2d 4096 50 10
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_alltoall 65536 100 20
mpirun -np 4 ./build/leonardo-cuda-nccl/src/xccl/cuda/cuda_nccl_cg_step 512 50 10
```
