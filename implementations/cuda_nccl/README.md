# CUDA + NCCL

CUDA examples using MPI for process launch and NCCL for GPU-resident communication.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_nccl_vector_add` | Each rank computes `c[i] = a[i] + b[i]` for a contiguous global slice. | NCCL point-to-point distributes inputs and gathers local results. |
| `cuda_nccl_halo_1d` | One-step 1D stencil over contiguous rank-owned segments. | NCCL `ncclSend`/`ncclRecv` exchanges device-buffer ghost cells with neighbors. |

## Run

```bash
mpirun -np 4 ./build/leonardo-cuda-nccl/implementations/cuda_nccl/cuda_nccl_vector_add 1048576
mpirun -np 4 ./build/leonardo-cuda-nccl/implementations/cuda_nccl/cuda_nccl_halo_1d 1048576
```
