# CUDA + NVSHMEM

CUDA examples using NVSHMEM symmetric buffers.

## Targets

| Target | Problem | Communication model |
| --- | --- | --- |
| `cuda_nvshmem_vector_add` | Each PE computes `c[i] = a[i] + b[i]` for a contiguous global slice. | Host-side NVSHMEM puts distribute inputs and collect local results. |
| `cuda_nvshmem_vector_add_device` | Same vector-add problem. | Device-side NVSHMEM calls move communication into CUDA kernels. |
| `cuda_nvshmem_halo_1d` | One-step 1D stencil over contiguous PE-owned segments. | Host-side one-sided puts write boundary values into neighbor ghost cells. |
| `cuda_nvshmem_halo_1d_iter` | Multi-iteration 1D halo stencil. | Host-side NVSHMEM orchestration repeats halo exchange and stencil compute. |
| `cuda_nvshmem_halo_1d_device` | One-step 1D halo stencil. | Device-side NVSHMEM puts and signals exchange ghost cells from CUDA kernels. |
| `cuda_nvshmem_halo_1d_device_iter` | Multi-iteration 1D halo stencil. | One collectively launched CUDA kernel repeats GPU-initiated halo exchange. |

## Run

```bash
mpirun -np 4 ./build/leonardo-cuda-nvshmem/implementations/cuda_nvshmem/cuda_nvshmem_vector_add 1048576
mpirun -np 4 ./build/leonardo-cuda-nvshmem/implementations/cuda_nvshmem/cuda_nvshmem_halo_1d 1048576
mpirun -np 4 ./build/leonardo-cuda-nvshmem/implementations/cuda_nvshmem/cuda_nvshmem_halo_1d_iter 1048576 100
```
