# CUDA + NVSHMEM

CUDA examples using NVSHMEM symmetric buffers.

- `cuda_nvshmem_vector_add` and `cuda_nvshmem_halo_1d` use host-side NVSHMEM calls around CUDA compute kernels.
- `cuda_nvshmem_vector_add_device` and `cuda_nvshmem_halo_1d_device` move communication into CUDA kernels using device-side NVSHMEM calls and point-to-point signaling.
