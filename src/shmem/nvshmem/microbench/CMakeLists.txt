add_executable(cuda_nvshmem_halo_1d halo_1d.cu)
target_include_directories(cuda_nvshmem_halo_1d PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_halo_1d PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_pingpong pingpong.cu)
target_include_directories(cuda_nvshmem_pingpong PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_pingpong PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_alltoall alltoall.cu)
target_include_directories(cuda_nvshmem_alltoall PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_alltoall PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_allreduce allreduce.cu)
target_include_directories(cuda_nvshmem_allreduce PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_allreduce PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

set_target_properties(cuda_nvshmem_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_allreduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_pingpong PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
