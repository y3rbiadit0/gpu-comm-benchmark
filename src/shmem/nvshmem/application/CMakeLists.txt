add_executable(cuda_nvshmem_cg_step cg_step.cu)
target_include_directories(cuda_nvshmem_cg_step PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_cg_step PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_moe moe.cu)
target_include_directories(cuda_nvshmem_moe PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_moe PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

set_target_properties(cuda_nvshmem_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
