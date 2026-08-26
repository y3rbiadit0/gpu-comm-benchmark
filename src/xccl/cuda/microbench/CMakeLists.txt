add_executable(cuda_nccl_halo_1d halo_1d.cu)
target_include_directories(cuda_nccl_halo_1d PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_halo_1d PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

add_executable(cuda_nccl_pingpong pingpong.cu)
target_include_directories(cuda_nccl_pingpong PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_pingpong PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

add_executable(cuda_nccl_alltoall alltoall.cu)
target_include_directories(cuda_nccl_alltoall PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_alltoall PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

add_executable(cuda_nccl_allreduce allreduce.cu)
target_include_directories(cuda_nccl_allreduce PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_allreduce PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

set_target_properties(cuda_nccl_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_nccl_allreduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_nccl_pingpong PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_nccl_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
