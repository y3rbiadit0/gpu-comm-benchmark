add_executable(cuda_nccl_cg_step cg_step.cu)
target_include_directories(cuda_nccl_cg_step PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_cg_step PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

add_executable(cuda_nccl_moe moe.cu)
target_include_directories(cuda_nccl_moe PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_moe PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

set_target_properties(cuda_nccl_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_nccl_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
