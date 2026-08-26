add_executable(cuda_mpi_cg_step cg_step.cu)
target_link_libraries(cuda_mpi_cg_step PRIVATE gpu_bench_common MPI::MPI_CXX)

add_executable(cuda_mpi_moe moe.cu)
target_link_libraries(cuda_mpi_moe PRIVATE gpu_bench_common MPI::MPI_CXX)

set_target_properties(cuda_mpi_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
