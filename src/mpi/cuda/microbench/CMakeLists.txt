add_executable(cuda_mpi_halo_1d halo_1d.cu)
target_link_libraries(cuda_mpi_halo_1d PRIVATE gpu_bench_common MPI::MPI_CXX)

add_executable(cuda_mpi_pingpong pingpong.cu)
target_link_libraries(cuda_mpi_pingpong PRIVATE gpu_bench_common MPI::MPI_CXX)

add_executable(cuda_mpi_alltoall alltoall.cu)
target_link_libraries(cuda_mpi_alltoall PRIVATE gpu_bench_common MPI::MPI_CXX)

add_executable(cuda_mpi_allreduce allreduce.cu)
target_link_libraries(cuda_mpi_allreduce PRIVATE gpu_bench_common MPI::MPI_CXX)

set_target_properties(cuda_mpi_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_allreduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_pingpong PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
