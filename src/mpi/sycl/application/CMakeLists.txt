add_executable(sycl_mpi_cg_step cg_step.cpp)
target_link_libraries(sycl_mpi_cg_step PRIVATE gpu_bench_common MPI::MPI_CXX)
gpu_bench_enable_sycl(sycl_mpi_cg_step)

add_executable(sycl_mpi_moe moe.cpp)
target_link_libraries(sycl_mpi_moe PRIVATE gpu_bench_common MPI::MPI_CXX)
gpu_bench_enable_sycl(sycl_mpi_moe)
