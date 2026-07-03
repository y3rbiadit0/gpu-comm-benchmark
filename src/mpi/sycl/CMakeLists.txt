include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)

add_executable(sycl_mpi_vector_add vector_add.cpp)
target_link_libraries(sycl_mpi_vector_add PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_vector_add)

add_executable(sycl_mpi_halo_1d halo_1d.cpp)
target_link_libraries(sycl_mpi_halo_1d PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_halo_1d)

add_executable(sycl_mpi_dot_product dot_product.cpp)
target_link_libraries(sycl_mpi_dot_product PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_dot_product)

add_executable(sycl_mpi_pingpong pingpong.cpp)
target_link_libraries(sycl_mpi_pingpong PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_pingpong)

add_executable(sycl_mpi_alltoall alltoall.cpp)
target_link_libraries(sycl_mpi_alltoall PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_alltoall)

add_executable(sycl_mpi_cg_step cg_step.cpp)
target_link_libraries(sycl_mpi_cg_step PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_cg_step)
