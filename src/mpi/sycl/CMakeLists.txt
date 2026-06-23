include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)

add_executable(sycl_mpi_vector_add vector_add.cpp)
target_link_libraries(sycl_mpi_vector_add PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_vector_add)

add_executable(sycl_mpi_halo_1d halo_1d.cpp)
target_link_libraries(sycl_mpi_halo_1d PRIVATE comm_playground_common MPI::MPI_CXX)
comm_playground_enable_sycl(sycl_mpi_halo_1d)
