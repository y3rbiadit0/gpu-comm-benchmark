include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

add_executable(cuda_mpi_vector_add vector_add.cu)
target_link_libraries(cuda_mpi_vector_add PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_halo_1d halo_1d.cu)
target_link_libraries(cuda_mpi_halo_1d PRIVATE comm_playground_common MPI::MPI_CXX)

set_target_properties(cuda_mpi_vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
