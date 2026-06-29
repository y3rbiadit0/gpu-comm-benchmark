include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

add_executable(cuda_mpi_vector_add vector_add.cu)
target_link_libraries(cuda_mpi_vector_add PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_halo_1d halo_1d.cu)
target_link_libraries(cuda_mpi_halo_1d PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_dot_product dot_product.cu)
target_link_libraries(cuda_mpi_dot_product PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_pingpong pingpong.cu)
target_link_libraries(cuda_mpi_pingpong PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_halo_2d halo_2d.cu)
target_link_libraries(cuda_mpi_halo_2d PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_alltoall alltoall.cu)
target_link_libraries(cuda_mpi_alltoall PRIVATE comm_playground_common MPI::MPI_CXX)

add_executable(cuda_mpi_cg_step cg_step.cu)
target_link_libraries(cuda_mpi_cg_step PRIVATE comm_playground_common MPI::MPI_CXX)

set_target_properties(cuda_mpi_vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_halo_2d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_dot_product PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_pingpong PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_mpi_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
