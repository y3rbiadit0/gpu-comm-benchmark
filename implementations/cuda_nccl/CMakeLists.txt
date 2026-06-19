include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ${NCCL_INCLUDE_DIR} $ENV{NCCL_HOME}
  PATH_SUFFIXES include)
find_library(NCCL_LIBRARY nccl
  HINTS ${NCCL_LIBRARY} $ENV{NCCL_HOME}
  PATH_SUFFIXES lib lib64)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_INCLUDE_DIR/NCCL_LIBRARY or NCCL_HOME")
endif()

add_executable(cuda_nccl_vector_add vector_add.cu)
target_include_directories(cuda_nccl_vector_add PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_vector_add PRIVATE comm_playground_common MPI::MPI_CXX ${NCCL_LIBRARY})

add_executable(cuda_nccl_halo_1d halo_1d.cu)
target_include_directories(cuda_nccl_halo_1d PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_halo_1d PRIVATE comm_playground_common MPI::MPI_CXX ${NCCL_LIBRARY})

set_target_properties(cuda_nccl_vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

set_target_properties(cuda_nccl_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
