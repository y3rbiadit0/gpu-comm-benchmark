include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
  HINTS ${NVSHMEM_INCLUDE_DIR} $ENV{NVSHMEM_HOME}
  PATH_SUFFIXES include)
find_library(NVSHMEM_HOST_LIBRARY nvshmem_host
  HINTS ${NVSHMEM_LIBRARY_PATH} $ENV{NVSHMEM_HOME}
  PATH_SUFFIXES lib lib64)
find_library(NVSHMEM_DEVICE_LIBRARY nvshmem_device
  HINTS ${NVSHMEM_LIBRARY_PATH} $ENV{NVSHMEM_HOME}
  PATH_SUFFIXES lib lib64)

if(NOT NVSHMEM_INCLUDE_DIR OR NOT NVSHMEM_HOST_LIBRARY OR NOT NVSHMEM_DEVICE_LIBRARY)
  message(FATAL_ERROR "NVSHMEM not found. Set NVSHMEM_INCLUDE_DIR/NVSHMEM_LIBRARY_PATH or NVSHMEM_HOME")
endif()

add_executable(cuda_nvshmem_halo_1d halo_1d.cu)
target_include_directories(cuda_nvshmem_halo_1d PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_halo_1d PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_halo_1d_optimized halo_1d-optimized.cu)
target_include_directories(cuda_nvshmem_halo_1d_optimized PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_halo_1d_optimized PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_pingpong pingpong.cu)
target_include_directories(cuda_nvshmem_pingpong PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_pingpong PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_alltoall alltoall.cu)
target_include_directories(cuda_nvshmem_alltoall PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_alltoall PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_allreduce allreduce.cu)
target_include_directories(cuda_nvshmem_allreduce PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_allreduce PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_cg_step cg_step.cu)
target_include_directories(cuda_nvshmem_cg_step PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_cg_step PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_moe moe.cu)
target_include_directories(cuda_nvshmem_moe PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_moe PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

set_target_properties(cuda_nvshmem_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_allreduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_pingpong PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_halo_1d_optimized PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
