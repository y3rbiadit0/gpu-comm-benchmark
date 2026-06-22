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

add_executable(cuda_nvshmem_vector_add vector_add.cu)
target_include_directories(cuda_nvshmem_vector_add PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_vector_add PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_vector_add_device vector_add-device.cu)
target_include_directories(cuda_nvshmem_vector_add_device PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_vector_add_device PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_halo_1d halo_1d.cu)
target_include_directories(cuda_nvshmem_halo_1d PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_halo_1d PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_halo_1d_device halo_1d-device.cu)
target_include_directories(cuda_nvshmem_halo_1d_device PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_halo_1d_device PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

add_executable(cuda_nvshmem_halo_1d_device_iter halo_1d-device-iter.cu)
target_include_directories(cuda_nvshmem_halo_1d_device_iter PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_halo_1d_device_iter PRIVATE
  comm_playground_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

set_target_properties(cuda_nvshmem_vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_vector_add_device PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_halo_1d_device PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_nvshmem_halo_1d_device_iter PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
