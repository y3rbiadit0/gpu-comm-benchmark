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

# Binaries stay in this directory rather than in the tier subdirectories: the
# experiment job scripts under cluster/leonardo/experiments hardcode
# build/<preset>/src/shmem/nvshmem/<target>, and the source layout is a reading aid, not
# a deployment contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Two tiers, see README "Benchmarks".
add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined
