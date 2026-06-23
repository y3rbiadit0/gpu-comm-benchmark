cmake_minimum_required(VERSION 3.24)

project(comm_playground LANGUAGES CXX)

option(ENABLE_CUDA_MPI "Build CUDA + MPI examples" OFF)
option(ENABLE_SYCL_MPI "Build SYCL + MPI examples" OFF)
option(ENABLE_SYCL_ONECCL "Build SYCL + oneCCL examples" OFF)
option(ENABLE_CUDA_NCCL "Build CUDA + NCCL examples" OFF)
option(ENABLE_CUDA_NVSHMEM "Build CUDA + NVSHMEM examples" OFF)
option(ENABLE_OSHMPI "Build OSHMPI examples" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(comm_playground_common INTERFACE)
target_include_directories(comm_playground_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(ENABLE_CUDA_MPI)
  add_subdirectory(src/cuda_mpi)
endif()

if(ENABLE_SYCL_MPI)
  add_subdirectory(src/sycl_mpi)
endif()

if(ENABLE_SYCL_ONECCL)
  add_subdirectory(src/sycl_oneccl)
endif()

if(ENABLE_CUDA_NCCL)
  add_subdirectory(src/cuda_nccl)
endif()

if(ENABLE_CUDA_NVSHMEM)
  add_subdirectory(src/cuda_nvshmem)
endif()

if(ENABLE_OSHMPI)
  add_subdirectory(src/oshmpi)
endif()
