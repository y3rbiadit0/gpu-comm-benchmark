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

# Binaries land here, not in the microbench/ and application/ subdirectories:
# cluster/harness/job.sh resolves one path per backend from the registry in
# cluster/<cluster>/backends.sh. The source layout is a reading aid; the flat
# output directory is the contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Two tiers, see README "Benchmarks".
add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined

# A third, conditional: benchmarks that call NCCL's device API, where the kernel
# issues its own communication. nccl_device.h arrived in NCCL 2.28, and the
# module NCCL on a given machine may be far older -- Leonardo's nvhpc 24.5 ships
# 2.18.5 -- so read the version out of the header that was actually found and
# skip the directory rather than fail the whole backend on a missing include.
# Select a newer NCCL with GPU_BENCH_NCCL_VERSION (cluster/<name>/layout.sh).
file(READ "${NCCL_INCLUDE_DIR}/nccl.h" GPU_BENCH_NCCL_HEADER)
string(REGEX MATCH "#define NCCL_MAJOR ([0-9]+)" _ "${GPU_BENCH_NCCL_HEADER}")
set(GPU_BENCH_NCCL_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define NCCL_MINOR ([0-9]+)" _ "${GPU_BENCH_NCCL_HEADER}")
set(GPU_BENCH_NCCL_MINOR "${CMAKE_MATCH_1}")

if(GPU_BENCH_NCCL_MAJOR GREATER_EQUAL 3
   OR (GPU_BENCH_NCCL_MAJOR EQUAL 2 AND GPU_BENCH_NCCL_MINOR GREATER_EQUAL 28))
  message(STATUS
    "NCCL ${GPU_BENCH_NCCL_MAJOR}.${GPU_BENCH_NCCL_MINOR}: building device API benchmarks")
  add_subdirectory(device)
else()
  message(STATUS
    "NCCL ${GPU_BENCH_NCCL_MAJOR}.${GPU_BENCH_NCCL_MINOR} has no device API (needs 2.28); "
    "skipping cuda_nccl_device_* targets")
endif()
