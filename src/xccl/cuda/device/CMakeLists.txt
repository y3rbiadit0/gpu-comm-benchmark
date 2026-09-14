# Benchmarks that issue communication from inside the kernel, through NCCL's
# device API. Reached only when the NCCL found above is new enough to have one
# -- the guard lives in the parent CMakeLists, so an older NCCL builds the rest
# of this backend normally instead of failing on a missing header.

add_executable(cuda_nccl_device_alltoall alltoall.cu)
target_include_directories(cuda_nccl_device_alltoall PRIVATE ${NCCL_INCLUDE_DIR})
target_link_libraries(cuda_nccl_device_alltoall PRIVATE gpu_bench_common MPI::MPI_CXX ${NCCL_LIBRARY})

# Matches the other CUDA NCCL targets: the device API is header-only over
# libnccl, so nothing here needs relocatable device code.
set_target_properties(cuda_nccl_device_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
