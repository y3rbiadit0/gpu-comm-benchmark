# Application benchmarks whose kernel issues its own communication, through
# NVSHMEM's device API.
#
# Unlike src/xccl/cuda/device, this needs no version guard: NVSHMEM has had
# device-side RMA, signals and team reductions since long before the 2.11 the
# nvhpc module ships, so every NVSHMEM this project builds against has one.
#
# Separate from ../CMakeLists.txt because these are a separate backend
# (cuda_nvshmem_device in cluster/leonardo/backends.sh), not a variant of
# cuda_nvshmem: the programming model differs, so a result from one must never
# be aggregated with a result from the other.

add_executable(cuda_nvshmem_device_cg_step cg_step.cu)
target_include_directories(cuda_nvshmem_device_cg_step PRIVATE ${NVSHMEM_INCLUDE_DIR})
target_link_libraries(cuda_nvshmem_device_cg_step PRIVATE
  gpu_bench_common
  MPI::MPI_CXX
  ${NVSHMEM_HOST_LIBRARY}
  ${NVSHMEM_DEVICE_LIBRARY}
)

# Required, not merely matching the sibling targets: the kernel calls NVSHMEM
# device functions that live in libnvshmem_device, so the device code has to be
# separately compiled and device-linked.
set_target_properties(cuda_nvshmem_device_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Scope of the in-kernel reduction, mirroring aCG's ACG_NVSHMEM_ALLREDUCE_SCOPE.
# The default reproduces aCG's default, which is what the archived device-solver
# runs were built with; change it only to run that comparison deliberately.
set(GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE "thread" CACHE STRING
    "Scope of the in-kernel NVSHMEM reduction: thread, warp, or block")
set_property(CACHE GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE PROPERTY STRINGS thread warp block)
if(GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE STREQUAL "block")
  target_compile_definitions(cuda_nvshmem_device_cg_step PRIVATE GPU_BENCH_NVSHMEM_ALLREDUCE_BLOCK)
elseif(GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE STREQUAL "warp")
  target_compile_definitions(cuda_nvshmem_device_cg_step PRIVATE GPU_BENCH_NVSHMEM_ALLREDUCE_WARP)
elseif(NOT GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE STREQUAL "thread")
  message(FATAL_ERROR
    "GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE must be thread, warp or block, "
    "not '${GPU_BENCH_NVSHMEM_ALLREDUCE_SCOPE}'")
endif()
