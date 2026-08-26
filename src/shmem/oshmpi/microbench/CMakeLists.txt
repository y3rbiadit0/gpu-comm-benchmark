add_executable(oshmpi_halo_1d halo_1d.cu)
target_include_directories(oshmpi_halo_1d PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_halo_1d PRIVATE gpu_bench_common oshmpi_space ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_halo_1d PRIVATE USE_CUDA=1)

add_executable(oshmpi_pingpong pingpong.cu)
target_include_directories(oshmpi_pingpong PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_pingpong PRIVATE gpu_bench_common oshmpi_space ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_pingpong PRIVATE USE_CUDA=1)

add_executable(oshmpi_alltoall alltoall.cu)
target_include_directories(oshmpi_alltoall PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_alltoall PRIVATE gpu_bench_common oshmpi_space ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_alltoall PRIVATE USE_CUDA=1)

add_executable(oshmpi_allreduce allreduce.cu)
target_include_directories(oshmpi_allreduce PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_allreduce PRIVATE gpu_bench_common oshmpi_space ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_allreduce PRIVATE USE_CUDA=1)

set_target_properties(oshmpi_alltoall PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

set_target_properties(oshmpi_allreduce PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

set_target_properties(oshmpi_pingpong PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

set_target_properties(oshmpi_halo_1d PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)
