add_executable(oshmpi_cg_step cg_step.cu)
target_include_directories(oshmpi_cg_step PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_cg_step PRIVATE gpu_bench_common oshmpi_space ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_cg_step PRIVATE USE_CUDA=1)

add_executable(oshmpi_moe moe.cu)
target_include_directories(oshmpi_moe PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_moe PRIVATE gpu_bench_common oshmpi_space ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_moe PRIVATE USE_CUDA=1)

set_target_properties(oshmpi_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

set_target_properties(oshmpi_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)
