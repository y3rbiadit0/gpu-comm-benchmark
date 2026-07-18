include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)
enable_language(C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OSHMPI REQUIRED)

add_executable(oshmpi_halo_1d halo_1d.cu oshmpi_space.c)
target_include_directories(oshmpi_halo_1d PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_halo_1d PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_halo_1d PRIVATE USE_CUDA=1)

add_executable(oshmpi_pingpong pingpong.cu oshmpi_space.c)
target_include_directories(oshmpi_pingpong PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_pingpong PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_pingpong PRIVATE USE_CUDA=1)

add_executable(oshmpi_alltoall alltoall.cu oshmpi_space.c)
target_include_directories(oshmpi_alltoall PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_alltoall PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_alltoall PRIVATE USE_CUDA=1)

add_executable(oshmpi_allreduce allreduce.cu)
target_include_directories(oshmpi_allreduce PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_allreduce PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_allreduce PRIVATE USE_CUDA=1)

add_executable(oshmpi_cg_step cg_step.cu oshmpi_space.c)
target_include_directories(oshmpi_cg_step PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_cg_step PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_cg_step PRIVATE USE_CUDA=1)

add_executable(oshmpi_moe moe.cu oshmpi_space.c)
target_include_directories(oshmpi_moe PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_moe PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_moe PRIVATE USE_CUDA=1)

set_target_properties(oshmpi_moe PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

set_target_properties(oshmpi_cg_step PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RUNTIME_LIBRARY Shared
)

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
