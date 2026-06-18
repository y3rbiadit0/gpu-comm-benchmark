include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OSHMPI REQUIRED)

add_executable(oshmpi_vector_add vector_add.cu)
target_include_directories(oshmpi_vector_add PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_vector_add PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_vector_add PRIVATE USE_CUDA=1)

set_target_properties(oshmpi_vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
