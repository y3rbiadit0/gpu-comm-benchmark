include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OSHMPI REQUIRED)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${OSHMPI_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${OSHMPI_LIBRARIES})
check_cxx_source_compiles("#include <cstddef>
#include <shmem.h>
#include <shmemx.h>
int main() {
  shmemx_space_t space;
  shmemx_space_config_t config = {};
  config.memkind = SHMEMX_MEM_CUDA;
  shmemx_space_create(config, &space);
  shmemx_space_attach(space);
  (void) shmemx_space_malloc(space, 1);
  shmemx_space_detach(space);
  shmemx_space_destroy(space);
  return 0;
}" COMM_PLAYGROUND_OSHMPI_HAS_SHMEMX_SPACE)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)

add_executable(oshmpi_vector_add vector_add.cu)
target_include_directories(oshmpi_vector_add PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_vector_add PRIVATE comm_playground_common ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_vector_add PRIVATE USE_CUDA=1)
if(NOT COMM_PLAYGROUND_OSHMPI_HAS_SHMEMX_SPACE)
  target_compile_definitions(oshmpi_vector_add PRIVATE COMM_PLAYGROUND_OSHMPI_SHMEMX_SPACE_FALLBACK=1)
endif()

set_target_properties(oshmpi_vector_add PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)
