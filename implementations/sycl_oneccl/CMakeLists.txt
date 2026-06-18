include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)

find_path(ONECCL_INCLUDE_DIR oneapi/ccl.hpp
  HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
  PATH_SUFFIXES include)
find_library(ONECCL_LIBRARY ccl
  HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
  PATH_SUFFIXES lib lib64)

if(NOT ONECCL_INCLUDE_DIR OR NOT ONECCL_LIBRARY)
  message(FATAL_ERROR "oneCCL not found. Set oneCCL_ROOT, ONECCL_ROOT, or CCL_ROOT")
endif()

add_executable(sycl_oneccl_vector_add vector_add.cpp)
target_include_directories(sycl_oneccl_vector_add PRIVATE ${ONECCL_INCLUDE_DIR})
target_link_libraries(sycl_oneccl_vector_add PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
comm_playground_enable_sycl(sycl_oneccl_vector_add)
