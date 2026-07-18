include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)

option(COMM_PLAYGROUND_ONECCL_USE_BUNDLED_MPI "Link SYCL oneCCL examples with oneCCL's bundled Intel MPI" OFF)

find_path(ONECCL_INCLUDE_DIR oneapi/ccl.hpp
  HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
  PATH_SUFFIXES include)
find_library(ONECCL_LIBRARY ccl
  HINTS ${oneCCL_ROOT} $ENV{ONECCL_ROOT} $ENV{CCL_ROOT}
  PATH_SUFFIXES lib lib64)

if(NOT ONECCL_INCLUDE_DIR OR NOT ONECCL_LIBRARY)
  message(FATAL_ERROR "oneCCL not found. Set oneCCL_ROOT, ONECCL_ROOT, or CCL_ROOT")
endif()

add_executable(sycl_oneccl_halo_1d halo_1d.cpp)
add_executable(sycl_oneccl_pingpong pingpong.cpp)
add_executable(sycl_oneccl_alltoall alltoall.cpp)
add_executable(sycl_oneccl_allreduce allreduce.cpp)
add_executable(sycl_oneccl_cg_step cg_step.cpp)
add_executable(sycl_oneccl_moe moe.cpp)
target_include_directories(sycl_oneccl_halo_1d PRIVATE ${ONECCL_INCLUDE_DIR})
target_include_directories(sycl_oneccl_pingpong PRIVATE ${ONECCL_INCLUDE_DIR})
target_include_directories(sycl_oneccl_alltoall PRIVATE ${ONECCL_INCLUDE_DIR})
target_include_directories(sycl_oneccl_allreduce PRIVATE ${ONECCL_INCLUDE_DIR})
target_include_directories(sycl_oneccl_cg_step PRIVATE ${ONECCL_INCLUDE_DIR})
target_include_directories(sycl_oneccl_moe PRIVATE ${ONECCL_INCLUDE_DIR})

if(COMM_PLAYGROUND_ONECCL_USE_BUNDLED_MPI)
  find_path(ONECCL_BUNDLED_MPI_INCLUDE_DIR mpi.h
    HINTS $ENV{ONECCL_NCCL_ROOT}/opt/mpi $ENV{ONECCL_ROOT}/opt/mpi $ENV{CCL_ROOT}/opt/mpi
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
  find_library(ONECCL_BUNDLED_MPI_LIBRARY mpi
    HINTS
      $ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release
      $ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib
      $ENV{ONECCL_ROOT}/opt/mpi/lib/release
      $ENV{ONECCL_ROOT}/opt/mpi/lib
      $ENV{HOME}/oneCCL-nccl/deps/mpi/lib
    NO_DEFAULT_PATH)

  if(NOT ONECCL_BUNDLED_MPI_LIBRARY)
    file(GLOB _oneccl_bundled_mpi_candidates
      "$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release/libmpi.so*"
      "$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/libmpi.so*"
      "$ENV{ONECCL_ROOT}/opt/mpi/lib/release/libmpi.so*"
      "$ENV{ONECCL_ROOT}/opt/mpi/lib/libmpi.so*"
      "$ENV{HOME}/oneCCL-nccl/deps/mpi/lib/libmpi.so*")
    if(_oneccl_bundled_mpi_candidates)
      list(SORT _oneccl_bundled_mpi_candidates)
      list(GET _oneccl_bundled_mpi_candidates 0 ONECCL_BUNDLED_MPI_LIBRARY)
    endif()
  endif()

  if(NOT ONECCL_BUNDLED_MPI_INCLUDE_DIR OR NOT ONECCL_BUNDLED_MPI_LIBRARY)
    message(FATAL_ERROR "oneCCL bundled MPI not found. Set ONECCL_NCCL_ROOT or ONECCL_BUNDLED_MPI_INCLUDE_DIR/ONECCL_BUNDLED_MPI_LIBRARY")
  endif()

  find_package(Threads REQUIRED)
  get_filename_component(ONECCL_BUNDLED_MPI_LIBRARY_DIR "${ONECCL_BUNDLED_MPI_LIBRARY}" DIRECTORY)
  target_include_directories(sycl_oneccl_halo_1d PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  target_include_directories(sycl_oneccl_pingpong PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  target_include_directories(sycl_oneccl_alltoall PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  target_include_directories(sycl_oneccl_allreduce PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  target_include_directories(sycl_oneccl_cg_step PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  target_include_directories(sycl_oneccl_moe PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  target_link_libraries(sycl_oneccl_halo_1d PRIVATE comm_playground_common ${ONECCL_LIBRARY} ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
  target_link_libraries(sycl_oneccl_pingpong PRIVATE comm_playground_common ${ONECCL_LIBRARY} ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
  target_link_libraries(sycl_oneccl_alltoall PRIVATE comm_playground_common ${ONECCL_LIBRARY} ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
  target_link_libraries(sycl_oneccl_allreduce PRIVATE comm_playground_common ${ONECCL_LIBRARY} ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
  target_link_libraries(sycl_oneccl_cg_step PRIVATE comm_playground_common ${ONECCL_LIBRARY} ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
  target_link_libraries(sycl_oneccl_moe PRIVATE comm_playground_common ${ONECCL_LIBRARY} ${ONECCL_BUNDLED_MPI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
  set_target_properties(sycl_oneccl_halo_1d PROPERTIES
    BUILD_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}"
    INSTALL_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
  set_target_properties(sycl_oneccl_pingpong PROPERTIES
    BUILD_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}"
    INSTALL_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
  set_target_properties(sycl_oneccl_alltoall PROPERTIES
    BUILD_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}"
    INSTALL_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
  set_target_properties(sycl_oneccl_allreduce PROPERTIES
    BUILD_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}"
    INSTALL_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
  set_target_properties(sycl_oneccl_cg_step PROPERTIES
    BUILD_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}"
    INSTALL_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
  set_target_properties(sycl_oneccl_moe PROPERTIES
    BUILD_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}"
    INSTALL_RPATH "${ONECCL_BUNDLED_MPI_LIBRARY_DIR};$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib/release;$ENV{ONECCL_NCCL_ROOT}/opt/mpi/lib;$ENV{HOME}/oneCCL-nccl/deps/mpi/lib;$ENV{ONECCL_NCCL_ROOT}/lib;$ENV{DPCPP_INSTALL}/lib;$ENV{GCC12_LIB}")
else()
  include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
  target_link_libraries(sycl_oneccl_halo_1d PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
  target_link_libraries(sycl_oneccl_pingpong PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
  target_link_libraries(sycl_oneccl_alltoall PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
  target_link_libraries(sycl_oneccl_allreduce PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
  target_link_libraries(sycl_oneccl_cg_step PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
  target_link_libraries(sycl_oneccl_moe PRIVATE comm_playground_common MPI::MPI_CXX ${ONECCL_LIBRARY})
endif()

comm_playground_enable_sycl(sycl_oneccl_halo_1d)
comm_playground_enable_sycl(sycl_oneccl_pingpong)
comm_playground_enable_sycl(sycl_oneccl_alltoall)
comm_playground_enable_sycl(sycl_oneccl_allreduce)
comm_playground_enable_sycl(sycl_oneccl_cg_step)
comm_playground_enable_sycl(sycl_oneccl_moe)
