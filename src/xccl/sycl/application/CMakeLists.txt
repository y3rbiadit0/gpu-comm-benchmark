foreach(benchmark IN LISTS GPU_BENCH_ONECCL_APPLICATION)
  set(target sycl_oneccl_${benchmark})
  add_executable(${target} ${benchmark}.cpp)
  target_include_directories(${target} PRIVATE ${ONECCL_INCLUDE_DIR})
  if(GPU_BENCH_ONECCL_USE_BUNDLED_MPI)
    target_include_directories(${target} PRIVATE ${ONECCL_BUNDLED_MPI_INCLUDE_DIR})
  endif()
  target_link_libraries(${target} PRIVATE ${_oneccl_link_libraries})
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${ONECCL_RUNTIME_RPATH}"
    INSTALL_RPATH "${ONECCL_RUNTIME_RPATH}")
  gpu_bench_enable_sycl(${target})
endforeach()
