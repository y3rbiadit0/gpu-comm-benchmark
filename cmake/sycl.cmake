set(GPU_BENCH_SYCL_FLAGS "-fsycl" CACHE STRING "Compiler and linker flags for SYCL targets")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("${GPU_BENCH_SYCL_FLAGS}" GPU_BENCH_HAS_SYCL_FLAG)

if(NOT GPU_BENCH_HAS_SYCL_FLAG)
  message(FATAL_ERROR "The active C++ compiler does not accept '${GPU_BENCH_SYCL_FLAGS}'. Configure with a SYCL compiler, for example: cmake --preset sycl-mpi -DCMAKE_CXX_COMPILER=icpx")
endif()

function(gpu_bench_enable_sycl target)
  separate_arguments(_sycl_flags NATIVE_COMMAND "${GPU_BENCH_SYCL_FLAGS}")
  target_compile_options(${target} PRIVATE ${_sycl_flags})
  target_link_options(${target} PRIVATE ${_sycl_flags})
endfunction()
