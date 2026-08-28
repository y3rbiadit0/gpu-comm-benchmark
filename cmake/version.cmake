file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" GPU_BENCH_VERSION)
string(STRIP "${GPU_BENCH_VERSION}" GPU_BENCH_VERSION)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../VERSION")

if(NOT GPU_BENCH_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "VERSION must contain a semantic version such as 0.1.0")
endif()

set(GPU_BENCH_SOURCE_REVISION "" CACHE STRING
    "Source revision embedded in benchmark output (auto-detected when empty)")

if(NOT GPU_BENCH_SOURCE_REVISION)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_LIST_DIR}/.."
              describe --tags --always --dirty
      RESULT_VARIABLE _gpu_bench_git_result
      OUTPUT_VARIABLE _gpu_bench_git_revision
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_gpu_bench_git_result EQUAL 0 AND _gpu_bench_git_revision)
      set(GPU_BENCH_SOURCE_REVISION "${_gpu_bench_git_revision}")
    endif()
  endif()
endif()

if(NOT GPU_BENCH_SOURCE_REVISION)
  set(GPU_BENCH_SOURCE_REVISION "unknown")
endif()

if(NOT GPU_BENCH_SOURCE_REVISION MATCHES "^[A-Za-z0-9._/+~-]+$")
  message(FATAL_ERROR
    "GPU_BENCH_SOURCE_REVISION must be a non-empty token containing no whitespace")
endif()
