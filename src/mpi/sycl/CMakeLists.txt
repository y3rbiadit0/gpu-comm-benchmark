include(${PROJECT_SOURCE_DIR}/cmake/mpi.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/sycl.cmake)

# Binaries stay in this directory rather than in the tier subdirectories: the
# experiment job scripts under cluster/leonardo/experiments hardcode
# build/<preset>/src/mpi/sycl/<target>, and the source layout is a reading aid, not
# a deployment contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Two tiers, see README "Benchmarks".
add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined
