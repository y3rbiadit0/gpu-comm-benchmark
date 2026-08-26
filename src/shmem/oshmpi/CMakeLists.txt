include(${PROJECT_SOURCE_DIR}/cmake/cuda.cmake)
enable_language(C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OSHMPI REQUIRED)

# The CUDA symmetric-space helper is shared by every OSHMPI target, both tiers,
# so it stays in this directory. PUBLIC carries its header path to whatever links
# it -- the benchmark sources include it by quoted name from one level down.
add_library(oshmpi_space STATIC oshmpi_space.c)
target_include_directories(oshmpi_space PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(oshmpi_space PRIVATE ${OSHMPI_INCLUDE_DIRS})
target_link_libraries(oshmpi_space PUBLIC ${OSHMPI_LIBRARIES})
target_compile_definitions(oshmpi_space PRIVATE USE_CUDA=1)

# Binaries stay in this directory rather than in the tier subdirectories: the
# experiment job scripts under cluster/leonardo/experiments hardcode
# build/<preset>/src/shmem/oshmpi/<target>, and the source layout is a reading aid, not
# a deployment contract.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Two tiers, see README "Benchmarks".
add_subdirectory(microbench)    # one operation, swept over message size
add_subdirectory(application)   # one problem size, several operations combined
