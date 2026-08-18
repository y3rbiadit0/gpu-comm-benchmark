#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* gpu_bench_oshmpi_space_create(size_t bytes);
void gpu_bench_oshmpi_space_destroy(void* space);
void* gpu_bench_oshmpi_space_malloc(void* space, size_t bytes);

#ifdef __cplusplus
}
#endif
