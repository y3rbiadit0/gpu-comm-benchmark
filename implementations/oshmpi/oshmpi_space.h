#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* comm_playground_oshmpi_space_create(size_t bytes);
void comm_playground_oshmpi_space_destroy(void* space);
void* comm_playground_oshmpi_space_malloc(void* space, size_t bytes);

#ifdef __cplusplus
}
#endif
