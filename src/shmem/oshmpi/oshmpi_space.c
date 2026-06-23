#ifndef USE_CUDA
#define USE_CUDA 1
#endif

#include "oshmpi_space.h"

#include <shmem.h>
#include <shmemx.h>

void* comm_playground_oshmpi_space_create(size_t bytes) {
  shmemx_space_config_t config;
  config.sheap_size = bytes;
  config.num_contexts = 0;
  config.hints = 0;
  config.memkind = SHMEMX_MEM_CUDA;
  config.device_handle = NULL;

  shmemx_space_t space;
  shmemx_space_create(config, &space);
  shmemx_space_attach(space);
  return space;
}

void comm_playground_oshmpi_space_destroy(void* space) {
  shmemx_space_detach((shmemx_space_t) space);
  shmemx_space_destroy((shmemx_space_t) space);
}

void* comm_playground_oshmpi_space_malloc(void* space, size_t bytes) {
  return shmemx_space_malloc((shmemx_space_t) space, bytes);
}
