#include <cuda_runtime.h>

#include <cstddef>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>
#include <shmemx.h>

#if defined(COMM_PLAYGROUND_OSHMPI_SHMEMX_SPACE_FALLBACK)
typedef void* shmemx_space_t;
typedef void* shmemx_device_handle_t;
typedef enum { SHMEMX_MEM_HOST, SHMEMX_MEM_CUDA, SHMEMX_MEM_ZE } shmemx_memkind_t;
typedef struct {
  size_t sheap_size;
  int num_contexts;
  shmemx_memkind_t memkind;
  shmemx_device_handle_t device_handle;
  long hints;
} shmemx_space_config_t;
extern "C" void shmemx_space_create(shmemx_space_config_t, shmemx_space_t*);
extern "C" void shmemx_space_destroy(shmemx_space_t);
extern "C" void shmemx_space_attach(shmemx_space_t);
extern "C" void shmemx_space_detach(shmemx_space_t);
extern "C" void* shmemx_space_malloc(shmemx_space_t, size_t);
#endif

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t offset,
                                  std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = offset + i;
    c[global_i] = a[global_i] + b[global_i];
  }
}

std::size_t local_size_for_rank(std::size_t global_size, int rank, int ranks) {
  const auto base = global_size / static_cast<std::size_t>(ranks);
  const auto remainder = global_size % static_cast<std::size_t>(ranks);
  return base + (static_cast<std::size_t>(rank) < remainder ? 1U : 0U);
}

std::size_t offset_for_rank(std::size_t global_size, int rank, int ranks) {
  const auto base = global_size / static_cast<std::size_t>(ranks);
  const auto remainder = global_size % static_cast<std::size_t>(ranks);
  const auto rank_value = static_cast<std::size_t>(rank);
  return rank_value * base + std::min(rank_value, remainder);
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();
  bool space_created = false;
  shmemx_space_t space;

  try {
    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto local_size = local_size_for_rank(global_size, pe, pes);
    const auto local_offset = offset_for_rank(global_size, pe, pes);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    shmemx_space_config_t config = {};
    config.sheap_size = std::max<std::size_t>(4U * global_size * sizeof(float), 1U << 20U);
    config.num_contexts = 0;
    config.hints = 0;
    config.memkind = SHMEMX_MEM_CUDA;
    config.device_handle = nullptr;
    shmemx_space_create(config, &space);
    shmemx_space_attach(space);
    space_created = true;

    auto* device_a = static_cast<float*>(shmemx_space_malloc(space, global_size * sizeof(float)));
    auto* device_b = static_cast<float*>(shmemx_space_malloc(space, global_size * sizeof(float)));
    auto* device_c = static_cast<float*>(shmemx_space_malloc(space, global_size * sizeof(float)));
    if (device_a == nullptr || device_b == nullptr || device_c == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI CUDA symmetric memory");
    }

    std::vector<float> host_global_a;
    std::vector<float> host_global_b;
    std::vector<float> host_global_c;
    if (pe == 0) {
      host_global_a.resize(global_size);
      host_global_b.resize(global_size);
      host_global_c.resize(global_size, 0.0F);
      for (std::size_t i = 0; i < global_size; ++i) {
        const auto value = static_cast<float>(i);
        host_global_a[i] = value;
        host_global_b[i] = 2.0F * value;
      }
      check_cuda(cudaMemcpy(device_a, host_global_a.data(), global_size * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(device_a)");
      check_cuda(cudaMemcpy(device_b, host_global_b.data(), global_size * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(device_b)");
    }
    check_cuda(cudaMemset(device_c, 0, global_size * sizeof(float)), "cudaMemset(device_c)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    shmem_barrier_all();
    comm_playground::wall_timer timer;

    if (pe == 0) {
      for (int target_pe = 1; target_pe < pes; ++target_pe) {
        shmem_putmem(device_a, device_a, global_size * sizeof(float), target_pe);
        shmem_putmem(device_b, device_b, global_size * sizeof(float), target_pe);
      }
    }
    shmem_barrier_all();

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      vector_add_kernel<<<grid_size, block_size>>>(device_a, device_b, device_c, local_offset, local_size);
      check_cuda(cudaGetLastError(), "vector_add_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel)");
    }

    shmem_barrier_all();
    if (pe != 0 && local_size > 0) {
      shmem_putmem(device_c + local_offset, device_c + local_offset, local_size * sizeof(float), 0);
    }
    shmem_barrier_all();

    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (pe == 0) {
      check_cuda(cudaMemcpy(host_global_c.data(), device_c, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_global_c)");
      global_ok = comm_playground::validate_vector_add(host_global_c.data(), global_size, 0) ? 1 : 0;
    }

    shmem_free(device_c);
    shmem_free(device_b);
    shmem_free(device_a);
    shmemx_space_detach(space);
    shmemx_space_destroy(space);
    space_created = false;

    if (pe == 0) {
      std::cout << "oshmpi_vector_add n=" << global_size << " pes=" << pes << " time_s=" << elapsed
                << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
    }

    shmem_finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    if (space_created) {
      shmemx_space_detach(space);
      shmemx_space_destroy(space);
    }
    shmem_global_exit(1);
  }
}
