#include <cuda_runtime.h>

#include <cstddef>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "oshmpi_space.h"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void init_interior_kernel(float* x, std::size_t global_offset, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = global_offset + i;
    x[global_i + 1U] = static_cast<float>(global_i);
  }
}

__global__ void halo_kernel(const float* x, float* y, std::size_t global_offset, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = global_offset + i;
    y[global_i] = 0.25F * x[global_i] + 0.5F * x[global_i + 1U] + 0.25F * x[global_i + 2U];
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
  void* space = nullptr;
  double* elapsed_by_pe = nullptr;

  try {
    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto local_size = local_size_for_rank(global_size, pe, pes);
    const auto local_offset = offset_for_rank(global_size, pe, pes);
    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    const auto symmetric_bytes = std::max<std::size_t>(4U * (global_size + 2U) * sizeof(float), 1U << 20U);
    space = comm_playground_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    space_created = true;

    auto* device_x = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, (global_size + 2U) * sizeof(float)));
    auto* device_y = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, global_size * sizeof(float)));
    if (device_x == nullptr || device_y == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI CUDA symmetric memory");
    }
    elapsed_by_pe = static_cast<double*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(double)));
    if (elapsed_by_pe == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI timing buffer");
    }

    check_cuda(cudaMemset(device_x, 0, (global_size + 2U) * sizeof(float)), "cudaMemset(device_x)");
    check_cuda(cudaMemset(device_y, 0, global_size * sizeof(float)), "cudaMemset(device_y)");
    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      init_interior_kernel<<<grid_size, block_size>>>(device_x, local_offset, local_size);
      check_cuda(cudaGetLastError(), "init_interior_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
    }

    shmem_barrier_all();
    comm_playground::wall_timer timer;

    if (local_size > 0) {
      if (left >= 0) {
        shmem_putmem(device_x + local_offset + 1U, device_x + local_offset + 1U, sizeof(float), left);
      }
      if (right >= 0) {
        shmem_putmem(device_x + local_offset + local_size, device_x + local_offset + local_size, sizeof(float), right);
      }
    }
    shmem_quiet();
    shmem_barrier_all();

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      halo_kernel<<<grid_size, block_size>>>(device_x, device_y, local_offset, local_size);
      check_cuda(cudaGetLastError(), "halo_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel)");
    }

    shmem_barrier_all();
    if (pe != 0 && local_size > 0) {
      shmem_putmem(device_y + local_offset, device_y + local_offset, local_size * sizeof(float), 0);
    }
    shmem_quiet();
    shmem_barrier_all();
    const auto elapsed = timer.seconds();

    shmem_putmem(elapsed_by_pe + pe, &elapsed, sizeof(elapsed), 0);
    shmem_quiet();
    shmem_barrier_all();

    double max_elapsed = elapsed;
    if (pe == 0) {
      max_elapsed = 0.0;
      for (int source_pe = 0; source_pe < pes; ++source_pe) {
        max_elapsed = std::max(max_elapsed, elapsed_by_pe[source_pe]);
      }
    }

    int global_ok = 1;
    if (pe == 0) {
      std::vector<float> host_y(global_size, 0.0F);
      check_cuda(cudaMemcpy(host_y.data(), device_y, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_y)");
      global_ok = comm_playground::validate_halo_1d(host_y.data(), global_size, 0, global_size) ? 1 : 0;
    }

    shmem_free(elapsed_by_pe);
    elapsed_by_pe = nullptr;
    shmem_free(device_y);
    shmem_free(device_x);
    comm_playground_oshmpi_space_destroy(space);
    space_created = false;

    if (pe == 0) {
      std::cout << "oshmpi_halo_1d n=" << global_size << " pes=" << pes << " time_s=" << max_elapsed
                << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
    }

    shmem_finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    if (space_created) {
      comm_playground_oshmpi_space_destroy(space);
    }
    shmem_global_exit(1);
  }
}
