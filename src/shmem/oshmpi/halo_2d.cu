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
#include "partition.hpp"
#include "report.hpp"
#include "stencil2d.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

// Halo column buffers live in the OSHMPI CUDA symmetric space; the field is plain device memory.

__global__ void init_kernel(float* field, std::size_t side, std::size_t local_cols, std::size_t width,
                            std::size_t col_offset) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    field[i * width + (jj + 1U)] = static_cast<float>(i + col_offset + jj);
  }
}

__global__ void pack_column_kernel(const float* padded, float* contiguous, std::size_t side, std::size_t width,
                                   std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    contiguous[i] = padded[i * width + column];
  }
}

__global__ void unpack_column_kernel(float* padded, const float* contiguous, std::size_t side, std::size_t width,
                                     std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    padded[i * width + column] = contiguous[i];
  }
}

__global__ void stencil_kernel(const float* old_field, float* new_field, std::size_t side,
                               std::size_t local_cols, std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    const auto j = jj + 1U;
    const float north = i > 0 ? old_field[(i - 1U) * width + j] : 0.0F;
    const float south = i + 1U < side ? old_field[(i + 1U) * width + j] : 0.0F;
    const float west = old_field[i * width + (j - 1U)];
    const float east = old_field[i * width + (j + 1U)];
    new_field[i * width + j] = 0.25F * (north + south + west + east);
  }
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();
  bool space_created = false;
  void* space = nullptr;
  double* stats_by_pe = nullptr;

  try {
    const auto side = comm_playground::parse_size_arg(argc, argv, 1U << 12U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 50);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 10);
    const auto local_cols = comm_playground::local_count(side, pe, pes);
    const auto col_offset = comm_playground::local_offset(side, pe, pes);
    const auto width = local_cols + 2U;
    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    const auto symmetric_bytes = std::max<std::size_t>(8U * side * sizeof(float), 1U << 20U);
    space = comm_playground_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    space_created = true;

    float* old_field = nullptr;
    float* new_field = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&old_field), side * width * sizeof(float)), "cudaMalloc(old)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&new_field), side * width * sizeof(float)), "cudaMalloc(new)");
    auto* send_west = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, side * sizeof(float)));
    auto* send_east = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, side * sizeof(float)));
    auto* recv_west = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, side * sizeof(float)));
    auto* recv_east = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, side * sizeof(float)));
    stats_by_pe = static_cast<double*>(shmem_malloc(4U * static_cast<std::size_t>(pes) * sizeof(double)));
    if (send_west == nullptr || send_east == nullptr || recv_west == nullptr || recv_east == nullptr ||
        stats_by_pe == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }

    check_cuda(cudaMemset(old_field, 0, side * width * sizeof(float)), "cudaMemset(old)");
    check_cuda(cudaMemset(new_field, 0, side * width * sizeof(float)), "cudaMemset(new)");
    check_cuda(cudaMemset(recv_west, 0, side * sizeof(float)), "cudaMemset(recv_west)");
    check_cuda(cudaMemset(recv_east, 0, side * sizeof(float)), "cudaMemset(recv_east)");

    const dim3 block2d(16, 16);
    const dim3 grid2d(static_cast<unsigned>((local_cols + block2d.x - 1U) / block2d.x),
                      static_cast<unsigned>((side + block2d.y - 1U) / block2d.y));
    constexpr int block1d = 256;
    const auto grid1d = static_cast<int>((side + block1d - 1) / block1d);

    if (local_cols > 0) {
      init_kernel<<<grid2d, block2d>>>(old_field, side, local_cols, width, col_offset);
      check_cuda(cudaGetLastError(), "init_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    shmem_barrier_all();
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      if (local_cols > 0) {
        pack_column_kernel<<<grid1d, block1d>>>(old_field, send_west, side, width, 1U);
        pack_column_kernel<<<grid1d, block1d>>>(old_field, send_east, side, width, local_cols);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pack)");
      }
      if (left >= 0) {
        shmem_putmem(recv_east, send_west, side * sizeof(float), left);
      }
      if (right >= 0) {
        shmem_putmem(recv_west, send_east, side * sizeof(float), right);
      }
      shmem_quiet();
      shmem_barrier_all();
      if (local_cols > 0) {
        unpack_column_kernel<<<grid1d, block1d>>>(old_field, recv_west, side, width, 0U);
        unpack_column_kernel<<<grid1d, block1d>>>(old_field, recv_east, side, width, local_cols + 1U);
        stencil_kernel<<<grid2d, block2d>>>(old_field, new_field, side, local_cols, width);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(stencil)");
      }
    });

    int local_ok = 1;
    if (local_cols > 0) {
      std::vector<float> host_new(side * width);
      check_cuda(cudaMemcpy(host_new.data(), new_field, side * width * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(new)");
      const auto field = [](std::size_t i, std::size_t jg) { return static_cast<float>(i + jg); };
      local_ok = comm_playground::validate_columns(
                     host_new.data(), side, local_cols, width, col_offset,
                     [&](std::size_t i, std::size_t jg) { return comm_playground::stencil5(i, jg, side, field); })
                     ? 1
                     : 0;
    }

    const double local_stats[4] = {stats.avg_s, stats.min_s, stats.max_s, static_cast<double>(local_ok)};
    shmem_putmem(stats_by_pe + 4 * pe, local_stats, 4U * sizeof(double), 0);
    shmem_quiet();
    shmem_barrier_all();

    double time_per_iter = stats.avg_s;
    double min_time = stats.min_s;
    double max_time = stats.max_s;
    int global_ok = local_ok;
    if (pe == 0) {
      global_ok = 1;
      for (int source_pe = 0; source_pe < pes; ++source_pe) {
        time_per_iter = std::max(time_per_iter, stats_by_pe[4 * source_pe + 0]);
        min_time = std::min(min_time, stats_by_pe[4 * source_pe + 1]);
        max_time = std::max(max_time, stats_by_pe[4 * source_pe + 2]);
        if (stats_by_pe[4 * source_pe + 3] < 0.5) {
          global_ok = 0;
        }
      }
    }

    check_cuda(cudaFree(new_field), "cudaFree(new)");
    check_cuda(cudaFree(old_field), "cudaFree(old)");
    shmem_free(stats_by_pe);
    comm_playground_oshmpi_space_destroy(space);
    space_created = false;

    if (pe == 0) {
      comm_playground::bench_report report;
      report.name = "oshmpi_halo_2d";
      report.n = side;
      report.ranks = pes;
      report.bytes_per_iter = 2U * side * sizeof(float);
      report.iterations = iterations;
      report.warmup = warmup;
      report.time_per_iter_s = time_per_iter;
      report.min_s = min_time;
      report.max_s = max_time;
      report.valid = global_ok != 0;
      comm_playground::print_report(report);
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
