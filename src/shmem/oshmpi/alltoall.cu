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
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
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
    const auto count = comm_playground::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto total = static_cast<std::size_t>(pes) * count;
    const auto block_bytes = count * sizeof(float);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    const auto symmetric_bytes = std::max<std::size_t>(4U * total * sizeof(float), 1U << 20U);
    space = comm_playground_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    space_created = true;

    auto* device_send = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, total * sizeof(float)));
    auto* device_recv = static_cast<float*>(comm_playground_oshmpi_space_malloc(space, total * sizeof(float)));
    stats_by_pe = static_cast<double*>(shmem_malloc(4U * static_cast<std::size_t>(pes) * sizeof(double)));
    if (device_send == nullptr || device_recv == nullptr || stats_by_pe == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }

    std::vector<float> host_send(total);
    comm_playground::fill_alltoall_send(host_send.data(), pe, pes, count);
    check_cuda(cudaMemcpy(device_send, host_send.data(), total * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaMemset(device_recv, 0, total * sizeof(float)), "cudaMemset(recv)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
    shmem_barrier_all();

    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      // One-sided all-to-all: put my block-for-dst into dst's recv slot for me.
      for (int dst = 0; dst < pes; ++dst) {
        float* dest = device_recv + static_cast<std::size_t>(pe) * count;
        const float* src = device_send + static_cast<std::size_t>(dst) * count;
        if (dst == pe) {
          check_cuda(cudaMemcpy(dest, src, block_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy(self)");
        } else {
          shmem_putmem(dest, src, block_bytes, dst);
        }
      }
      shmem_quiet();
      shmem_barrier_all();
    });

    std::vector<float> host_recv(total);
    check_cuda(cudaMemcpy(host_recv.data(), device_recv, total * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy(recv)");
    int local_ok = comm_playground::validate_alltoall(host_recv.data(),pe, pes, count) ? 1 : 0;

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

    shmem_free(stats_by_pe);
    comm_playground_oshmpi_space_destroy(space);
    space_created = false;

    if (pe == 0) {
      comm_playground::bench_report report;
      report.name = "oshmpi_alltoall";
      report.n = count;
      report.ranks = pes;
      report.bytes_per_iter = total * sizeof(float);
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
