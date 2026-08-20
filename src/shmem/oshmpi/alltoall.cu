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
#include "stats/collective_shmem.hpp"
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
  void* space = nullptr;
  double* sample_gather = nullptr;
  double* ok_by_pe = nullptr;

  try {
    const auto count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto total = static_cast<std::size_t>(pes) * count;
    const auto block_bytes = count * sizeof(float);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    const auto symmetric_bytes = std::max<std::size_t>(4U * total * sizeof(float), 1U << 20U);
    space = gpu_bench_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    auto* device_send = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, total * sizeof(float)));
    auto* device_recv = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, total * sizeof(float)));
    sample_gather = static_cast<double*>(
        shmem_malloc(gpu_bench::collective_gather_elements(pes, iterations) * sizeof(double)));
    ok_by_pe = static_cast<double*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(double)));
    if (device_send == nullptr || device_recv == nullptr || sample_gather == nullptr ||
        ok_by_pe == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }

    std::vector<float> host_send(total);
    gpu_bench::fill_alltoall_send(host_send.data(), pe, pes, count);
    check_cuda(cudaMemcpy(device_send, host_send.data(), total * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaMemset(device_recv, 0, total * sizeof(float)), "cudaMemset(recv)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
    shmem_barrier_all();

    const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
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
    const auto global = gpu_bench::collective_stats(stats, sample_gather, pe, pes);

    std::vector<float> host_recv(total);
    check_cuda(cudaMemcpy(host_recv.data(), device_recv, total * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy(recv)");
    const int local_ok = gpu_bench::validate_alltoall(host_recv.data(), pe, pes, count) ? 1 : 0;

    const double local_value = static_cast<double>(local_ok);
    shmem_putmem(ok_by_pe + pe, &local_value, sizeof(double), 0);
    shmem_quiet();
    shmem_barrier_all();

    int global_ok = local_ok;
    if (pe == 0) {
      global_ok = 1;
      for (int source_pe = 0; source_pe < pes; ++source_pe) {
        if (ok_by_pe[source_pe] < 0.5) {
          global_ok = 0;
        }
      }
    }

    shmem_free(ok_by_pe);
    shmem_free(sample_gather);
    // The space allocations go back before the space they came from does, as
    // OSHMPI's own CUDA-space test does.
    shmem_free(device_recv);
    shmem_free(device_send);
    gpu_bench_oshmpi_space_destroy(space);

    if (pe == 0) {
      gpu_bench::bench_report report;
      report.name = "oshmpi_alltoall";
      report.n = count;
      report.ranks = pes;
      report.bytes_per_iter = total * sizeof(float);
      report.iterations = iterations;
      report.warmup = warmup;
      report.time_per_iter_s = global.avg_s;
      report.min_s = global.min_s;
      report.max_s = global.max_s;
      gpu_bench::set_distribution(report, global);
      report.valid = global_ok != 0;
      gpu_bench::print_report(report);
    }

    shmem_finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    // Space cleanup is collective and is unsafe when another PE may still be
    // inside the operation that failed locally.
    shmem_global_exit(1);
  }
}
