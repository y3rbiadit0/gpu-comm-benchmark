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
#include "partition.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void dot_partial_kernel(const double* a, const double* b, double* partial, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double local = 0.0;
  for (auto k = i; k < n; k += stride) {
    local += a[k] * b[k];
  }
  atomicAdd(partial, local);
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();

  double* sym_source = nullptr;
  double* sym_result = nullptr;
  double* pwrk = nullptr;
  long* psync = nullptr;
  double* stats_by_pe = nullptr;

  try {
    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto local_size = comm_playground::local_count(global_size, pe, pes);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    // a[i] = b[i] = 1 so the global dot product equals the global size exactly.
    // The local dot is computed on the GPU, then reduced across PEs on the host
    // symmetric heap -- the natural OSHMPI (SHMEM-over-MPI) model.
    std::vector<double> host_chunk(local_size, 1.0);

    double* device_a = nullptr;
    double* device_b = nullptr;
    double* device_partial = nullptr;
    if (local_size > 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), local_size * sizeof(double)), "cudaMalloc(device_a)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), local_size * sizeof(double)), "cudaMalloc(device_b)");
      check_cuda(cudaMemcpy(device_a, host_chunk.data(), local_size * sizeof(double), cudaMemcpyHostToDevice),
                 "cudaMemcpy(device_a)");
      check_cuda(cudaMemcpy(device_b, host_chunk.data(), local_size * sizeof(double), cudaMemcpyHostToDevice),
                 "cudaMemcpy(device_b)");
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_partial), sizeof(double)), "cudaMalloc(device_partial)");
    check_cuda(cudaMemset(device_partial, 0, sizeof(double)), "cudaMemset(device_partial)");
    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>(std::min<std::size_t>(
          (local_size + block_size - 1U) / block_size, 65535U));
      dot_partial_kernel<<<grid_size, block_size>>>(device_a, device_b, device_partial, local_size);
      check_cuda(cudaGetLastError(), "dot_partial_kernel");
    }
    double host_partial = 0.0;
    check_cuda(cudaMemcpy(&host_partial, device_partial, sizeof(double), cudaMemcpyDeviceToHost),
               "cudaMemcpy(host_partial)");

    sym_source = static_cast<double*>(shmem_malloc(sizeof(double)));
    sym_result = static_cast<double*>(shmem_malloc(sizeof(double)));
    pwrk = static_cast<double*>(shmem_malloc(SHMEM_REDUCE_MIN_WRKDATA_SIZE * sizeof(double)));
    psync = static_cast<long*>(shmem_malloc(SHMEM_REDUCE_SYNC_SIZE * sizeof(long)));
    stats_by_pe = static_cast<double*>(shmem_malloc(3U * static_cast<std::size_t>(pes) * sizeof(double)));
    if (sym_source == nullptr || sym_result == nullptr || pwrk == nullptr || psync == nullptr ||
        stats_by_pe == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }
    for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; ++i) {
      psync[i] = SHMEM_SYNC_VALUE;
    }
    *sym_source = host_partial;

    shmem_barrier_all();
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      shmem_double_sum_to_all(sym_result, sym_source, 1, 0, 0, pes, pwrk, psync);
    });

    const double local_stats[3] = {stats.avg_s, stats.min_s, stats.max_s};
    shmem_putmem(stats_by_pe + 3 * pe, local_stats, 3U * sizeof(double), 0);
    shmem_quiet();
    shmem_barrier_all();

    double time_per_iter = stats.avg_s;
    double min_time = stats.min_s;
    double max_time = stats.max_s;
    if (pe == 0) {
      for (int source_pe = 0; source_pe < pes; ++source_pe) {
        time_per_iter = std::max(time_per_iter, stats_by_pe[3 * source_pe + 0]);
        min_time = std::min(min_time, stats_by_pe[3 * source_pe + 1]);
        max_time = std::max(max_time, stats_by_pe[3 * source_pe + 2]);
      }
    }

    int global_ok = 1;
    if (pe == 0) {
      global_ok = comm_playground::nearly_equal(*sym_result, static_cast<double>(global_size)) ? 1 : 0;
    }

    if (local_size > 0) {
      check_cuda(cudaFree(device_a), "cudaFree(device_a)");
      check_cuda(cudaFree(device_b), "cudaFree(device_b)");
    }
    check_cuda(cudaFree(device_partial), "cudaFree(device_partial)");
    shmem_free(stats_by_pe);
    shmem_free(psync);
    shmem_free(pwrk);
    shmem_free(sym_result);
    shmem_free(sym_source);

    if (pe == 0) {
      comm_playground::bench_report report;
      report.name = "oshmpi_dot_product";
      report.n = global_size;
      report.ranks = pes;
      report.bytes_per_iter = sizeof(double);
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
    shmem_global_exit(1);
  }
}
