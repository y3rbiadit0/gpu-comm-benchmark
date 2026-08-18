#include <mpi.h>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

void check_nvshmem(int status, const char* call) {
  if (status != 0) {
    throw std::runtime_error(std::string(call) + " failed with status " + std::to_string(status));
  }
}

bool validate_result(const float* values, std::size_t count, float expected) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!gpu_bench::nearly_equal(values[i], expected)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank = 0;
  int mpi_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
  bool nvshmem_initialized = false;

  try {
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(mpi_rank % device_count), "cudaSetDevice");

    nvshmemx_init_attr_t attr = {};
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    check_nvshmem(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr), "nvshmemx_init_attr");
    nvshmem_initialized = true;

    const int pe = nvshmem_my_pe();
    const int pes = nvshmem_n_pes();
    if (pe != mpi_rank || pes != mpi_ranks) {
      throw std::runtime_error("NVSHMEM PE layout does not match MPI rank layout");
    }

    const auto max_elements = gpu_bench::parse_size_arg(argc, argv, 4194304U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elements);
    const auto max_bytes =
        gpu_bench::checked_size_multiply(max_elements, sizeof(float), "allreduce allocation");

    std::vector<float> host_source(max_elements, static_cast<float>(pe + 1));
    std::vector<float> host_result(max_elements);
    auto* device_source = static_cast<float*>(nvshmem_malloc(max_bytes));
    auto* device_result = static_cast<float*>(nvshmem_malloc(max_bytes));
    if (device_source == nullptr || device_result == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }

    check_cuda(cudaMemcpy(device_source, host_source.data(), max_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(source)");
    check_cuda(cudaMemset(device_result, 0, max_bytes), "cudaMemset(result)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    const auto expected = static_cast<float>(static_cast<double>(pes) * (pes + 1) / 2.0);
    int all_sizes_ok = 1;
    for (const auto count : message_sizes) {
      const auto bytes = gpu_bench::checked_size_multiply(count, sizeof(float), "allreduce message");
      nvshmem_barrier_all();
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        check_nvshmem(nvshmem_float_sum_reduce(NVSHMEM_TEAM_WORLD, device_result, device_source, count),
                      "nvshmem_float_sum_reduce");
      });

      double time_per_iter = 0.0;
      double min_time = 0.0;
      double max_time = 0.0;
      MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      check_cuda(cudaMemcpy(host_result.data(), device_result, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(result)");
      int local_ok = validate_result(host_result.data(), count, expected) ? 1 : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sizes_ok = std::min(all_sizes_ok, global_ok);

      if (pe == 0) {
        const double algorithm_gbytes_per_s =
            time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9 : 0.0;
        const double bus_gbytes_per_s =
            algorithm_gbytes_per_s * 2.0 * static_cast<double>(pes - 1) / static_cast<double>(pes);

        gpu_bench::bench_report report;
        report.name = "cuda_nvshmem_allreduce";
        report.n = count;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = min_time;
        report.max_s = max_time;
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 reduction=sum bus_gbytes_per_s=" +
                       std::to_string(bus_gbytes_per_s) + " memory=device_symmetric";
        gpu_bench::print_report(report);
      }
    }

    nvshmem_free(device_result);
    nvshmem_free(device_source);
    nvshmem_finalize();
    nvshmem_initialized = false;

    MPI_Finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << mpi_rank << ": " << error.what() << '\n';
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
