#include <mpi.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "collective_stats_mpi.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

int mpi_count(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("allreduce count exceeds int range");
  }
  return static_cast<int>(value);
}

__global__ void fill_kernel(float* values, std::size_t count, float value) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    values[i] = value;
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  try {
    const auto max_elems = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elems);
    mpi_count(max_elems);
    const auto max_bytes =
        gpu_bench::checked_size_multiply(max_elems, sizeof(float), "allreduce allocation");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    float* device_send = nullptr;
    float* device_recv = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), max_bytes), "cudaMalloc(send)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), max_bytes), "cudaMalloc(recv)");

    int all_sizes_ok = 1;
    std::vector<float> host_recv;
    for (const std::size_t size : message_sizes) {
      const int count = mpi_count(size);
      const auto bytes = gpu_bench::checked_size_multiply(size, sizeof(float), "allreduce message");
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((size + block_size - 1U) / block_size);
      fill_kernel<<<grid_size, block_size>>>(device_send, size, static_cast<float>(rank + 1));
      check_cuda(cudaGetLastError(), "fill_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fill)");

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        MPI_Allreduce(device_send, device_recv, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      });

      host_recv.resize(size);
      check_cuda(cudaMemcpy(host_recv.data(), device_recv, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv)");
      const float expected = static_cast<float>(static_cast<double>(ranks) * (ranks + 1.0) / 2.0);
      int local_ok = 1;
      for (std::size_t i = 0; i < size; ++i) {
        if (!gpu_bench::nearly_equal(host_recv[i], expected)) {
          local_ok = 0;
          break;
        }
      }

      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
      const auto global = gpu_bench::collective_stats(stats);
      const double max_avg = global.avg_s;
      all_sizes_ok = all_sizes_ok && global_ok;

      if (rank == 0) {
        const double algorithm_gbytes_per_s =
            max_avg > 0.0 ? static_cast<double>(bytes) / max_avg / 1.0e9 : 0.0;
        const double bus_gbytes_per_s = ranks > 1
                                           ? algorithm_gbytes_per_s * 2.0 * static_cast<double>(ranks - 1) /
                                                 static_cast<double>(ranks)
                                           : 0.0;

        gpu_bench::bench_report report;
        report.name = "cuda_mpi_allreduce";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = max_avg;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 reduction=sum bus_gbytes_per_s=" + std::to_string(bus_gbytes_per_s);
        gpu_bench::print_report(report);
      }
    }

    check_cuda(cudaFree(device_send), "cudaFree(send)");
    check_cuda(cudaFree(device_recv), "cudaFree(recv)");

    MPI_Finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
