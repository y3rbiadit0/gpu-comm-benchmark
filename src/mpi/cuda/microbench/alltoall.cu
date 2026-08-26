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
#include "stats/collective_mpi.hpp"
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
    throw std::runtime_error("alltoall count exceeds int range");
  }
  return static_cast<int>(value);
}

// Per-peer block value encodes (source rank, destination rank): on rank r the
// block destined for rank s is filled with r*ranks + s. After the exchange rank
// r's block received from rank s must equal s*ranks + r -- a full permutation check.
}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  try {
    const auto max_count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_count);
    const auto max_total = static_cast<std::size_t>(ranks) * max_count;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    // Allocated once at the largest swept size; each size uses the leading
    // ranks*count elements, which matches the per-peer block layout.
    float* device_send = nullptr;
    float* device_recv = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), max_total * sizeof(float)),
               "cudaMalloc(send)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), max_total * sizeof(float)),
               "cudaMalloc(recv)");

    std::vector<float> host_send(max_total);
    std::vector<float> host_recv(max_total);
    int all_sizes_ok = 1;

    for (const auto count : message_sizes) {
      const int peer_count = mpi_count(count);
      const auto total = static_cast<std::size_t>(ranks) * count;
      const auto bytes = total * sizeof(float);

      // The per-peer block layout depends on count, so refill for each size.
      gpu_bench::fill_alltoall_send(host_send.data(), rank, ranks, count);
      check_cuda(cudaMemcpy(device_send, host_send.data(), bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(send)");
      check_cuda(cudaMemset(device_recv, 0, bytes), "cudaMemset(recv)");

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        MPI_Alltoall(device_send, peer_count, MPI_FLOAT, device_recv, peer_count, MPI_FLOAT,
                     MPI_COMM_WORLD);
      });

      const auto global = gpu_bench::collective_stats(stats);

      check_cuda(cudaMemcpy(host_recv.data(), device_recv, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv)");
      const int local_ok = gpu_bench::validate_alltoall(host_recv.data(), rank, ranks, count) ? 1 : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sizes_ok = all_sizes_ok && global_ok;

      if (rank == 0) {
        gpu_bench::bench_report report;
        report.name = "cuda_mpi_alltoall";
        report.n = count;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;  // per-rank send volume, self-block included
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 count_per_peer=" + std::to_string(count) +
                       " bus_gbytes_per_s=" + std::to_string(gpu_bench::alltoall_bus_gbytes_per_s(
                                                  bytes, global.avg_s, ranks));
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
