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
    const auto count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const int peer_count = mpi_count(count);
    const auto total = static_cast<std::size_t>(ranks) * count;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    std::vector<float> host_send(total);
    gpu_bench::fill_alltoall_send(host_send.data(), rank, ranks, count);

    float* device_send = nullptr;
    float* device_recv = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), total * sizeof(float)), "cudaMalloc(send)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), total * sizeof(float)), "cudaMalloc(recv)");
    check_cuda(cudaMemcpy(device_send, host_send.data(), total * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaMemset(device_recv, 0, total * sizeof(float)), "cudaMemset(recv)");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
      MPI_Alltoall(device_send, peer_count, MPI_FLOAT, device_recv, peer_count, MPI_FLOAT, MPI_COMM_WORLD);
    });

    double time_per_iter = 0.0;
    double min_time = 0.0;
    double max_time = 0.0;
    MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<float> host_recv(total);
    check_cuda(cudaMemcpy(host_recv.data(), device_recv, total * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy(recv)");
    int local_ok = gpu_bench::validate_alltoall(host_recv.data(),rank, ranks, count) ? 1 : 0;
    int global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    check_cuda(cudaFree(device_send), "cudaFree(send)");
    check_cuda(cudaFree(device_recv), "cudaFree(recv)");

    if (rank == 0) {
      gpu_bench::bench_report report;
      report.name = "cuda_mpi_alltoall";
      report.n = count;
      report.ranks = ranks;
      report.bytes_per_iter = total * sizeof(float);  // per-rank send volume
      report.iterations = iterations;
      report.warmup = warmup;
      report.time_per_iter_s = time_per_iter;
      report.min_s = min_time;
      report.max_s = max_time;
      report.valid = global_ok != 0;
      gpu_bench::print_report(report);
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
