#include <mpi.h>

#include <cuda_runtime.h>

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

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  try {
    if (ranks != 2) {
      throw std::runtime_error("pingpong requires exactly 2 ranks");
    }
    const auto max_elems = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elems);
    const int peer = rank == 0 ? 1 : 0;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    std::vector<float> host_send(max_elems);
    for (std::size_t i = 0; i < max_elems; ++i) {
      host_send[i] = static_cast<float>(i % 1024U);
    }

    float* device_send = nullptr;
    float* device_recv = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), max_elems * sizeof(float)), "cudaMalloc(send)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), max_elems * sizeof(float)), "cudaMalloc(recv)");
    check_cuda(cudaMemcpy(device_send, host_send.data(), max_elems * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");

    for (std::size_t size : message_sizes) {
      const int count = static_cast<int>(size);

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        if (rank == 0) {
          MPI_Send(device_send, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD);
          MPI_Recv(device_recv, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(device_recv, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Send(device_recv, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD);
        }
      });

      if (rank == 0) {
        std::vector<float> host_recv(size);
        check_cuda(cudaMemcpy(host_recv.data(), device_recv, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(recv)");
        bool ok = true;
        for (std::size_t i = 0; i < size && ok; ++i) {
          ok = gpu_bench::nearly_equal(host_recv[i], host_send[i]);
        }

        gpu_bench::bench_report report;
        report.name = "cuda_mpi_pingpong";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = size * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        // One-way latency = half the measured round trip.
        report.time_per_iter_s = 0.5 * stats.avg_s;
        report.min_s = 0.5 * stats.min_s;
        report.max_s = 0.5 * stats.max_s;
        gpu_bench::set_local_distribution(report, stats, 0.5);
        report.valid = ok;
        gpu_bench::print_report(report);
      }
    }

    check_cuda(cudaFree(device_send), "cudaFree(send)");
    check_cuda(cudaFree(device_recv), "cudaFree(recv)");

    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
