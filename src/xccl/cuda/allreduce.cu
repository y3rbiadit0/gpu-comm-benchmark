#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstddef>
#include <exception>
#include <iostream>
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

void check_nccl(ncclResult_t status, const char* call) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(status));
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;

  try {
    const auto max_elements = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elements);
    const auto max_bytes =
        gpu_bench::checked_size_multiply(max_elements, sizeof(float), "allreduce allocation");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    ncclUniqueId id;
    if (rank == 0) {
      check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    check_nccl(ncclCommInitRank(&comm, ranks, id, rank), "ncclCommInitRank");

    std::vector<float> host_send(max_elements, static_cast<float>(rank + 1));
    std::vector<float> host_recv(max_elements);

    float* device_send = nullptr;
    float* device_recv = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), max_bytes), "cudaMalloc(send)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), max_bytes), "cudaMalloc(recv)");
    check_cuda(cudaMemcpy(device_send, host_send.data(), max_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");

    int all_sizes_ok = 1;
    for (std::size_t size : message_sizes) {
      const auto bytes = gpu_bench::checked_size_multiply(size, sizeof(float), "allreduce message");
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        check_nccl(ncclAllReduce(device_send, device_recv, size, ncclFloat, ncclSum, comm, stream),
                   "ncclAllReduce");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(allreduce)");
      });

      const auto global = gpu_bench::collective_stats(stats);
      const double time_per_iter = global.avg_s;

      check_cuda(cudaMemcpy(host_recv.data(), device_recv, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv)");
      const auto expected = static_cast<float>(static_cast<double>(ranks) * (ranks + 1.0) / 2.0);
      int local_ok = 1;
      for (std::size_t i = 0; i < size; ++i) {
        if (!gpu_bench::nearly_equal(host_recv[i], expected)) {
          local_ok = 0;
        }
      }
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sizes_ok = all_sizes_ok && global_ok;

      if (rank == 0) {
        const double algorithm_gbytes_per_s =
            time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9 : 0.0;
        const double bus_gbytes_per_s =
            algorithm_gbytes_per_s * 2.0 * static_cast<double>(ranks - 1) / static_cast<double>(ranks);

        gpu_bench::bench_report report;
        report.name = "cuda_nccl_allreduce";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
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
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    MPI_Finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    if (comm != nullptr) ncclCommAbort(comm);
    if (stream != nullptr) cudaStreamDestroy(stream);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
