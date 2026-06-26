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
    const auto count = comm_playground::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto total = static_cast<std::size_t>(ranks) * count;

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

    std::vector<float> host_send(total);
    comm_playground::fill_alltoall_send(host_send.data(), rank, ranks, count);

    float* device_send = nullptr;
    float* device_recv = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), total * sizeof(float)), "cudaMalloc(send)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_recv), total * sizeof(float)), "cudaMalloc(recv)");
    check_cuda(cudaMemcpy(device_send, host_send.data(), total * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaMemset(device_recv, 0, total * sizeof(float)), "cudaMemset(recv)");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      // NCCL has no native all-to-all; emulate with a grouped send/recv to every peer.
      check_nccl(ncclGroupStart(), "ncclGroupStart(alltoall)");
      for (int peer = 0; peer < ranks; ++peer) {
        check_nccl(ncclSend(device_send + static_cast<std::size_t>(peer) * count, count, ncclFloat, peer, comm,
                            stream),
                   "ncclSend(alltoall)");
        check_nccl(ncclRecv(device_recv + static_cast<std::size_t>(peer) * count, count, ncclFloat, peer, comm,
                            stream),
                   "ncclRecv(alltoall)");
      }
      check_nccl(ncclGroupEnd(), "ncclGroupEnd(alltoall)");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(alltoall)");
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
    int local_ok = comm_playground::validate_alltoall(host_recv.data(),rank, ranks, count) ? 1 : 0;
    int global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    check_cuda(cudaFree(device_send), "cudaFree(send)");
    check_cuda(cudaFree(device_recv), "cudaFree(recv)");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    if (rank == 0) {
      comm_playground::bench_report report;
      report.name = "cuda_nccl_alltoall";
      report.n = count;
      report.ranks = ranks;
      report.bytes_per_iter = total * sizeof(float);
      report.iterations = iterations;
      report.warmup = warmup;
      report.time_per_iter_s = time_per_iter;
      report.min_s = min_time;
      report.max_s = max_time;
      report.valid = global_ok != 0;
      comm_playground::print_report(report);
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    if (comm != nullptr) ncclCommAbort(comm);
    if (stream != nullptr) cudaStreamDestroy(stream);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
