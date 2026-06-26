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
    if (ranks != 2) {
      throw std::runtime_error("pingpong requires exactly 2 ranks");
    }
    const auto max_elems = comm_playground::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = comm_playground::parse_size_list_arg(argc, argv, 4, max_elems);
    const int peer = rank == 0 ? 1 : 0;

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
      const std::size_t count = size;

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
        if (rank == 0) {
          check_nccl(ncclGroupStart(), "ncclGroupStart(send)");
          check_nccl(ncclSend(device_send, count, ncclFloat, peer, comm, stream), "ncclSend(ping)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(send)");
          check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(ping)");
          check_nccl(ncclGroupStart(), "ncclGroupStart(recv)");
          check_nccl(ncclRecv(device_recv, count, ncclFloat, peer, comm, stream), "ncclRecv(pong)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(recv)");
          check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(pong)");
        } else {
          check_nccl(ncclGroupStart(), "ncclGroupStart(recv)");
          check_nccl(ncclRecv(device_recv, count, ncclFloat, peer, comm, stream), "ncclRecv(ping)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(recv)");
          check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(ping)");
          check_nccl(ncclGroupStart(), "ncclGroupStart(send)");
          check_nccl(ncclSend(device_recv, count, ncclFloat, peer, comm, stream), "ncclSend(pong)");
          check_nccl(ncclGroupEnd(), "ncclGroupEnd(send)");
          check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(pong)");
        }
      });

      if (rank == 0) {
        std::vector<float> host_recv(size);
        check_cuda(cudaMemcpy(host_recv.data(), device_recv, size * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy(recv)");
        bool ok = true;
        for (std::size_t i = 0; i < size && ok; ++i) {
          ok = comm_playground::nearly_equal(host_recv[i], host_send[i]);
        }

        comm_playground::bench_report report;
        report.name = "cuda_nccl_pingpong";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = size * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = 0.5 * stats.avg_s;
        report.min_s = 0.5 * stats.min_s;
        report.max_s = 0.5 * stats.max_s;
        report.valid = ok;
        comm_playground::print_report(report);
      }
    }

    check_cuda(cudaFree(device_send), "cudaFree(send)");
    check_cuda(cudaFree(device_recv), "cudaFree(recv)");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    if (comm != nullptr) ncclCommAbort(comm);
    if (stream != nullptr) cudaStreamDestroy(stream);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
