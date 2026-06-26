#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
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

void check_nccl(ncclResult_t status, const char* call) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(status));
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
  MPI_Init(&argc, &argv);

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;

  try {
    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto local_size = comm_playground::local_count(global_size, rank, ranks);

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

    // a[i] = b[i] = 1 so the global dot product equals the global size exactly.
    std::vector<double> host_chunk(local_size, 1.0);

    double* device_a = nullptr;
    double* device_b = nullptr;
    double* device_partial = nullptr;
    double* device_result = nullptr;
    if (local_size > 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), local_size * sizeof(double)), "cudaMalloc(device_a)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), local_size * sizeof(double)), "cudaMalloc(device_b)");
      check_cuda(cudaMemcpy(device_a, host_chunk.data(), local_size * sizeof(double), cudaMemcpyHostToDevice),
                 "cudaMemcpy(device_a)");
      check_cuda(cudaMemcpy(device_b, host_chunk.data(), local_size * sizeof(double), cudaMemcpyHostToDevice),
                 "cudaMemcpy(device_b)");
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_partial), sizeof(double)), "cudaMalloc(device_partial)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_result), sizeof(double)), "cudaMalloc(device_result)");

    check_cuda(cudaMemset(device_partial, 0, sizeof(double)), "cudaMemset(device_partial)");
    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>(std::min<std::size_t>(
          (local_size + block_size - 1U) / block_size, 65535U));
      dot_partial_kernel<<<grid_size, block_size>>>(device_a, device_b, device_partial, local_size);
      check_cuda(cudaGetLastError(), "dot_partial_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(partial)");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      check_nccl(ncclAllReduce(device_partial, device_result, 1, ncclDouble, ncclSum, comm, stream),
                 "ncclAllReduce");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(allreduce)");
    });

    double time_per_iter = 0.0;
    double min_time = 0.0;
    double max_time = 0.0;
    MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_ok = 1;
    if (rank == 0) {
      double result = 0.0;
      check_cuda(cudaMemcpy(&result, device_result, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(result)");
      global_ok = comm_playground::nearly_equal(result, static_cast<double>(global_size)) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (local_size > 0) {
      check_cuda(cudaFree(device_a), "cudaFree(device_a)");
      check_cuda(cudaFree(device_b), "cudaFree(device_b)");
    }
    check_cuda(cudaFree(device_partial), "cudaFree(device_partial)");
    check_cuda(cudaFree(device_result), "cudaFree(device_result)");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    if (rank == 0) {
      comm_playground::bench_report report;
      report.name = "cuda_nccl_dot_product";
      report.n = global_size;
      report.ranks = ranks;
      report.bytes_per_iter = sizeof(double);
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
