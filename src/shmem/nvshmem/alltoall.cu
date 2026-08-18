#include <mpi.h>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

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
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    nvshmem_initialized = true;

    const int pe = nvshmem_my_pe();
    const int pes = nvshmem_n_pes();
    if (pe != mpi_rank || pes != mpi_ranks) {
      throw std::runtime_error("NVSHMEM PE layout does not match MPI rank layout");
    }

    const auto count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto total = static_cast<std::size_t>(pes) * count;

    std::vector<float> host_send(total);
    gpu_bench::fill_alltoall_send(host_send.data(), pe, pes, count);

    auto* device_send = static_cast<float*>(nvshmem_malloc(total * sizeof(float)));
    auto* device_recv = static_cast<float*>(nvshmem_malloc(total * sizeof(float)));
    if (device_send == nullptr || device_recv == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }
    check_cuda(cudaMemcpy(device_send, host_send.data(), total * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaMemset(device_recv, 0, total * sizeof(float)), "cudaMemset(recv)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    nvshmem_barrier_all();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
      nvshmem_float_alltoall(NVSHMEM_TEAM_WORLD, device_recv, device_send, count);
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
    int local_ok = gpu_bench::validate_alltoall(host_recv.data(),pe, pes, count) ? 1 : 0;
    int global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    nvshmem_free(device_send);
    nvshmem_free(device_recv);
    nvshmem_finalize();
    nvshmem_initialized = false;

    if (pe == 0) {
      gpu_bench::bench_report report;
      report.name = "cuda_nvshmem_alltoall";
      report.n = count;
      report.ranks = pes;
      report.bytes_per_iter = total * sizeof(float);
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
    std::cerr << "rank " << mpi_rank << ": " << error.what() << '\n';
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
