#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>

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

// Comm-only 1D halo exchange benchmark, NCCL.
//
// Periodic ring with a swept halo width H. NCCL has no halo collective, so the
// exchange is modelled with grouped point-to-point ncclSend/ncclRecv. Buffers
// are slice-local and GPU-resident:
//
//   [ left_halo(cap) | interior(2*cap) | right_halo(cap) ]   cap = max halo width
//
// Interior markers (left/right boundary, exact in float) let each rank validate
// its received halos locally. MPI is used only for bootstrap (ncclUniqueId
// broadcast) and for cross-rank reductions of the timing/validation result. The
// reported GB/s is send+receive ("bus") bandwidth: 4 * H * sizeof(float).

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

__global__ void fill_interior_kernel(float* interior, std::size_t n_local, std::size_t half,
                                     float left_marker, float right_marker) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_local) {
    interior[i] = i < half ? left_marker : right_marker;
  }
}

int nccl_count(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("NCCL count exceeds int range");
  }
  return static_cast<int>(value);
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
    if (ranks < 2) {
      throw std::runtime_error("ring halo exchange requires at least 2 ranks");
    }

    const auto max_halo = gpu_bench::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto halo_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_halo);
    const int batch_samples = gpu_bench::parse_positive_int_env("GPU_BENCH_BATCH_SAMPLES", 10);
    const int isolated_samples =
        gpu_bench::parse_positive_int_env("GPU_BENCH_ISOLATED_SAMPLES", 100);
    const auto batch_counts = gpu_bench::batch_iteration_counts(iterations);

    const int left = (rank - 1 + ranks) % ranks;
    const int right = (rank + 1) % ranks;

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

    const std::size_t cap = max_halo;
    const std::size_t n_local = 2U * cap;
    const std::size_t total = n_local + 2U * cap;
    const float left_marker = static_cast<float>(2 * (rank + 1));
    const float right_marker = static_cast<float>(2 * (rank + 1) + 1);
    const float expect_left = static_cast<float>(2 * (left + 1) + 1);
    const float expect_right = static_cast<float>(2 * (right + 1));

    float* buf = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&buf), total * sizeof(float)), "cudaMalloc(buf)");
    check_cuda(cudaMemset(buf, 0, total * sizeof(float)), "cudaMemset(buf)");
    float* interior = buf + cap;
    {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((n_local + block_size - 1U) / block_size);
      fill_interior_kernel<<<grid_size, block_size, 0, stream>>>(interior, n_local, cap, left_marker,
                                                                 right_marker);
      check_cuda(cudaGetLastError(), "fill_interior_kernel");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(fill)");
    }

    std::vector<float> host_left;
    std::vector<float> host_right;
    int all_cases_ok = 1;

    for (const std::size_t halo : halo_sizes) {
      const int count = nccl_count(halo);
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;

      for (const int batch_iters : batch_counts) {
        const int samples =
            gpu_bench::batch_samples_for(batch_iters, batch_samples, isolated_samples);
        const char* case_name = batch_iters == 1 ? "isolated" : "steady";
        int local_ok = 1;
        const auto stats = gpu_bench::run_batched_benchmark(
            warmup, batch_iters, samples,
            [&]() {
              check_cuda(cudaMemsetAsync(recv_left, 0, halo * sizeof(float), stream),
                         "cudaMemsetAsync(recv_left)");
              check_cuda(cudaMemsetAsync(recv_right, 0, halo * sizeof(float), stream),
                         "cudaMemsetAsync(recv_right)");
              check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(poison)");
              MPI_Barrier(MPI_COMM_WORLD);
            },
            [&](int exchange_count) {
              for (int i = 0; i < exchange_count; ++i) {
                check_nccl(ncclGroupStart(), "ncclGroupStart");
                check_nccl(ncclRecv(recv_left, count, ncclFloat, left, comm, stream),
                           "ncclRecv(left halo)");
                check_nccl(ncclRecv(recv_right, count, ncclFloat, right, comm, stream),
                           "ncclRecv(right halo)");
                check_nccl(ncclSend(send_right, count, ncclFloat, right, comm, stream),
                           "ncclSend(right boundary)");
                check_nccl(ncclSend(send_left, count, ncclFloat, left, comm, stream),
                           "ncclSend(left boundary)");
                check_nccl(ncclGroupEnd(), "ncclGroupEnd");
              }
              check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(halo batch)");
            },
            [&]() {
              host_left.assign(halo, 0.0F);
              host_right.assign(halo, 0.0F);
              check_cuda(cudaMemcpy(host_left.data(), recv_left, halo * sizeof(float),
                                    cudaMemcpyDeviceToHost), "cudaMemcpy(recv_left)");
              check_cuda(cudaMemcpy(host_right.data(), recv_right, halo * sizeof(float),
                                    cudaMemcpyDeviceToHost), "cudaMemcpy(recv_right)");
              for (std::size_t i = 0; i < halo; ++i) {
                if (!gpu_bench::nearly_equal(host_left[i], expect_left) ||
                    !gpu_bench::nearly_equal(host_right[i], expect_right)) {
                  local_ok = 0;
                  break;
                }
              }
            });

        int global_ok = 1;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        all_cases_ok = all_cases_ok && global_ok;
        const auto global = gpu_bench::collective_stats(stats);

        if (rank == 0) {
          gpu_bench::bench_report report;
          report.name = "cuda_nccl_halo_1d";
          report.n = halo;
          report.ranks = ranks;
          report.bytes_per_iter = 4U * halo * sizeof(float);
          report.iterations = batch_iters;
          report.warmup = warmup;
          report.time_per_iter_s = global.avg_s;
          report.min_s = global.min_s;
          report.max_s = global.max_s;
          gpu_bench::set_distribution(report, global);
          report.valid = global_ok != 0;
          report.extra = std::string("case=") + case_name + " timing=batch batch_iters=" +
                         std::to_string(batch_iters) + " batch_samples=" +
                         std::to_string(samples) +
                         " submission=host-stream completion=stream-sync halo_elems=" +
                         std::to_string(halo) + " topology=ring bw=sendrecv";
          gpu_bench::print_report(report);
        }
      }
    }

    check_cuda(cudaFree(buf), "cudaFree(buf)");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    MPI_Finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    if (comm != nullptr) ncclCommAbort(comm);
    if (stream != nullptr) cudaStreamDestroy(stream);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
