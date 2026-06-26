#include <mpi.h>

#include <cuda_runtime.h>

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
#include "stencil2d.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void init_kernel(float* field, std::size_t side, std::size_t local_cols, std::size_t width,
                            std::size_t col_offset) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    const auto jg = col_offset + jj;
    field[i * width + (jj + 1U)] = static_cast<float>(i + jg);
  }
}

__global__ void pack_column_kernel(const float* padded, float* contiguous, std::size_t side, std::size_t width,
                                   std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    contiguous[i] = padded[i * width + column];
  }
}

__global__ void unpack_column_kernel(float* padded, const float* contiguous, std::size_t side, std::size_t width,
                                     std::size_t column) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < side) {
    padded[i * width + column] = contiguous[i];
  }
}

__global__ void stencil_kernel(const float* old_field, float* new_field, std::size_t side,
                               std::size_t local_cols, std::size_t width) {
  const auto jj = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto i = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (jj < local_cols && i < side) {
    const auto j = jj + 1U;
    const float north = i > 0 ? old_field[(i - 1U) * width + j] : 0.0F;
    const float south = i + 1U < side ? old_field[(i + 1U) * width + j] : 0.0F;
    const float west = old_field[i * width + (j - 1U)];
    const float east = old_field[i * width + (j + 1U)];
    new_field[i * width + j] = 0.25F * (north + south + west + east);
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
    const auto side = comm_playground::parse_size_arg(argc, argv, 1U << 12U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 50);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 10);
    const auto local_cols = comm_playground::local_count(side, rank, ranks);
    const auto col_offset = comm_playground::local_offset(side, rank, ranks);
    const auto width = local_cols + 2U;
    const int left = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int right = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int column_count = static_cast<int>(side);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    float* old_field = nullptr;
    float* new_field = nullptr;
    float* send_west = nullptr;
    float* send_east = nullptr;
    float* recv_west = nullptr;
    float* recv_east = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&old_field), side * width * sizeof(float)), "cudaMalloc(old)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&new_field), side * width * sizeof(float)), "cudaMalloc(new)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&send_west), side * sizeof(float)), "cudaMalloc(send_west)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&send_east), side * sizeof(float)), "cudaMalloc(send_east)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&recv_west), side * sizeof(float)), "cudaMalloc(recv_west)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&recv_east), side * sizeof(float)), "cudaMalloc(recv_east)");

    check_cuda(cudaMemset(old_field, 0, side * width * sizeof(float)), "cudaMemset(old)");
    check_cuda(cudaMemset(new_field, 0, side * width * sizeof(float)), "cudaMemset(new)");
    check_cuda(cudaMemset(recv_west, 0, side * sizeof(float)), "cudaMemset(recv_west)");
    check_cuda(cudaMemset(recv_east, 0, side * sizeof(float)), "cudaMemset(recv_east)");

    const dim3 block2d(16, 16);
    const dim3 grid2d(static_cast<unsigned>((local_cols + block2d.x - 1U) / block2d.x),
                      static_cast<unsigned>((side + block2d.y - 1U) / block2d.y));
    constexpr int block1d = 256;
    const auto grid1d = static_cast<int>((side + block1d - 1) / block1d);

    if (local_cols > 0) {
      init_kernel<<<grid2d, block2d>>>(old_field, side, local_cols, width, col_offset);
      check_cuda(cudaGetLastError(), "init_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      if (local_cols > 0) {
        pack_column_kernel<<<grid1d, block1d>>>(old_field, send_west, side, width, 1U);
        pack_column_kernel<<<grid1d, block1d>>>(old_field, send_east, side, width, local_cols);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pack)");
      }
      // Ring exchange (CUDA-aware MPI on device buffers): west column travels left
      // and arrives as the east ghost of the left neighbour, and vice versa.
      MPI_Sendrecv(send_west, column_count, MPI_FLOAT, left, 0, recv_east, column_count, MPI_FLOAT, right, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(send_east, column_count, MPI_FLOAT, right, 1, recv_west, column_count, MPI_FLOAT, left, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (local_cols > 0) {
        unpack_column_kernel<<<grid1d, block1d>>>(old_field, recv_west, side, width, 0U);
        unpack_column_kernel<<<grid1d, block1d>>>(old_field, recv_east, side, width, local_cols + 1U);
        stencil_kernel<<<grid2d, block2d>>>(old_field, new_field, side, local_cols, width);
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(stencil)");
      }
    });

    double time_per_iter = 0.0;
    double min_time = 0.0;
    double max_time = 0.0;
    MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int local_ok = 1;
    if (local_cols > 0) {
      std::vector<float> host_new(side * width);
      check_cuda(cudaMemcpy(host_new.data(), new_field, side * width * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(new)");
      const auto field = [](std::size_t i, std::size_t jg) { return static_cast<float>(i + jg); };
      local_ok = comm_playground::validate_columns(
                     host_new.data(), side, local_cols, width, col_offset,
                     [&](std::size_t i, std::size_t jg) { return comm_playground::stencil5(i, jg, side, field); })
                     ? 1
                     : 0;
    }
    int global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    check_cuda(cudaFree(old_field), "cudaFree(old)");
    check_cuda(cudaFree(new_field), "cudaFree(new)");
    check_cuda(cudaFree(send_west), "cudaFree(send_west)");
    check_cuda(cudaFree(send_east), "cudaFree(send_east)");
    check_cuda(cudaFree(recv_west), "cudaFree(recv_west)");
    check_cuda(cudaFree(recv_east), "cudaFree(recv_east)");

    if (rank == 0) {
      comm_playground::bench_report report;
      report.name = "cuda_mpi_halo_2d";
      report.n = side;
      report.ranks = ranks;
      report.bytes_per_iter = 2U * side * sizeof(float);  // two halo columns per interior rank
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
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
