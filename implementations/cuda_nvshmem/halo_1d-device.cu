#include <mpi.h>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

constexpr int left_signal = 0;
constexpr int right_signal = 1;

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void init_interior_kernel(float* x, std::size_t global_offset, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = global_offset + i;
    x[global_i + 1U] = static_cast<float>(global_i);
  }
}

__global__ void halo_exchange_kernel(float* x, std::uint64_t* signals, std::size_t global_offset,
                                     std::size_t local_size, int left, int right) {
  if (threadIdx.x != 0 || blockIdx.x != 0 || local_size == 0) {
    return;
  }

  if (left >= 0) {
    nvshmem_float_p(x + global_offset + 1U, x[global_offset + 1U], left);
    nvshmem_fence();
    nvshmemx_signal_op(signals + right_signal, 1ULL, NVSHMEM_SIGNAL_SET, left);
  }
  if (right >= 0) {
    nvshmem_float_p(x + global_offset + local_size, x[global_offset + local_size], right);
    nvshmem_fence();
    nvshmemx_signal_op(signals + left_signal, 1ULL, NVSHMEM_SIGNAL_SET, right);
  }

  if (left >= 0) {
    nvshmem_signal_wait_until(signals + left_signal, NVSHMEM_CMP_GE, 1ULL);
  }
  if (right >= 0) {
    nvshmem_signal_wait_until(signals + right_signal, NVSHMEM_CMP_GE, 1ULL);
  }
}

__global__ void halo_kernel(const float* x, float* y, std::size_t global_offset, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = global_offset + i;
    y[global_i] = 0.25F * x[global_i] + 0.5F * x[global_i + 1U] + 0.25F * x[global_i + 2U];
  }
}

std::size_t local_size_for_rank(std::size_t global_size, int rank, int ranks) {
  const auto base = global_size / static_cast<std::size_t>(ranks);
  const auto remainder = global_size % static_cast<std::size_t>(ranks);
  return base + (static_cast<std::size_t>(rank) < remainder ? 1U : 0U);
}

std::size_t offset_for_rank(std::size_t global_size, int rank, int ranks) {
  const auto base = global_size / static_cast<std::size_t>(ranks);
  const auto remainder = global_size % static_cast<std::size_t>(ranks);
  const auto rank_value = static_cast<std::size_t>(rank);
  return rank_value * base + std::min(rank_value, remainder);
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

    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto local_size = local_size_for_rank(global_size, pe, pes);
    const auto local_offset = offset_for_rank(global_size, pe, pes);
    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    auto* device_x = static_cast<float*>(nvshmem_malloc((global_size + 2U) * sizeof(float)));
    auto* device_y = static_cast<float*>(nvshmem_malloc(global_size * sizeof(float)));
    auto* signals = static_cast<std::uint64_t*>(nvshmem_malloc(2U * sizeof(std::uint64_t)));
    if (device_x == nullptr || device_y == nullptr || signals == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }

    check_cuda(cudaMemset(device_x, 0, (global_size + 2U) * sizeof(float)), "cudaMemset(device_x)");
    check_cuda(cudaMemset(device_y, 0, global_size * sizeof(float)), "cudaMemset(device_y)");
    check_cuda(cudaMemset(signals, 0, 2U * sizeof(std::uint64_t)), "cudaMemset(signals)");
    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      init_interior_kernel<<<grid_size, block_size>>>(device_x, local_offset, local_size);
      check_cuda(cudaGetLastError(), "init_interior_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    nvshmem_barrier_all();
    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    halo_exchange_kernel<<<1, 1>>>(device_x, signals, local_offset, local_size, left, right);
    check_cuda(cudaGetLastError(), "halo_exchange_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(halo_exchange)");

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      halo_kernel<<<grid_size, block_size>>>(device_x, device_y, local_offset, local_size);
      check_cuda(cudaGetLastError(), "halo_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(kernel)");
    }

    nvshmem_barrier_all();
    if (pe != 0 && local_size > 0) {
      nvshmem_float_put(device_y + local_offset, device_y + local_offset, local_size, 0);
    }
    nvshmem_quiet();
    nvshmem_barrier_all();
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (pe == 0) {
      std::vector<float> host_y(global_size, 0.0F);
      check_cuda(cudaMemcpy(host_y.data(), device_y, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_y)");
      global_ok = comm_playground::validate_halo_1d(host_y.data(), global_size, 0, global_size) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    nvshmem_free(signals);
    nvshmem_free(device_y);
    nvshmem_free(device_x);
    nvshmem_finalize();
    nvshmem_initialized = false;

    if (pe == 0) {
      std::cout << "cuda_nvshmem_halo_1d_device n=" << global_size << " pes=" << pes
                << " time_s=" << max_elapsed << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
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
