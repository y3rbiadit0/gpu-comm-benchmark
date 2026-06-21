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

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void init_root_inputs_kernel(float* a, float* b, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto value = static_cast<float>(i);
    a[i] = value;
    b[i] = 2.0F * value;
  }
}

__global__ void distribute_inputs_kernel(float* a, float* b, std::uint64_t* ready, std::size_t n, int pe,
                                         int pes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  if (pe == 0) {
    for (int target_pe = 1; target_pe < pes; ++target_pe) {
      nvshmem_float_put(a, a, n, target_pe);
      nvshmem_float_put(b, b, n, target_pe);
      nvshmem_fence();
      nvshmemx_signal_op(ready, 1ULL, NVSHMEM_SIGNAL_SET, target_pe);
    }
  } else {
    nvshmem_signal_wait_until(ready, NVSHMEM_CMP_GE, 1ULL);
  }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t offset,
                                  std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = offset + i;
    c[global_i] = a[global_i] + b[global_i];
  }
}

__global__ void gather_results_kernel(float* c, std::uint64_t* gathered, std::size_t offset,
                                      std::size_t n, int pe, int pes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  if (pe == 0) {
    for (int source_pe = 1; source_pe < pes; ++source_pe) {
      nvshmem_signal_wait_until(gathered + source_pe, NVSHMEM_CMP_GE, 1ULL);
    }
  } else {
    if (n > 0) {
      nvshmem_float_put(c + offset, c + offset, n, 0);
    }
    nvshmem_fence();
    nvshmemx_signal_op(gathered + pe, 1ULL, NVSHMEM_SIGNAL_SET, 0);
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

    auto* device_a = static_cast<float*>(nvshmem_malloc(global_size * sizeof(float)));
    auto* device_b = static_cast<float*>(nvshmem_malloc(global_size * sizeof(float)));
    auto* device_c = static_cast<float*>(nvshmem_malloc(global_size * sizeof(float)));
    auto* ready = static_cast<std::uint64_t*>(nvshmem_malloc(sizeof(std::uint64_t)));
    auto* gathered = static_cast<std::uint64_t*>(
        nvshmem_malloc(static_cast<std::size_t>(pes) * sizeof(std::uint64_t)));
    if (device_a == nullptr || device_b == nullptr || device_c == nullptr || ready == nullptr ||
        gathered == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }

    check_cuda(cudaMemset(device_a, 0, global_size * sizeof(float)), "cudaMemset(device_a)");
    check_cuda(cudaMemset(device_b, 0, global_size * sizeof(float)), "cudaMemset(device_b)");
    check_cuda(cudaMemset(device_c, 0, global_size * sizeof(float)), "cudaMemset(device_c)");
    check_cuda(cudaMemset(ready, 0, sizeof(std::uint64_t)), "cudaMemset(ready)");
    check_cuda(cudaMemset(gathered, 0, static_cast<std::size_t>(pes) * sizeof(std::uint64_t)), "cudaMemset(gathered)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(memset)");

    if (pe == 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((global_size + block_size - 1U) / block_size);
      init_root_inputs_kernel<<<grid_size, block_size>>>(device_a, device_b, global_size);
      check_cuda(cudaGetLastError(), "init_root_inputs_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    nvshmem_barrier_all();
    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    distribute_inputs_kernel<<<1, 1>>>(device_a, device_b, ready, global_size, pe, pes);
    check_cuda(cudaGetLastError(), "distribute_inputs_kernel");

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      vector_add_kernel<<<grid_size, block_size>>>(device_a, device_b, device_c, local_offset, local_size);
      check_cuda(cudaGetLastError(), "vector_add_kernel");
    }

    gather_results_kernel<<<1, 1>>>(device_c, gathered, local_offset, local_size, pe, pes);
    check_cuda(cudaGetLastError(), "gather_results_kernel");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(gather)");
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (pe == 0) {
      std::vector<float> host_global_c(global_size, 0.0F);
      check_cuda(cudaMemcpy(host_global_c.data(), device_c, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_global_c)");
      global_ok = comm_playground::validate_vector_add(host_global_c.data(), global_size, 0) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    nvshmem_free(gathered);
    nvshmem_free(ready);
    nvshmem_free(device_a);
    nvshmem_free(device_b);
    nvshmem_free(device_c);
    nvshmem_finalize();
    nvshmem_initialized = false;

    if (pe == 0) {
      std::cout << "cuda_nvshmem_vector_add_device n=" << global_size << " pes=" << pes
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
