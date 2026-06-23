#include <mpi.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

void check_nvshmem(int status, const char* call) {
  if (status != 0) {
    throw std::runtime_error(std::string(call) + ": NVSHMEM error " + std::to_string(status));
  }
}

int parse_iterations_arg(int argc, char** argv, int default_value) {
  if (argc < 3) {
    return default_value;
  }

  char* end = nullptr;
  const auto value = std::strtol(argv[2], &end, 10);
  if (end == argv[2] || *end != '\0' || value <= 0) {
    throw std::invalid_argument("expected a positive iteration count");
  }

  return static_cast<int>(value);
}

__global__ void init_interior_kernel(float* x, std::size_t global_offset, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto global_i = global_offset + i;
    x[global_i + 1U] = static_cast<float>(global_i);
  }
}

__global__ void halo_iter_kernel(float* x0, float* x1, std::uint64_t* signals,
                                 std::size_t global_offset, std::size_t local_size, int iterations,
                                 int left, int right) {
  auto grid = cooperative_groups::this_grid();
  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const bool leader = tid == 0;

  for (int iteration = 0; iteration < iterations; ++iteration) {
    auto* in = iteration % 2 == 0 ? x0 : x1;
    auto* out = iteration % 2 == 0 ? x1 : x0;
    const auto signal_value = static_cast<std::uint64_t>(iteration + 1);

    if (leader && local_size > 0) {
      if (left >= 0) {
        nvshmem_float_p(in + global_offset + 1U, in[global_offset + 1U], left);
        nvshmem_fence();
        nvshmemx_signal_op(signals + right_signal, signal_value, NVSHMEM_SIGNAL_SET, left);
      }
      if (right >= 0) {
        nvshmem_float_p(in + global_offset + local_size, in[global_offset + local_size], right);
        nvshmem_fence();
        nvshmemx_signal_op(signals + left_signal, signal_value, NVSHMEM_SIGNAL_SET, right);
      }

      if (left >= 0) {
        nvshmem_signal_wait_until(signals + left_signal, NVSHMEM_CMP_GE, signal_value);
      }
      if (right >= 0) {
        nvshmem_signal_wait_until(signals + right_signal, NVSHMEM_CMP_GE, signal_value);
      }
    }

    grid.sync();

    for (std::size_t i = tid; i < local_size; i += stride) {
      const auto global_i = global_offset + i;
      out[global_i + 1U] = 0.25F * in[global_i] + 0.5F * in[global_i + 1U] +
                           0.25F * in[global_i + 2U];
    }

    grid.sync();
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

int cooperative_grid_size(const void* kernel, int block_size, std::size_t local_size) {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");

  cudaDeviceProp prop = {};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  int active_blocks_per_sm = 0;
  check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, block_size, 0),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

  const auto requested = static_cast<int>((local_size + static_cast<std::size_t>(block_size) - 1U) /
                                          static_cast<std::size_t>(block_size));
  const auto resident = std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
  return std::max(1, std::min(requested, resident));
}

bool validate_halo_iterations(const float* values, std::size_t global_size, int iterations) {
  std::vector<float> current(global_size + 2U, 0.0F);
  std::vector<float> next(global_size + 2U, 0.0F);

  for (std::size_t i = 0; i < global_size; ++i) {
    current[i + 1U] = static_cast<float>(i);
  }

  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::fill(next.begin(), next.end(), 0.0F);
    for (std::size_t i = 0; i < global_size; ++i) {
      next[i + 1U] = 0.25F * current[i] + 0.5F * current[i + 1U] + 0.25F * current[i + 2U];
    }
    current.swap(next);
  }

  for (std::size_t i = 0; i < global_size; ++i) {
    if (!comm_playground::nearly_equal(values[i], current[i + 1U])) {
      return false;
    }
  }
  return true;
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
    const auto iterations = parse_iterations_arg(argc, argv, 100);
    const auto local_size = local_size_for_rank(global_size, pe, pes);
    const auto local_offset = offset_for_rank(global_size, pe, pes);
    const int left = pe == 0 ? -1 : pe - 1;
    const int right = pe + 1 == pes ? -1 : pe + 1;

    auto* device_x0 = static_cast<float*>(nvshmem_malloc((global_size + 2U) * sizeof(float)));
    auto* device_x1 = static_cast<float*>(nvshmem_malloc((global_size + 2U) * sizeof(float)));
    auto* device_result = static_cast<float*>(nvshmem_malloc(global_size * sizeof(float)));
    auto* signals = static_cast<std::uint64_t*>(nvshmem_malloc(2U * sizeof(std::uint64_t)));
    if (device_x0 == nullptr || device_x1 == nullptr || device_result == nullptr || signals == nullptr) {
      throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
    }

    check_cuda(cudaMemset(device_x0, 0, (global_size + 2U) * sizeof(float)), "cudaMemset(device_x0)");
    check_cuda(cudaMemset(device_x1, 0, (global_size + 2U) * sizeof(float)), "cudaMemset(device_x1)");
    check_cuda(cudaMemset(device_result, 0, global_size * sizeof(float)), "cudaMemset(device_result)");
    check_cuda(cudaMemset(signals, 0, 2U * sizeof(std::uint64_t)), "cudaMemset(signals)");
    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      init_interior_kernel<<<grid_size, block_size>>>(device_x0, local_offset, local_size);
      check_cuda(cudaGetLastError(), "init_interior_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    constexpr int block_size = 256;
    const auto grid_size = cooperative_grid_size((const void*)halo_iter_kernel, block_size, local_size);

    nvshmem_barrier_all();
    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    auto launch_local_offset = local_offset;
    auto launch_local_size = local_size;
    auto launch_iterations = iterations;
    auto launch_left = left;
    auto launch_right = right;
    void* args[] = {&device_x0, &device_x1, &signals, &launch_local_offset, &launch_local_size,
                    &launch_iterations, &launch_left, &launch_right};
    check_nvshmem(nvshmemx_collective_launch((const void*)halo_iter_kernel, dim3(grid_size),
                                             dim3(block_size), args, 0, 0),
                  "nvshmemx_collective_launch(halo_iter_kernel)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(halo_iter_kernel)");

    auto* final_buffer = iterations % 2 == 0 ? device_x0 : device_x1;
    if (pe == 0 && local_size > 0) {
      check_cuda(cudaMemcpy(device_result + local_offset, final_buffer + local_offset + 1U,
                            local_size * sizeof(float), cudaMemcpyDeviceToDevice),
                 "cudaMemcpy(root result)");
    }

    nvshmem_barrier_all();
    if (pe != 0 && local_size > 0) {
      nvshmem_float_put(device_result + local_offset, final_buffer + local_offset + 1U, local_size, 0);
    }
    nvshmem_quiet();
    nvshmem_barrier_all();
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (pe == 0) {
      std::vector<float> host_result(global_size, 0.0F);
      check_cuda(cudaMemcpy(host_result.data(), device_result, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_result)");
      global_ok = validate_halo_iterations(host_result.data(), global_size, iterations) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    nvshmem_free(signals);
    nvshmem_free(device_result);
    nvshmem_free(device_x1);
    nvshmem_free(device_x0);
    nvshmem_finalize();
    nvshmem_initialized = false;

    if (pe == 0) {
      std::cout << "cuda_nvshmem_halo_1d_device_iter n=" << global_size << " iterations=" << iterations
                << " pes=" << pes << " time_s=" << max_elapsed
                << " time_per_iter_s=" << (max_elapsed / static_cast<double>(iterations))
                << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
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
