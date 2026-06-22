#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
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

void check_mpi(int status, const char* call) {
  if (status != MPI_SUCCESS) {
    throw std::runtime_error(std::string(call) + ": MPI error " + std::to_string(status));
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
    x[i + 1U] = static_cast<float>(global_offset + i);
  }
}

__global__ void halo_step_kernel(const float* in, float* out, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto j = i + 1U;
    out[j] = 0.25F * in[j - 1U] + 0.5F * in[j] + 0.25F * in[j + 1U];
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

int mpi_count(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("MPI count exceeds int range");
  }
  return static_cast<int>(value);
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

void init_halo_requests(float* buffer, std::size_t local_size, int halo_count, int left, int right,
                        std::array<MPI_Request, 4>& requests) {
  check_mpi(MPI_Recv_init(buffer + local_size + 1U, halo_count, MPI_FLOAT, right, 0, MPI_COMM_WORLD,
                          &requests[0]),
            "MPI_Recv_init(right ghost)");
  check_mpi(MPI_Send_init(buffer + 1U, halo_count, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &requests[1]),
            "MPI_Send_init(left boundary)");
  check_mpi(MPI_Recv_init(buffer, halo_count, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &requests[2]),
            "MPI_Recv_init(left ghost)");
  check_mpi(MPI_Send_init(buffer + local_size, halo_count, MPI_FLOAT, right, 1, MPI_COMM_WORLD,
                          &requests[3]),
            "MPI_Send_init(right boundary)");
}

void free_halo_requests(std::array<MPI_Request, 4>& requests) {
  for (auto& request : requests) {
    check_mpi(MPI_Request_free(&request), "MPI_Request_free");
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
    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = parse_iterations_arg(argc, argv, 100);
    const auto local_size = local_size_for_rank(global_size, rank, ranks);
    const auto local_offset = offset_for_rank(global_size, rank, ranks);
    const int left = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int right = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const auto local_count = mpi_count(local_size);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<float> global_result;
    if (rank == 0) {
      counts.resize(static_cast<std::size_t>(ranks));
      displacements.resize(static_cast<std::size_t>(ranks));
      for (int target_rank = 0; target_rank < ranks; ++target_rank) {
        counts[static_cast<std::size_t>(target_rank)] =
            mpi_count(local_size_for_rank(global_size, target_rank, ranks));
        displacements[static_cast<std::size_t>(target_rank)] =
            mpi_count(offset_for_rank(global_size, target_rank, ranks));
      }
      global_result.resize(global_size, 0.0F);
    }

    float* device_x0 = nullptr;
    float* device_x1 = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_x0), (local_size + 2U) * sizeof(float)),
               "cudaMalloc(device_x0)");
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_x1), (local_size + 2U) * sizeof(float)),
               "cudaMalloc(device_x1)");
    check_cuda(cudaMemset(device_x0, 0, (local_size + 2U) * sizeof(float)), "cudaMemset(device_x0)");
    check_cuda(cudaMemset(device_x1, 0, (local_size + 2U) * sizeof(float)), "cudaMemset(device_x1)");

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      init_interior_kernel<<<grid_size, block_size>>>(device_x0, local_offset, local_size);
      check_cuda(cudaGetLastError(), "init_interior_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

    const auto halo_count = local_size > 0 ? 1 : 0;
    std::array<MPI_Request, 4> x0_requests = {};
    std::array<MPI_Request, 4> x1_requests = {};
    init_halo_requests(device_x0, local_size, halo_count, left, right, x0_requests);
    init_halo_requests(device_x1, local_size, halo_count, left, right, x1_requests);

    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    constexpr int block_size = 256;
    const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
    for (int iteration = 0; iteration < iterations; ++iteration) {
      auto& requests = iteration % 2 == 0 ? x0_requests : x1_requests;
      auto* in = iteration % 2 == 0 ? device_x0 : device_x1;
      auto* out = iteration % 2 == 0 ? device_x1 : device_x0;

      check_mpi(MPI_Startall(static_cast<int>(requests.size()), requests.data()), "MPI_Startall");
      check_mpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE),
                "MPI_Waitall");

      if (local_size > 0) {
        halo_step_kernel<<<grid_size, block_size>>>(in, out, local_size);
        check_cuda(cudaGetLastError(), "halo_step_kernel");
      }
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(halo_step)");
    }

    auto* final_buffer = iterations % 2 == 0 ? device_x0 : device_x1;
    MPI_Gatherv(final_buffer + 1U, local_count, MPI_FLOAT, rank == 0 ? global_result.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (rank == 0) {
      global_ok = validate_halo_iterations(global_result.data(), global_size, iterations) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    free_halo_requests(x1_requests);
    free_halo_requests(x0_requests);
    check_cuda(cudaFree(device_x1), "cudaFree(device_x1)");
    check_cuda(cudaFree(device_x0), "cudaFree(device_x0)");

    if (rank == 0) {
      std::cout << "cuda_mpi_halo_1d_cuda_aware_persistent_iter n=" << global_size
                << " iterations=" << iterations << " ranks=" << ranks << " time_s=" << max_elapsed
                << " time_per_iter_s=" << (max_elapsed / static_cast<double>(iterations))
                << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
