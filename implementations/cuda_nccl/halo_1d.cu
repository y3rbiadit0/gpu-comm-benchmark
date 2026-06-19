#include <mpi.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstddef>
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

void check_nccl(ncclResult_t status, const char* call) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(call) + ": " + ncclGetErrorString(status));
  }
}

__global__ void init_interior_kernel(float* x, std::size_t global_offset, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i + 1U] = static_cast<float>(global_offset + i);
  }
}

__global__ void halo_kernel(const float* x, float* y, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto j = i + 1U;
    y[i] = 0.25F * x[j - 1U] + 0.5F * x[j] + 0.25F * x[j + 1U];
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
    const auto local_size = local_size_for_rank(global_size, rank, ranks);
    const auto local_offset = offset_for_rank(global_size, rank, ranks);
    const auto local_count = mpi_count(local_size);
    const int left = rank == 0 ? -1 : rank - 1;
    const int right = rank + 1 == ranks ? -1 : rank + 1;

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<float> host_global_y;
    if (rank == 0) {
      counts.resize(static_cast<std::size_t>(ranks));
      displacements.resize(static_cast<std::size_t>(ranks));
      for (int target_rank = 0; target_rank < ranks; ++target_rank) {
        counts[static_cast<std::size_t>(target_rank)] = mpi_count(local_size_for_rank(global_size, target_rank, ranks));
        displacements[static_cast<std::size_t>(target_rank)] = mpi_count(offset_for_rank(global_size, target_rank, ranks));
      }
      host_global_y.resize(global_size, 0.0F);
    }

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

    float* device_x = nullptr;
    float* device_y = nullptr;
    float* device_global_y = nullptr;
    if (local_size > 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_x), (local_size + 2U) * sizeof(float)),
                 "cudaMalloc(device_x)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_y), local_size * sizeof(float)), "cudaMalloc(device_y)");
      check_cuda(cudaMemsetAsync(device_x, 0, (local_size + 2U) * sizeof(float), stream), "cudaMemset(device_x)");
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      init_interior_kernel<<<grid_size, block_size, 0, stream>>>(device_x, local_offset, local_size);
      check_cuda(cudaGetLastError(), "init_interior_kernel");
    }
    if (rank == 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_global_y), global_size * sizeof(float)),
                 "cudaMalloc(device_global_y)");
      check_cuda(cudaMemsetAsync(device_global_y, 0, global_size * sizeof(float), stream),
                 "cudaMemset(device_global_y)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(init)");

    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    if (ranks > 1 && local_size > 0) {
      check_nccl(ncclGroupStart(), "ncclGroupStart(halo)");
      if (left >= 0) {
        check_nccl(ncclRecv(device_x, 1, ncclFloat, left, comm, stream), "ncclRecv(left ghost)");
        check_nccl(ncclSend(device_x + 1U, 1, ncclFloat, left, comm, stream), "ncclSend(left boundary)");
      }
      if (right >= 0) {
        check_nccl(ncclSend(device_x + local_size, 1, ncclFloat, right, comm, stream),
                   "ncclSend(right boundary)");
        check_nccl(ncclRecv(device_x + local_size + 1U, 1, ncclFloat, right, comm, stream),
                   "ncclRecv(right ghost)");
      }
      check_nccl(ncclGroupEnd(), "ncclGroupEnd(halo)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(halo)");

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      halo_kernel<<<grid_size, block_size, 0, stream>>>(device_x, device_y, local_size);
      check_cuda(cudaGetLastError(), "halo_kernel");
    }

    if (rank == 0 && local_size > 0) {
      check_cuda(cudaMemcpyAsync(device_global_y + local_offset, device_y, local_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream),
                 "cudaMemcpyAsync(root device_y)");
    }
    if (ranks > 1) {
      check_nccl(ncclGroupStart(), "ncclGroupStart(gather)");
      if (rank == 0) {
        for (int source_rank = 1; source_rank < ranks; ++source_rank) {
          const auto count = counts[static_cast<std::size_t>(source_rank)];
          const auto displacement = displacements[static_cast<std::size_t>(source_rank)];
          if (count > 0) {
            check_nccl(ncclRecv(device_global_y + displacement, count, ncclFloat, source_rank, comm, stream),
                       "ncclRecv(global_y)");
          }
        }
      } else if (local_count > 0) {
        check_nccl(ncclSend(device_y, local_count, ncclFloat, 0, comm, stream), "ncclSend(device_y)");
      }
      check_nccl(ncclGroupEnd(), "ncclGroupEnd(gather)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(gather)");
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (rank == 0) {
      check_cuda(cudaMemcpy(host_global_y.data(), device_global_y, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_global_y)");
      global_ok = comm_playground::validate_halo_1d(host_global_y.data(), global_size, 0, global_size) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (device_x != nullptr) check_cuda(cudaFree(device_x), "cudaFree(device_x)");
    if (device_y != nullptr) check_cuda(cudaFree(device_y), "cudaFree(device_y)");
    if (device_global_y != nullptr) check_cuda(cudaFree(device_global_y), "cudaFree(device_global_y)");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    if (rank == 0) {
      std::cout << "cuda_nccl_halo_1d n=" << global_size << " ranks=" << ranks << " time_s=" << max_elapsed
                << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
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
