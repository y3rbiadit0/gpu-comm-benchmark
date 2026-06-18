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

__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t n) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
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
    throw std::runtime_error("vector chunk is too large for MPI int counts");
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

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<float> host_global_a;
    std::vector<float> host_global_b;
    std::vector<float> host_global_c;

    if (rank == 0) {
      counts.resize(static_cast<std::size_t>(ranks));
      displacements.resize(static_cast<std::size_t>(ranks));
      for (int target_rank = 0; target_rank < ranks; ++target_rank) {
        counts[static_cast<std::size_t>(target_rank)] =
            mpi_count(local_size_for_rank(global_size, target_rank, ranks));
        displacements[static_cast<std::size_t>(target_rank)] =
            mpi_count(offset_for_rank(global_size, target_rank, ranks));
      }

      host_global_a.resize(global_size);
      host_global_b.resize(global_size);
      host_global_c.resize(global_size, 0.0F);
      for (std::size_t i = 0; i < global_size; ++i) {
        const auto value = static_cast<float>(i);
        host_global_a[i] = value;
        host_global_b[i] = 2.0F * value;
      }
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

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    float* device_global_a = nullptr;
    float* device_global_b = nullptr;
    float* device_global_c = nullptr;

    if (local_size > 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_a), local_size * sizeof(float)), "cudaMalloc(device_a)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_b), local_size * sizeof(float)), "cudaMalloc(device_b)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_c), local_size * sizeof(float)), "cudaMalloc(device_c)");
    }

    if (rank == 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_global_a), global_size * sizeof(float)),
                 "cudaMalloc(device_global_a)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_global_b), global_size * sizeof(float)),
                 "cudaMalloc(device_global_b)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_global_c), global_size * sizeof(float)),
                 "cudaMalloc(device_global_c)");
      check_cuda(cudaMemcpyAsync(device_global_a, host_global_a.data(), global_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(device_global_a)");
      check_cuda(cudaMemcpyAsync(device_global_b, host_global_b.data(), global_size * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync(device_global_b)");
      check_cuda(cudaMemsetAsync(device_global_c, 0, global_size * sizeof(float), stream),
                 "cudaMemsetAsync(device_global_c)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(init)");

    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    if (rank == 0 && local_size > 0) {
      check_cuda(cudaMemcpyAsync(device_a, device_global_a + local_offset, local_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream),
                 "cudaMemcpyAsync(root device_a)");
      check_cuda(cudaMemcpyAsync(device_b, device_global_b + local_offset, local_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream),
                 "cudaMemcpyAsync(root device_b)");
    }

    if (ranks > 1) {
      check_nccl(ncclGroupStart(), "ncclGroupStart(scatter)");
      if (rank == 0) {
        for (int target_rank = 1; target_rank < ranks; ++target_rank) {
          const auto count = counts[static_cast<std::size_t>(target_rank)];
          const auto displacement = displacements[static_cast<std::size_t>(target_rank)];
          if (count > 0) {
            check_nccl(ncclSend(device_global_a + displacement, count, ncclFloat, target_rank, comm, stream),
                       "ncclSend(global_a)");
            check_nccl(ncclSend(device_global_b + displacement, count, ncclFloat, target_rank, comm, stream),
                       "ncclSend(global_b)");
          }
        }
      } else if (local_count > 0) {
        check_nccl(ncclRecv(device_a, local_count, ncclFloat, 0, comm, stream), "ncclRecv(device_a)");
        check_nccl(ncclRecv(device_b, local_count, ncclFloat, 0, comm, stream), "ncclRecv(device_b)");
      }
      check_nccl(ncclGroupEnd(), "ncclGroupEnd(scatter)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(scatter)");

    if (local_size > 0) {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((local_size + block_size - 1U) / block_size);
      vector_add_kernel<<<grid_size, block_size, 0, stream>>>(device_a, device_b, device_c, local_size);
      check_cuda(cudaGetLastError(), "vector_add_kernel");
    }

    if (rank == 0 && local_size > 0) {
      check_cuda(cudaMemcpyAsync(device_global_c + local_offset, device_c, local_size * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream),
                 "cudaMemcpyAsync(root device_c)");
    }

    if (ranks > 1) {
      check_nccl(ncclGroupStart(), "ncclGroupStart(gather)");
      if (rank == 0) {
        for (int source_rank = 1; source_rank < ranks; ++source_rank) {
          const auto count = counts[static_cast<std::size_t>(source_rank)];
          const auto displacement = displacements[static_cast<std::size_t>(source_rank)];
          if (count > 0) {
            check_nccl(ncclRecv(device_global_c + displacement, count, ncclFloat, source_rank, comm, stream),
                       "ncclRecv(global_c)");
          }
        }
      } else if (local_count > 0) {
        check_nccl(ncclSend(device_c, local_count, ncclFloat, 0, comm, stream), "ncclSend(device_c)");
      }
      check_nccl(ncclGroupEnd(), "ncclGroupEnd(gather)");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(gather)");

    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (rank == 0) {
      check_cuda(cudaMemcpy(host_global_c.data(), device_global_c, global_size * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy(host_global_c)");
      global_ok = comm_playground::validate_vector_add(host_global_c.data(), global_size, 0) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (device_a != nullptr) check_cuda(cudaFree(device_a), "cudaFree(device_a)");
    if (device_b != nullptr) check_cuda(cudaFree(device_b), "cudaFree(device_b)");
    if (device_c != nullptr) check_cuda(cudaFree(device_c), "cudaFree(device_c)");
    if (device_global_a != nullptr) check_cuda(cudaFree(device_global_a), "cudaFree(device_global_a)");
    if (device_global_b != nullptr) check_cuda(cudaFree(device_global_b), "cudaFree(device_global_b)");
    if (device_global_c != nullptr) check_cuda(cudaFree(device_global_c), "cudaFree(device_global_c)");
    check_nccl(ncclCommDestroy(comm), "ncclCommDestroy");
    check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    if (rank == 0) {
      std::cout << "cuda_nccl_vector_add n=" << global_size << " ranks=" << ranks
                << " time_s=" << max_elapsed << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
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
