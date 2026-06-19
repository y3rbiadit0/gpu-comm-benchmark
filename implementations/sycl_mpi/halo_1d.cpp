#include <mpi.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

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

  try {
    const auto global_size = comm_playground::parse_size_arg(argc, argv, 1U << 20U);
    const auto local_size = local_size_for_rank(global_size, rank, ranks);
    const auto local_offset = offset_for_rank(global_size, rank, ranks);
    const int left = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int right = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<float> global_y;
    if (rank == 0) {
      counts.resize(static_cast<std::size_t>(ranks));
      displacements.resize(static_cast<std::size_t>(ranks));
      for (int target_rank = 0; target_rank < ranks; ++target_rank) {
        counts[static_cast<std::size_t>(target_rank)] = mpi_count(local_size_for_rank(global_size, target_rank, ranks));
        displacements[static_cast<std::size_t>(target_rank)] = mpi_count(offset_for_rank(global_size, target_rank, ranks));
      }
      global_y.resize(global_size, 0.0F);
    }

    sycl::queue queue{sycl::default_selector_v};
    std::vector<float> host_x(local_size + 2U, 0.0F);
    std::vector<float> host_y(local_size, 0.0F);
    for (std::size_t i = 0; i < local_size; ++i) {
      host_x[i + 1U] = static_cast<float>(local_offset + i);
    }

    float* device_x = nullptr;
    float* device_y = nullptr;
    if (local_size > 0) {
      device_x = sycl::malloc_device<float>(local_size + 2U, queue);
      device_y = sycl::malloc_device<float>(local_size, queue);
      if (device_x == nullptr || device_y == nullptr) {
        throw std::runtime_error("failed to allocate SYCL device memory");
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    float send_left = local_size > 0 ? host_x[1] : 0.0F;
    float send_right = local_size > 0 ? host_x[local_size] : 0.0F;
    MPI_Sendrecv(&send_left, 1, MPI_FLOAT, left, 0, &host_x[local_size + 1U], 1, MPI_FLOAT, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&send_right, 1, MPI_FLOAT, right, 1, &host_x[0], 1, MPI_FLOAT, left, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

    if (local_size > 0) {
      queue.copy(host_x.data(), device_x, local_size + 2U).wait();
      queue.parallel_for(sycl::range<1>{local_size}, [=](sycl::id<1> id) {
        const auto i = id[0];
        const auto j = i + 1U;
        device_y[i] = 0.25F * device_x[j - 1U] + 0.5F * device_x[j] + 0.25F * device_x[j + 1U];
      }).wait();
      queue.copy(device_y, host_y.data(), local_size).wait();
    }

    MPI_Gatherv(host_y.data(), mpi_count(local_size), MPI_FLOAT, rank == 0 ? global_y.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_FLOAT, 0,
                MPI_COMM_WORLD);
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (rank == 0) {
      global_ok = comm_playground::validate_halo_1d(global_y.data(), global_size, 0, global_size) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (local_size > 0) {
      sycl::free(device_y, queue);
      sycl::free(device_x, queue);
    }

    if (rank == 0) {
      std::cout << "sycl_mpi_halo_1d n=" << global_size << " ranks=" << ranks
                << " device=\"" << queue.get_device().get_info<sycl::info::device::name>() << "\""
                << " time_s=" << max_elapsed << " validation=" << (global_ok ? "PASS" : "FAIL") << '\n';
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
