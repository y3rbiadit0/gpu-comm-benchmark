#include <mpi.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
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
    const auto offset = offset_for_rank(global_size, rank, ranks);

    sycl::queue queue{sycl::default_selector_v};

    std::vector<float> host_a(local_size);
    std::vector<float> host_b(local_size);
    std::vector<float> host_c(local_size, 0.0F);

    for (std::size_t i = 0; i < local_size; ++i) {
      const auto value = static_cast<float>(offset + i);
      host_a[i] = value;
      host_b[i] = 2.0F * value;
    }

    float* device_a = sycl::malloc_device<float>(local_size, queue);
    float* device_b = sycl::malloc_device<float>(local_size, queue);
    float* device_c = sycl::malloc_device<float>(local_size, queue);

    if (device_a == nullptr || device_b == nullptr || device_c == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }

    queue.copy(host_a.data(), device_a, local_size).wait();
    queue.copy(host_b.data(), device_b, local_size).wait();

    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    queue.parallel_for(sycl::range<1>{local_size}, [=](sycl::id<1> id) {
      const auto i = id[0];
      device_c[i] = device_a[i] + device_b[i];
    }).wait();

    const auto elapsed = timer.seconds();

    queue.copy(device_c, host_c.data(), local_size).wait();

    const int local_ok = comm_playground::validate_vector_add(host_c.data(), local_size, offset) ? 1 : 0;
    int global_ok = 0;
    MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    sycl::free(device_a, queue);
    sycl::free(device_b, queue);
    sycl::free(device_c, queue);

    if (rank == 0) {
      std::cout << "sycl_mpi_vector_add n=" << global_size << " ranks=" << ranks
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
