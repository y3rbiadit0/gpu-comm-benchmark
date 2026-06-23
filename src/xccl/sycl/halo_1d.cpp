#include <mpi.h>
#include <oneapi/ccl.hpp>
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

sycl::device device_for_rank(int rank) {
  const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (devices.empty()) {
    throw std::runtime_error("no SYCL GPU devices available");
  }
  return devices[static_cast<std::size_t>(rank) % devices.size()];
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

    ccl::init();
    sycl::device device = device_for_rank(rank);
    sycl::context context(device);
    sycl::queue queue(context, device, sycl::property::queue::in_order());

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type address;
    if (rank == 0) {
      kvs = ccl::create_main_kvs();
      address = kvs->get_address();
      MPI_Bcast(address.data(), address.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
      MPI_Bcast(address.data(), address.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
      kvs = ccl::create_kvs(address);
    }

    auto ccl_device = ccl::create_device(queue.get_device());
    auto ccl_context = ccl::create_context(queue.get_context());
    auto comm = ccl::create_communicator(ranks, rank, ccl_device, ccl_context, kvs);
    auto stream = ccl::create_stream(queue);

    float* device_x = sycl::malloc_device<float>(global_size + 2U, queue);
    float* device_full_x = sycl::malloc_device<float>(global_size + 2U, queue);
    float* device_y = sycl::malloc_device<float>(global_size, queue);
    float* device_result = sycl::malloc_device<float>(global_size, queue);
    if (device_x == nullptr || device_full_x == nullptr || device_y == nullptr || device_result == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }

    queue.memset(device_x, 0, (global_size + 2U) * sizeof(float)).wait();
    queue.memset(device_full_x, 0, (global_size + 2U) * sizeof(float)).wait();
    queue.memset(device_y, 0, global_size * sizeof(float)).wait();
    queue.memset(device_result, 0, global_size * sizeof(float)).wait();

    if (local_size > 0) {
      queue.parallel_for(sycl::range<1>{local_size}, [=](sycl::id<1> id) {
        const auto i = id[0];
        const auto global_i = local_offset + i;
        device_x[global_i + 1U] = static_cast<float>(global_i);
      }).wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    comm_playground::wall_timer timer;

    ccl::allreduce(device_x, device_full_x, global_size + 2U, ccl::datatype::float32, ccl::reduction::sum, comm,
                   stream).wait();

    if (local_size > 0) {
      queue.parallel_for(sycl::range<1>{local_size}, [=](sycl::id<1> id) {
        const auto i = id[0];
        const auto global_i = local_offset + i;
        device_y[global_i] = 0.25F * device_full_x[global_i] + 0.5F * device_full_x[global_i + 1U]
                             + 0.25F * device_full_x[global_i + 2U];
      }).wait();
    }

    ccl::allreduce(device_y, device_result, global_size, ccl::datatype::float32, ccl::reduction::sum, comm, stream)
        .wait();
    queue.wait();
    const auto elapsed = timer.seconds();

    int global_ok = 1;
    if (rank == 0) {
      std::vector<float> host_y(global_size, 0.0F);
      queue.copy(device_result, host_y.data(), global_size).wait();
      global_ok = comm_playground::validate_halo_1d(host_y.data(), global_size, 0, global_size) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    sycl::free(device_x, queue);
    sycl::free(device_full_x, queue);
    sycl::free(device_y, queue);
    sycl::free(device_result, queue);

    if (rank == 0) {
      std::cout << "sycl_oneccl_halo_1d n=" << global_size << " ranks=" << ranks
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
