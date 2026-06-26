#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "partition.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

namespace {

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
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto local_size = comm_playground::local_count(global_size, rank, ranks);

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

    // a[i] = b[i] = 1 so the global dot product equals the global size exactly.
    std::vector<double> host_chunk(local_size, 1.0);

    double* device_a = nullptr;
    double* device_b = nullptr;
    double* device_partial = sycl::malloc_device<double>(1, queue);
    double* device_result = sycl::malloc_device<double>(1, queue);
    if (device_partial == nullptr || device_result == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }
    if (local_size > 0) {
      device_a = sycl::malloc_device<double>(local_size, queue);
      device_b = sycl::malloc_device<double>(local_size, queue);
      if (device_a == nullptr || device_b == nullptr) {
        throw std::runtime_error("failed to allocate SYCL device memory");
      }
      queue.copy(host_chunk.data(), device_a, local_size).wait();
      queue.copy(host_chunk.data(), device_b, local_size).wait();
    }

    // Compute the local partial once; time only the cross-rank scalar allreduce.
    queue.memset(device_partial, 0, sizeof(double)).wait();
    if (local_size > 0) {
      queue.submit([&](sycl::handler& handler) {
        auto sum = sycl::reduction(device_partial, sycl::plus<double>());
        handler.parallel_for(sycl::range<1>{local_size}, sum, [=](sycl::id<1> id, auto& accumulator) {
          accumulator += device_a[id[0]] * device_b[id[0]];
        });
      }).wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      ccl::allreduce(device_partial, device_result, 1, ccl::datatype::float64, ccl::reduction::sum, comm, stream)
          .wait();
    });

    double time_per_iter = 0.0;
    double min_time = 0.0;
    double max_time = 0.0;
    MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_ok = 1;
    if (rank == 0) {
      double result = 0.0;
      queue.copy(device_result, &result, 1).wait();
      global_ok = comm_playground::nearly_equal(result, static_cast<double>(global_size)) ? 1 : 0;
    }
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (local_size > 0) {
      sycl::free(device_a, queue);
      sycl::free(device_b, queue);
    }
    sycl::free(device_partial, queue);
    sycl::free(device_result, queue);

    if (rank == 0) {
      comm_playground::bench_report report;
      report.name = "sycl_oneccl_dot_product";
      report.n = global_size;
      report.ranks = ranks;
      report.bytes_per_iter = sizeof(double);
      report.iterations = iterations;
      report.warmup = warmup;
      report.time_per_iter_s = time_per_iter;
      report.min_s = min_time;
      report.max_s = max_time;
      report.valid = global_ok != 0;
      report.extra = "device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
      comm_playground::print_report(report);
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
