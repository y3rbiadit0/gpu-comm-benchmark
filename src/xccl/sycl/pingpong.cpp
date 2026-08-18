#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// NOTE: point-to-point (ccl::send / ccl::recv) must be supported by the active
// oneCCL backend. The UNISA NCCL-enabled fork does not implement every primitive
// (e.g. broadcast); if pt2pt is likewise unimplemented this benchmark will report
// a backend error -- treat that as a documented gap rather than a bug here.

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
    if (ranks != 2) {
      throw std::runtime_error("pingpong requires exactly 2 ranks");
    }
    const auto max_elems = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elems);
    const int peer = rank == 0 ? 1 : 0;

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

    std::vector<float> host_send(max_elems);
    for (std::size_t i = 0; i < max_elems; ++i) {
      host_send[i] = static_cast<float>(i % 1024U);
    }

    float* device_send = sycl::malloc_device<float>(max_elems, queue);
    float* device_recv = sycl::malloc_device<float>(max_elems, queue);
    if (device_send == nullptr || device_recv == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }
    queue.copy(host_send.data(), device_send, max_elems).wait();

    for (std::size_t size : message_sizes) {
      const std::size_t count = size;

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        if (rank == 0) {
          ccl::send(device_send, count, ccl::datatype::float32, peer, comm, stream).wait();
          ccl::recv(device_recv, count, ccl::datatype::float32, peer, comm, stream).wait();
        } else {
          ccl::recv(device_recv, count, ccl::datatype::float32, peer, comm, stream).wait();
          ccl::send(device_recv, count, ccl::datatype::float32, peer, comm, stream).wait();
        }
      });

      if (rank == 0) {
        std::vector<float> host_recv(size);
        queue.copy(device_recv, host_recv.data(), size).wait();
        bool ok = true;
        for (std::size_t i = 0; i < size && ok; ++i) {
          ok = gpu_bench::nearly_equal(host_recv[i], host_send[i]);
        }

        gpu_bench::bench_report report;
        report.name = "sycl_oneccl_pingpong";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = size * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = 0.5 * stats.avg_s;
        report.min_s = 0.5 * stats.min_s;
        report.max_s = 0.5 * stats.max_s;
        gpu_bench::set_local_distribution(report, stats, 0.5);
        report.valid = ok;
        report.extra = "device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
        gpu_bench::print_report(report);
      }
    }

    sycl::free(device_send, queue);
    sycl::free(device_recv, queue);

    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
