#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "collective_stats_mpi.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// NOTE: uses the oneCCL all-to-all collective (ccl::alltoall). The UNISA
// NCCL-enabled oneCCL fork does not implement every primitive (e.g. broadcast);
// if alltoall is unimplemented this binary reports a backend error -- a documented
// gap, consistent with the other oneCCL benchmarks.

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
    const auto count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto total = static_cast<std::size_t>(ranks) * count;

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

    std::vector<float> host_send(total);
    gpu_bench::fill_alltoall_send(host_send.data(), rank, ranks, count);

    float* device_send = sycl::malloc_device<float>(total, queue);
    float* device_recv = sycl::malloc_device<float>(total, queue);
    if (device_send == nullptr || device_recv == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }
    queue.copy(host_send.data(), device_send, total).wait();
    queue.memset(device_recv, 0, total * sizeof(float)).wait();

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
      ccl::alltoall(device_send, device_recv, count, ccl::datatype::float32, comm, stream).wait();
    });

    const auto global = gpu_bench::collective_stats(stats);

    std::vector<float> host_recv(total);
    queue.copy(device_recv, host_recv.data(), total).wait();
    int local_ok = gpu_bench::validate_alltoall(host_recv.data(),rank, ranks, count) ? 1 : 0;
    int global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    sycl::free(device_send, queue);
    sycl::free(device_recv, queue);

    if (rank == 0) {
      gpu_bench::bench_report report;
      report.name = "sycl_oneccl_alltoall";
      report.n = count;
      report.ranks = ranks;
      report.bytes_per_iter = total * sizeof(float);
      report.iterations = iterations;
      report.warmup = warmup;
      report.time_per_iter_s = global.avg_s;
      report.min_s = global.min_s;
      report.max_s = global.max_s;
      gpu_bench::set_distribution(report, global);
      report.valid = global_ok != 0;
      report.extra = "device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
      gpu_bench::print_report(report);
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
