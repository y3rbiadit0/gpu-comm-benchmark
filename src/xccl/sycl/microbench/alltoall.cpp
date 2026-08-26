#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "stats/collective_mpi.hpp"
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
    const auto max_count = gpu_bench::parse_size_arg(argc, argv, 1U << 16U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_count);
    const auto max_total = static_cast<std::size_t>(ranks) * max_count;

    ccl::init();

    sycl::device device = device_for_rank(rank);
    sycl::context context(device);
    sycl::queue queue(context, device, sycl::property::queue::in_order());

    int all_sizes_ok = 1;  // must outlive the oneCCL scope below

    // oneCCL's communicator, stream and KVS hold resources bound to the MPI
    // endpoints underneath -- for the OSHMPI backend, shared-memory segments
    // that UCX still has endpoints into. They must therefore be destroyed
    // *before* MPI_Finalize. Declared at try-block scope they outlived it, and
    // MPI_Finalize tore down UCX while they were still alive: allreduce 2n4g
    // completed and validated all 23 sizes, then died with SIGBUS in
    // uct_mm_ep_flush. This scope closes them first.
    {
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

    // Allocated once at the largest swept size; each size uses the leading
    // ranks*count elements, which matches the per-peer block layout.
    float* device_send = sycl::malloc_device<float>(max_total, queue);
    float* device_recv = sycl::malloc_device<float>(max_total, queue);
    if (device_send == nullptr || device_recv == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }

    std::vector<float> host_send(max_total);
    std::vector<float> host_recv(max_total);
    for (const auto count : message_sizes) {
      const auto total = static_cast<std::size_t>(ranks) * count;
      const auto bytes = total * sizeof(float);

      // The per-peer block layout depends on count, so refill for each size.
      gpu_bench::fill_alltoall_send(host_send.data(), rank, ranks, count);
      queue.copy(host_send.data(), device_send, total).wait();
      queue.memset(device_recv, 0, bytes).wait();

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        ccl::alltoall(device_send, device_recv, count, ccl::datatype::float32, comm, stream).wait();
      });

      const auto global = gpu_bench::collective_stats(stats);

      queue.copy(device_recv, host_recv.data(), total).wait();
      const int local_ok = gpu_bench::validate_alltoall(host_recv.data(), rank, ranks, count) ? 1 : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_sizes_ok = all_sizes_ok && global_ok;

      if (rank == 0) {
        gpu_bench::bench_report report;
        report.name = "sycl_oneccl_alltoall";
        report.n = count;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = global.avg_s;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 count_per_peer=" + std::to_string(count) +
                       " bus_gbytes_per_s=" + std::to_string(gpu_bench::alltoall_bus_gbytes_per_s(
                                                  bytes, global.avg_s, ranks)) +
                       " device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
        gpu_bench::print_report(report);
      }
    }

    sycl::free(device_send, queue);
    sycl::free(device_recv, queue);

    }  // oneCCL objects released here, before MPI_Finalize

    MPI_Finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
