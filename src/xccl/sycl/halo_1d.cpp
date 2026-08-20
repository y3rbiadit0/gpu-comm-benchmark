#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "stats/collective_mpi.hpp"
#include "oneccl.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// Comm-only 1D halo exchange benchmark, oneCCL.
//
// Periodic ring with a swept halo width H. The exchange is modelled with
// oneCCL point-to-point ccl::send/ccl::recv (the natural analog of NCCL
// ncclSend/ncclRecv), grouped so all neighbour operations are enqueued before
// execution - a true neighbour exchange rather than a collective emulation.
// Buffers are slice-local and device-resident:
//
//   [ left_halo(cap) | interior(2*cap) | right_halo(cap) ]   cap = max halo width
//
// Interior markers (left/right boundary, exact in float) let each rank validate
// its received halos locally. MPI is used only for bootstrap (KVS address
// broadcast) and cross-rank reductions. Reported GB/s is send+receive ("bus")
// bandwidth: 4 * H * sizeof(float).

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
    if (ranks < 2) {
      throw std::runtime_error("ring halo exchange requires at least 2 ranks");
    }

    const auto max_halo = gpu_bench::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto halo_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_halo);
    const int batch_samples = gpu_bench::parse_positive_int_env("GPU_BENCH_BATCH_SAMPLES", 10);
    const auto batch_counts = gpu_bench::batch_iteration_counts(iterations);

    const int left = (rank - 1 + ranks) % ranks;
    const int right = (rank + 1) % ranks;

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

    const std::size_t cap = max_halo;
    const std::size_t n_local = 2U * cap;
    const std::size_t total = n_local + 2U * cap;
    const float left_marker = static_cast<float>(2 * (rank + 1));
    const float right_marker = static_cast<float>(2 * (rank + 1) + 1);
    const float expect_left = static_cast<float>(2 * (left + 1) + 1);
    const float expect_right = static_cast<float>(2 * (right + 1));

    float* buf = sycl::malloc_device<float>(total, queue);
    if (buf == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }
    queue.memset(buf, 0, total * sizeof(float)).wait();
    float* interior = buf + cap;
    queue.parallel_for(sycl::range<1>{n_local}, [=](sycl::id<1> id) {
      const auto i = id[0];
      interior[i] = i < cap ? left_marker : right_marker;
    }).wait();

    std::vector<float> host_left;
    std::vector<float> host_right;
    int all_cases_ok = 1;

    for (const std::size_t halo : halo_sizes) {
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;

      for (const int batch_iters : batch_counts) {
        const char* case_name = batch_iters == 1 ? "isolated" : "steady";
        int local_ok = 1;
        const auto stats = gpu_bench::run_batched_benchmark(
            warmup, batch_iters, batch_samples,
            [&]() {
              queue.fill(recv_left, 0.0F, halo);
              queue.fill(recv_right, 0.0F, halo);
              queue.wait_and_throw();
              MPI_Barrier(MPI_COMM_WORLD);
            },
            [&](int exchange_count) {
              std::vector<ccl::event> events;
              events.reserve(static_cast<std::size_t>(exchange_count) * 4U);
              for (int i = 0; i < exchange_count; ++i) {
                // Grouping prevents an in-order stream from blocking on the first
                // unmatched receive before its matching sends have been enqueued.
                gpu_bench::ccl_group_scope group;
                events.push_back(ccl::recv(recv_left, halo, ccl::datatype::float32, left, comm, stream));
                events.push_back(ccl::recv(recv_right, halo, ccl::datatype::float32, right, comm, stream));
                events.push_back(ccl::send(send_right, halo, ccl::datatype::float32, right, comm, stream));
                events.push_back(ccl::send(send_left, halo, ccl::datatype::float32, left, comm, stream));
                group.end();
              }
              for (auto& event : events) {
                event.wait();
              }
              queue.wait_and_throw();
            },
            [&]() {
              host_left.assign(halo, 0.0F);
              host_right.assign(halo, 0.0F);
              queue.copy(recv_left, host_left.data(), halo).wait();
              queue.copy(recv_right, host_right.data(), halo).wait();
              for (std::size_t i = 0; i < halo; ++i) {
                if (!gpu_bench::nearly_equal(host_left[i], expect_left) ||
                    !gpu_bench::nearly_equal(host_right[i], expect_right)) {
                  local_ok = 0;
                  break;
                }
              }
            });

        int global_ok = 1;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        all_cases_ok = all_cases_ok && global_ok;
        const auto global = gpu_bench::collective_stats(stats);

        if (rank == 0) {
          gpu_bench::bench_report report;
          report.name = "sycl_oneccl_halo_1d";
          report.n = halo;
          report.ranks = ranks;
          report.bytes_per_iter = 4U * halo * sizeof(float);
          report.iterations = batch_iters;
          report.warmup = warmup;
          report.time_per_iter_s = global.avg_s;
          report.min_s = global.min_s;
          report.max_s = global.max_s;
          gpu_bench::set_distribution(report, global);
          report.valid = global_ok != 0;
          report.extra = std::string("case=") + case_name + " timing=batch batch_iters=" +
                         std::to_string(batch_iters) + " batch_samples=" +
                         std::to_string(batch_samples) +
                         " submission=host-stream completion=event-wait halo_elems=" +
                         std::to_string(halo) + " topology=ring bw=sendrecv device=\"" +
                         queue.get_device().get_info<sycl::info::device::name>() + "\"";
          gpu_bench::print_report(report);
        }
      }
    }

    sycl::free(buf, queue);

    MPI_Finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
