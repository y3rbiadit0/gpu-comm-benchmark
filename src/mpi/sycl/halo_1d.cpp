#include <mpi.h>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// Comm-only 1D halo exchange benchmark, SYCL-aware MPI.
//
// Periodic ring: every rank exchanges a width-H halo with its left and right
// neighbour. Buffers are slice-local and device-resident:
//
//   [ left_halo(cap) | interior(2*cap) | right_halo(cap) ]   cap = max halo width
//
// The interior's first half is tagged with a "left boundary" marker and its
// second half with a "right boundary" marker, both exactly representable in
// float. After the exchange each rank validates its received halos locally
// against the markers its neighbours must have sent - no gather required.
//
// Send pointers are SYCL device pointers, so this measures SYCL-aware MPI, not
// host staging. Halo width H is swept; the reported GB/s is send+receive
// ("bus") bandwidth: 4 * H * sizeof(float) per rank per iteration.

namespace {

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
    if (ranks < 2) {
      throw std::runtime_error("ring halo exchange requires at least 2 ranks");
    }

    const auto max_halo = gpu_bench::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto halo_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_halo);

    const int left = (rank - 1 + ranks) % ranks;
    const int right = (rank + 1) % ranks;

    sycl::queue queue{sycl::default_selector_v};

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

    for (const std::size_t halo : halo_sizes) {
      const int count = mpi_count(halo);
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        MPI_Request reqs[4];
        // tag 0: messages travelling right; tag 1: messages travelling left.
        MPI_Irecv(recv_left, count, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recv_right, count, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(send_right, count, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(send_left, count, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
      });

      host_left.assign(halo, 0.0F);
      host_right.assign(halo, 0.0F);
      queue.copy(recv_left, host_left.data(), halo).wait();
      queue.copy(recv_right, host_right.data(), halo).wait();
      int local_ok = 1;
      for (std::size_t i = 0; i < halo; ++i) {
        if (!gpu_bench::nearly_equal(host_left[i], expect_left) ||
            !gpu_bench::nearly_equal(host_right[i], expect_right)) {
          local_ok = 0;
          break;
        }
      }

      int global_ok = 1;
      double max_avg = 0.0;
      double min_min = 0.0;
      double max_max = 0.0;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
      MPI_Reduce(&stats.avg_s, &max_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&stats.min_s, &min_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&stats.max_s, &max_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if (rank == 0) {
        gpu_bench::bench_report report;
        report.name = "sycl_mpi_halo_1d";
        report.n = halo;
        report.ranks = ranks;
        report.bytes_per_iter = 4U * halo * sizeof(float);  // 2 sends + 2 recvs per rank
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = max_avg;
        report.min_s = min_min;
        report.max_s = max_max;
        report.valid = global_ok != 0;
        report.extra = "halo_elems=" + std::to_string(halo) + " topology=ring bw=sendrecv device=\"" +
                       queue.get_device().get_info<sycl::info::device::name>() + "\"";
        gpu_bench::print_report(report);
      }
    }

    sycl::free(buf, queue);

    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
