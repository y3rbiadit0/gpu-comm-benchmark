#include <mpi.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cli.hpp"
#include "partition.hpp"
#include "report.hpp"
#include "stencil2d.hpp"
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
    const auto side = comm_playground::parse_size_arg(argc, argv, 1U << 12U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 50);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 10);
    const auto local_cols = comm_playground::local_count(side, rank, ranks);
    const auto col_offset = comm_playground::local_offset(side, rank, ranks);
    const auto width = local_cols + 2U;
    const int left = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int right = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int column_count = static_cast<int>(side);

    sycl::queue queue(device_for_rank(rank), sycl::property::queue::in_order());

    float* old_field = sycl::malloc_device<float>(side * width, queue);
    float* new_field = sycl::malloc_device<float>(side * width, queue);
    float* send_west = sycl::malloc_device<float>(side, queue);
    float* send_east = sycl::malloc_device<float>(side, queue);
    float* recv_west = sycl::malloc_device<float>(side, queue);
    float* recv_east = sycl::malloc_device<float>(side, queue);
    if (old_field == nullptr || new_field == nullptr || send_west == nullptr || send_east == nullptr ||
        recv_west == nullptr || recv_east == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }

    queue.memset(old_field, 0, side * width * sizeof(float)).wait();
    queue.memset(new_field, 0, side * width * sizeof(float)).wait();
    queue.memset(recv_west, 0, side * sizeof(float)).wait();
    queue.memset(recv_east, 0, side * sizeof(float)).wait();

    if (local_cols > 0) {
      queue.parallel_for(sycl::range<2>{side, local_cols}, [=](sycl::id<2> id) {
        const auto i = id[0];
        const auto jj = id[1];
        old_field[i * width + (jj + 1U)] = static_cast<float>(i + col_offset + jj);
      }).wait();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
      if (local_cols > 0) {
        const auto last = local_cols;
        queue.parallel_for(sycl::range<1>{side}, [=](sycl::id<1> id) {
          const auto i = id[0];
          send_west[i] = old_field[i * width + 1U];
          send_east[i] = old_field[i * width + last];
        }).wait();
      }
      MPI_Sendrecv(send_west, column_count, MPI_FLOAT, left, 0, recv_east, column_count, MPI_FLOAT, right, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(send_east, column_count, MPI_FLOAT, right, 1, recv_west, column_count, MPI_FLOAT, left, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (local_cols > 0) {
        const auto east_ghost = local_cols + 1U;
        queue.parallel_for(sycl::range<1>{side}, [=](sycl::id<1> id) {
          const auto i = id[0];
          old_field[i * width + 0U] = recv_west[i];
          old_field[i * width + east_ghost] = recv_east[i];
        }).wait();
        queue.parallel_for(sycl::range<2>{side, local_cols}, [=](sycl::id<2> id) {
          const auto i = id[0];
          const auto j = id[1] + 1U;
          const float north = i > 0 ? old_field[(i - 1U) * width + j] : 0.0F;
          const float south = i + 1U < side ? old_field[(i + 1U) * width + j] : 0.0F;
          const float west = old_field[i * width + (j - 1U)];
          const float east = old_field[i * width + (j + 1U)];
          new_field[i * width + j] = 0.25F * (north + south + west + east);
        }).wait();
      }
    });

    double time_per_iter = 0.0;
    double min_time = 0.0;
    double max_time = 0.0;
    MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int local_ok = 1;
    if (local_cols > 0) {
      std::vector<float> host_new(side * width);
      queue.copy(new_field, host_new.data(), side * width).wait();
      const auto field = [](std::size_t i, std::size_t jg) { return static_cast<float>(i + jg); };
      local_ok = comm_playground::validate_columns(
                     host_new.data(), side, local_cols, width, col_offset,
                     [&](std::size_t i, std::size_t jg) { return comm_playground::stencil5(i, jg, side, field); })
                     ? 1
                     : 0;
    }
    int global_ok = 1;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    sycl::free(old_field, queue);
    sycl::free(new_field, queue);
    sycl::free(send_west, queue);
    sycl::free(send_east, queue);
    sycl::free(recv_west, queue);
    sycl::free(recv_east, queue);

    if (rank == 0) {
      comm_playground::bench_report report;
      report.name = "sycl_mpi_halo_2d";
      report.n = side;
      report.ranks = ranks;
      report.bytes_per_iter = 2U * side * sizeof(float);
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
