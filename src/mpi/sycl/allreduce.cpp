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

namespace {

int mpi_count(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("allreduce count exceeds int range");
  }
  return static_cast<int>(value);
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
    const auto max_elems = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elems);
    mpi_count(max_elems);
    gpu_bench::checked_size_multiply(max_elems, sizeof(float), "allreduce allocation");

    sycl::queue queue{device_for_rank(rank), sycl::property::queue::in_order()};

    float* device_send = sycl::malloc_device<float>(max_elems, queue);
    float* device_recv = sycl::malloc_device<float>(max_elems, queue);
    if (device_send == nullptr || device_recv == nullptr) {
      throw std::runtime_error("failed to allocate SYCL device memory");
    }

    int all_sizes_ok = 1;
    std::vector<float> host_recv;
    for (const std::size_t size : message_sizes) {
      const int count = mpi_count(size);
      const auto bytes = gpu_bench::checked_size_multiply(size, sizeof(float), "allreduce message");
      queue.fill(device_send, static_cast<float>(rank + 1), size).wait();

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        MPI_Allreduce(device_send, device_recv, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      });

      host_recv.resize(size);
      queue.copy(device_recv, host_recv.data(), size).wait();
      const float expected = static_cast<float>(static_cast<double>(ranks) * (ranks + 1.0) / 2.0);
      int local_ok = 1;
      for (std::size_t i = 0; i < size; ++i) {
        if (!gpu_bench::nearly_equal(host_recv[i], expected)) {
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
      all_sizes_ok = all_sizes_ok && global_ok;

      if (rank == 0) {
        const double algorithm_gbytes_per_s =
            max_avg > 0.0 ? static_cast<double>(bytes) / max_avg / 1.0e9 : 0.0;
        const double bus_gbytes_per_s = ranks > 1
                                           ? algorithm_gbytes_per_s * 2.0 * static_cast<double>(ranks - 1) /
                                                 static_cast<double>(ranks)
                                           : 0.0;

        gpu_bench::bench_report report;
        report.name = "sycl_mpi_allreduce";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = max_avg;
        report.min_s = min_min;
        report.max_s = max_max;
        gpu_bench::set_local_distribution(report, stats);
        report.valid = global_ok != 0;
        report.extra = "datatype=float32 reduction=sum bus_gbytes_per_s=" + std::to_string(bus_gbytes_per_s) +
                       " device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
        gpu_bench::print_report(report);
      }
    }

    sycl::free(device_send, queue);
    sycl::free(device_recv, queue);

    MPI_Finalize();
    return all_sizes_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
