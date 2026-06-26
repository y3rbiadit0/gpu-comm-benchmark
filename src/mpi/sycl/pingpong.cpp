#include <mpi.h>
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
    const auto max_elems = comm_playground::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = comm_playground::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = comm_playground::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = comm_playground::parse_size_list_arg(argc, argv, 4, max_elems);
    const int peer = rank == 0 ? 1 : 0;

    sycl::queue queue{sycl::default_selector_v};

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
      const int count = static_cast<int>(size);

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
        if (rank == 0) {
          MPI_Send(device_send, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD);
          MPI_Recv(device_recv, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(device_recv, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Send(device_recv, count, MPI_FLOAT, peer, 0, MPI_COMM_WORLD);
        }
      });

      if (rank == 0) {
        std::vector<float> host_recv(size);
        queue.copy(device_recv, host_recv.data(), size).wait();
        bool ok = true;
        for (std::size_t i = 0; i < size && ok; ++i) {
          ok = comm_playground::nearly_equal(host_recv[i], host_send[i]);
        }

        comm_playground::bench_report report;
        report.name = "sycl_mpi_pingpong";
        report.n = size;
        report.ranks = ranks;
        report.bytes_per_iter = size * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = 0.5 * stats.avg_s;
        report.min_s = 0.5 * stats.min_s;
        report.max_s = 0.5 * stats.max_s;
        report.valid = ok;
        report.extra = "device=\"" + queue.get_device().get_info<sycl::info::device::name>() + "\"";
        comm_playground::print_report(report);
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
