#include <mpi.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "stats/collective_mpi.hpp"
#include "moe.hpp"
#include "report.hpp"
#include "timing.hpp"

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
    if (argc > 6) {
      throw std::invalid_argument(
          "usage: sycl_mpi_moe <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]");
    }
    const auto tokens = gpu_bench::parse_moe_size_arg(argc, argv, 1, 16384U, "token count");
    const auto hidden = gpu_bench::parse_moe_size_arg(argc, argv, 2, 256U, "hidden size");
    const auto iterations = gpu_bench::parse_moe_positive_int_arg(argc, argv, 3, 100, "iteration count");
    const auto warmup = gpu_bench::parse_moe_positive_int_arg(argc, argv, 4, 20, "warmup count");
    const auto routing_cases = gpu_bench::parse_moe_routing_cases(argc, argv, 5);
    const auto payload_elements = gpu_bench::moe_checked_multiply(tokens, hidden, "MoE payload");
    const auto bytes = gpu_bench::moe_checked_multiply(
        gpu_bench::moe_checked_multiply(2U, payload_elements, "MoE useful bytes"), sizeof(float),
        "MoE useful bytes");

    sycl::queue queue(device_for_rank(rank), sycl::property::queue::in_order());

    int all_cases_ok = 1;
    for (const auto routing : routing_cases) {
      const auto plan = gpu_bench::make_moe_plan(tokens, hidden, rank, ranks, routing);
      const auto host_send = gpu_bench::pack_moe_send(plan);
      std::vector<float> host_dispatch(plan.recv_elements);
      std::vector<float> host_combined(plan.send_elements);

      const auto dispatch_allocation = std::max<std::size_t>(plan.recv_elements, 1U);
      float* device_send = sycl::malloc_device<float>(plan.send_elements, queue);
      float* device_dispatch = sycl::malloc_device<float>(dispatch_allocation, queue);
      float* device_combined = sycl::malloc_device<float>(plan.send_elements, queue);
      if (device_send == nullptr || device_dispatch == nullptr || device_combined == nullptr) {
        throw std::runtime_error("failed to allocate SYCL device memory");
      }
      queue.copy(host_send.data(), device_send, plan.send_elements).wait();
      queue.memset(device_dispatch, 0, dispatch_allocation * sizeof(float)).wait();
      queue.memset(device_combined, 0, plan.send_elements * sizeof(float)).wait();

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        MPI_Alltoallv(device_send, plan.send_counts.data(), plan.send_displacements.data(), MPI_FLOAT,
                      device_dispatch, plan.recv_counts.data(), plan.recv_displacements.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);
        MPI_Alltoallv(device_dispatch, plan.recv_counts.data(), plan.recv_displacements.data(), MPI_FLOAT,
                      device_combined, plan.send_counts.data(), plan.send_displacements.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);
      });

      const auto global = gpu_bench::collective_stats(stats);
      const double time_per_iter = global.avg_s;

      if (plan.recv_elements > 0) {
        queue.copy(device_dispatch, host_dispatch.data(), plan.recv_elements).wait();
      }
      queue.copy(device_combined, host_combined.data(), plan.send_elements).wait();
      int local_ok = gpu_bench::validate_moe_dispatch(host_dispatch.data(), plan) &&
                             gpu_bench::validate_moe_combined(host_combined.data(), host_send)
                         ? 1
                         : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_cases_ok = std::min(all_cases_ok, global_ok);

      sycl::free(device_send, queue);
      sycl::free(device_dispatch, queue);
      sycl::free(device_combined, queue);

      if (rank == 0) {
        const double useful_gbytes_per_s = time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9
                                                               : 0.0;
        const double imbalance = static_cast<double>(plan.max_expert_tokens) / static_cast<double>(tokens);
        std::ostringstream extra;
        extra << "case=" << gpu_bench::moe_routing_name(routing)
              << " routing=" << gpu_bench::moe_routing_name(routing) << " tokens=" << tokens
              << " hidden=" << hidden << " top_k=1 max_expert_tokens=" << plan.max_expert_tokens
              << " expert_imbalance=" << imbalance << " useful_gbytes_per_s=" << useful_gbytes_per_s
              << " status=" << (global_ok ? "OK" : "ERROR");

        gpu_bench::bench_report report;
        report.name = "sycl_mpi_moe";
        report.n = tokens;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = global.min_s;
        report.max_s = global.max_s;
        gpu_bench::set_distribution(report, global);
        report.valid = global_ok != 0;
        report.extra = extra.str();
        gpu_bench::print_report(report);
      }
    }

    MPI_Finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
