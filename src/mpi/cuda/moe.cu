#include <mpi.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe.hpp"
#include "report.hpp"
#include "timing.hpp"

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
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
          "usage: cuda_mpi_moe <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]");
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

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(rank % device_count), "cudaSetDevice");

    int all_cases_ok = 1;
    for (const auto routing : routing_cases) {
      const auto plan = gpu_bench::make_moe_plan(tokens, hidden, rank, ranks, routing);
      const auto host_send = gpu_bench::pack_moe_send(plan);
      std::vector<float> host_dispatch(plan.recv_elements);
      std::vector<float> host_combined(plan.send_elements);

      float* device_send = nullptr;
      float* device_dispatch = nullptr;
      float* device_combined = nullptr;
      const auto dispatch_allocation = std::max<std::size_t>(plan.recv_elements, 1U);
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_send), plan.send_elements * sizeof(float)),
                 "cudaMalloc(send)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_dispatch), dispatch_allocation * sizeof(float)),
                 "cudaMalloc(dispatch)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_combined), plan.send_elements * sizeof(float)),
                 "cudaMalloc(combined)");
      check_cuda(cudaMemcpy(device_send, host_send.data(), plan.send_elements * sizeof(float), cudaMemcpyHostToDevice),
                 "cudaMemcpy(send)");
      check_cuda(cudaMemset(device_dispatch, 0, dispatch_allocation * sizeof(float)), "cudaMemset(dispatch)");
      check_cuda(cudaMemset(device_combined, 0, plan.send_elements * sizeof(float)), "cudaMemset(combined)");

      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        MPI_Alltoallv(device_send, plan.send_counts.data(), plan.send_displacements.data(), MPI_FLOAT,
                      device_dispatch, plan.recv_counts.data(), plan.recv_displacements.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);
        MPI_Alltoallv(device_dispatch, plan.recv_counts.data(), plan.recv_displacements.data(), MPI_FLOAT,
                      device_combined, plan.send_counts.data(), plan.send_displacements.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);
      });

      double time_per_iter = 0.0;
      double min_time = 0.0;
      double max_time = 0.0;
      MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
      MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

      if (plan.recv_elements > 0) {
        check_cuda(cudaMemcpy(host_dispatch.data(), device_dispatch, plan.recv_elements * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(dispatch)");
      }
      check_cuda(cudaMemcpy(host_combined.data(), device_combined, plan.send_elements * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy(combined)");
      int local_ok = gpu_bench::validate_moe_dispatch(host_dispatch.data(), plan) &&
                             gpu_bench::validate_moe_combined(host_combined.data(), host_send)
                         ? 1
                         : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_cases_ok = std::min(all_cases_ok, global_ok);

      check_cuda(cudaFree(device_send), "cudaFree(send)");
      check_cuda(cudaFree(device_dispatch), "cudaFree(dispatch)");
      check_cuda(cudaFree(device_combined), "cudaFree(combined)");

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
        report.name = "cuda_mpi_moe";
        report.n = tokens;
        report.ranks = ranks;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = min_time;
        report.max_s = max_time;
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
