#include <cuda_runtime.h>

#include <cstddef>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe.hpp"
#include "oshmpi_space.h"
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
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();
  bool space_created = false;
  void* space = nullptr;
  double* stats_by_pe = nullptr;

  try {
    if (argc > 6) {
      throw std::invalid_argument(
          "usage: oshmpi_moe <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]");
    }
    const auto tokens = gpu_bench::parse_moe_size_arg(argc, argv, 1, 16384U, "token count");
    const auto hidden = gpu_bench::parse_moe_size_arg(argc, argv, 2, 256U, "hidden size");
    const auto iterations = gpu_bench::parse_moe_positive_int_arg(argc, argv, 3, 100, "iteration count");
    const auto warmup = gpu_bench::parse_moe_positive_int_arg(argc, argv, 4, 20, "warmup count");
    const auto routing_cases = gpu_bench::parse_moe_routing_cases(argc, argv, 5);
    const auto payload_elements = gpu_bench::moe_checked_multiply(tokens, hidden, "MoE payload");
    const auto payload_bytes =
        gpu_bench::moe_checked_multiply(payload_elements, sizeof(float), "MoE payload");
    const auto bytes = gpu_bench::moe_checked_multiply(
        gpu_bench::moe_checked_multiply(2U, payload_elements, "MoE useful bytes"), sizeof(float),
        "MoE useful bytes");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    stats_by_pe = static_cast<double*>(
        shmem_malloc(4U * static_cast<std::size_t>(pes) * sizeof(double)));
    if (stats_by_pe == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }

    int all_cases_ok = 1;
    for (const auto routing : routing_cases) {
      std::vector<gpu_bench::moe_plan> plans;
      plans.reserve(static_cast<std::size_t>(pes));
      for (int plan_pe = 0; plan_pe < pes; ++plan_pe) {
        plans.push_back(gpu_bench::make_moe_plan(tokens, hidden, plan_pe, pes, routing));
      }
      const auto& plan = plans[static_cast<std::size_t>(pe)];
      const auto host_send = gpu_bench::pack_moe_send(plan);
      std::vector<float> host_dispatch(plan.recv_elements);
      std::vector<float> host_combined(plan.send_elements);

      const auto dispatch_elements = std::max<std::size_t>(
          gpu_bench::moe_checked_multiply(plan.max_expert_tokens, hidden, "MoE dispatch buffer"), 1U);
      const auto dispatch_bytes =
          gpu_bench::moe_checked_multiply(dispatch_elements, sizeof(float), "MoE dispatch buffer");
      const auto allocation_bytes = gpu_bench::moe_checked_add(
          gpu_bench::moe_checked_multiply(2U, payload_bytes, "MoE symmetric allocation"),
          dispatch_bytes, "MoE symmetric allocation");
      const auto symmetric_bytes = std::max<std::size_t>(
          gpu_bench::moe_checked_multiply(2U, allocation_bytes, "MoE symmetric space"), 1U << 20U);
      space = gpu_bench_oshmpi_space_create(symmetric_bytes);
      if (space == nullptr) {
        throw std::runtime_error("failed to create OSHMPI CUDA memory space");
      }
      space_created = true;

      auto* device_send =
          static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, payload_bytes));
      auto* device_dispatch =
          static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, dispatch_bytes));
      auto* device_combined =
          static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, payload_bytes));
      if (device_send == nullptr || device_dispatch == nullptr || device_combined == nullptr) {
        throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
      }

      check_cuda(cudaMemcpy(device_send, host_send.data(), payload_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(send)");
      check_cuda(cudaMemset(device_dispatch, 0, dispatch_bytes), "cudaMemset(dispatch)");
      check_cuda(cudaMemset(device_combined, 0, payload_bytes), "cudaMemset(combined)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
      shmem_barrier_all();

      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        for (int destination = 0; destination < pes; ++destination) {
          const auto destination_index = static_cast<std::size_t>(destination);
          const auto count = static_cast<std::size_t>(plan.send_counts[destination_index]);
          if (count == 0) {
            continue;
          }
          float* target = device_dispatch +
                          plans[destination_index].recv_displacements[static_cast<std::size_t>(pe)];
          const float* source = device_send + plan.send_displacements[destination_index];
          const auto block_bytes = count * sizeof(float);
          if (destination == pe) {
            check_cuda(cudaMemcpy(target, source, block_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(dispatch self)");
          } else {
            shmem_putmem(target, source, block_bytes, destination);
          }
        }
        shmem_quiet();
        shmem_barrier_all();

        for (int source_pe = 0; source_pe < pes; ++source_pe) {
          const auto source_index = static_cast<std::size_t>(source_pe);
          const auto count = static_cast<std::size_t>(plan.recv_counts[source_index]);
          if (count == 0) {
            continue;
          }
          float* target =
              device_combined + plans[source_index].send_displacements[static_cast<std::size_t>(pe)];
          const float* source = device_dispatch + plan.recv_displacements[source_index];
          const auto block_bytes = count * sizeof(float);
          if (source_pe == pe) {
            check_cuda(cudaMemcpy(target, source, block_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(combine self)");
          } else {
            shmem_putmem(target, source, block_bytes, source_pe);
          }
        }
        shmem_quiet();
        shmem_barrier_all();
      });

      if (plan.recv_elements > 0) {
        check_cuda(cudaMemcpy(host_dispatch.data(), device_dispatch, plan.recv_elements * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(dispatch)");
      }
      check_cuda(cudaMemcpy(host_combined.data(), device_combined, payload_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(combined)");
      const int local_ok = gpu_bench::validate_moe_dispatch(host_dispatch.data(), plan) &&
                                   gpu_bench::validate_moe_combined(host_combined.data(), host_send)
                               ? 1
                               : 0;
      const double local_values[4] = {
          stats.avg_s, stats.min_s, stats.max_s, static_cast<double>(local_ok)};
      shmem_putmem(stats_by_pe + 4 * pe, local_values, 4U * sizeof(double), 0);
      shmem_quiet();
      shmem_barrier_all();

      double time_per_iter = stats.avg_s;
      double min_time = stats.min_s;
      double max_time = stats.max_s;
      int global_ok = local_ok;
      if (pe == 0) {
        global_ok = 1;
        for (int source_pe = 0; source_pe < pes; ++source_pe) {
          time_per_iter = std::max(time_per_iter, stats_by_pe[4 * source_pe + 0]);
          min_time = std::min(min_time, stats_by_pe[4 * source_pe + 1]);
          max_time = std::max(max_time, stats_by_pe[4 * source_pe + 2]);
          if (stats_by_pe[4 * source_pe + 3] < 0.5) {
            global_ok = 0;
          }
        }
        const double global_value = static_cast<double>(global_ok);
        for (int target_pe = 0; target_pe < pes; ++target_pe) {
          shmem_putmem(stats_by_pe, &global_value, sizeof(global_value), target_pe);
        }
        shmem_quiet();
      }
      shmem_barrier_all();
      global_ok = stats_by_pe[0] >= 0.5 ? 1 : 0;
      all_cases_ok = std::min(all_cases_ok, global_ok);

      gpu_bench_oshmpi_space_destroy(space);
      space = nullptr;
      space_created = false;

      if (pe == 0) {
        const double useful_gbytes_per_s = time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9
                                                               : 0.0;
        const double imbalance = static_cast<double>(plan.max_expert_tokens) / static_cast<double>(tokens);
        std::ostringstream extra;
        extra << "case=" << gpu_bench::moe_routing_name(routing)
              << " routing=" << gpu_bench::moe_routing_name(routing) << " tokens=" << tokens
              << " hidden=" << hidden << " top_k=1 max_expert_tokens=" << plan.max_expert_tokens
              << " expert_imbalance=" << imbalance << " useful_gbytes_per_s=" << useful_gbytes_per_s
              << " status=" << (global_ok ? "OK" : "ERROR") << " memory=device_symmetric";

        gpu_bench::bench_report report;
        report.name = "oshmpi_moe";
        report.n = tokens;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = min_time;
        report.max_s = max_time;
        gpu_bench::set_local_distribution(report, stats);
        report.valid = global_ok != 0;
        report.extra = extra.str();
        gpu_bench::print_report(report);
      }
    }

    shmem_free(stats_by_pe);
    shmem_finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    if (space_created) {
      gpu_bench_oshmpi_space_destroy(space);
    }
    shmem_global_exit(1);
  }
}
