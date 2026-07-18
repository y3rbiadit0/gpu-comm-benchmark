#include <mpi.h>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

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

void check_nvshmem(int status, const char* call) {
  if (status != 0) {
    throw std::runtime_error(std::string(call) + " failed with status " + std::to_string(status));
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int mpi_rank = 0;
  int mpi_ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
  bool nvshmem_initialized = false;

  try {
    if (argc > 6) {
      throw std::invalid_argument(
          "usage: cuda_nvshmem_moe <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]");
    }
    const auto tokens = comm_playground::parse_moe_size_arg(argc, argv, 1, 16384U, "token count");
    const auto hidden = comm_playground::parse_moe_size_arg(argc, argv, 2, 256U, "hidden size");
    const auto iterations = comm_playground::parse_moe_positive_int_arg(argc, argv, 3, 100, "iteration count");
    const auto warmup = comm_playground::parse_moe_positive_int_arg(argc, argv, 4, 20, "warmup count");
    const auto routing_cases = comm_playground::parse_moe_routing_cases(argc, argv, 5);
    const auto payload_elements = comm_playground::moe_checked_multiply(tokens, hidden, "MoE payload");
    const auto payload_bytes =
        comm_playground::moe_checked_multiply(payload_elements, sizeof(float), "MoE payload");
    const auto bytes = comm_playground::moe_checked_multiply(
        comm_playground::moe_checked_multiply(2U, payload_elements, "MoE useful bytes"), sizeof(float),
        "MoE useful bytes");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(mpi_rank % device_count), "cudaSetDevice");

    nvshmemx_init_attr_t attr = {};
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    check_nvshmem(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr), "nvshmemx_init_attr");
    nvshmem_initialized = true;

    const int pe = nvshmem_my_pe();
    const int pes = nvshmem_n_pes();
    if (pe != mpi_rank || pes != mpi_ranks) {
      throw std::runtime_error("NVSHMEM PE layout does not match MPI rank layout");
    }

    int all_cases_ok = 1;
    for (const auto routing : routing_cases) {
      std::vector<comm_playground::moe_plan> plans;
      plans.reserve(static_cast<std::size_t>(pes));
      for (int plan_pe = 0; plan_pe < pes; ++plan_pe) {
        plans.push_back(comm_playground::make_moe_plan(tokens, hidden, plan_pe, pes, routing));
      }
      const auto& plan = plans[static_cast<std::size_t>(pe)];
      const auto host_send = comm_playground::pack_moe_send(plan);
      std::vector<float> host_dispatch(plan.recv_elements);
      std::vector<float> host_combined(plan.send_elements);

      const auto dispatch_elements = std::max<std::size_t>(
          comm_playground::moe_checked_multiply(plan.max_expert_tokens, hidden, "MoE dispatch buffer"), 1U);
      const auto dispatch_bytes =
          comm_playground::moe_checked_multiply(dispatch_elements, sizeof(float), "MoE dispatch buffer");
      auto* device_send = static_cast<float*>(nvshmem_malloc(payload_bytes));
      auto* device_dispatch = static_cast<float*>(nvshmem_malloc(dispatch_bytes));
      auto* device_combined = static_cast<float*>(nvshmem_malloc(payload_bytes));
      if (device_send == nullptr || device_dispatch == nullptr || device_combined == nullptr) {
        throw std::runtime_error("failed to allocate NVSHMEM symmetric memory");
      }

      check_cuda(cudaMemcpy(device_send, host_send.data(), payload_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy(send)");
      check_cuda(cudaMemset(device_dispatch, 0, dispatch_bytes), "cudaMemset(dispatch)");
      check_cuda(cudaMemset(device_combined, 0, payload_bytes), "cudaMemset(combined)");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");

      nvshmem_barrier_all();
      MPI_Barrier(MPI_COMM_WORLD);
      const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
        for (int destination = 0; destination < pes; ++destination) {
          const auto destination_index = static_cast<std::size_t>(destination);
          const auto count = static_cast<std::size_t>(plan.send_counts[destination_index]);
          if (count == 0) {
            continue;
          }
          float* target = device_dispatch +
                          plans[destination_index].recv_displacements[static_cast<std::size_t>(pe)];
          const float* source = device_send + plan.send_displacements[destination_index];
          if (destination == pe) {
            check_cuda(cudaMemcpy(target, source, count * sizeof(float), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(dispatch self)");
          } else {
            nvshmem_float_put(target, source, count, destination);
          }
        }
        nvshmem_quiet();
        nvshmem_barrier_all();

        for (int source_pe = 0; source_pe < pes; ++source_pe) {
          const auto source_index = static_cast<std::size_t>(source_pe);
          const auto count = static_cast<std::size_t>(plan.recv_counts[source_index]);
          if (count == 0) {
            continue;
          }
          float* target =
              device_combined + plans[source_index].send_displacements[static_cast<std::size_t>(pe)];
          const float* source = device_dispatch + plan.recv_displacements[source_index];
          if (source_pe == pe) {
            check_cuda(cudaMemcpy(target, source, count * sizeof(float), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy(combine self)");
          } else {
            nvshmem_float_put(target, source, count, source_pe);
          }
        }
        nvshmem_quiet();
        nvshmem_barrier_all();
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
      check_cuda(cudaMemcpy(host_combined.data(), device_combined, payload_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(combined)");
      int local_ok = comm_playground::validate_moe_dispatch(host_dispatch.data(), plan) &&
                             comm_playground::validate_moe_combined(host_combined.data(), host_send)
                         ? 1
                         : 0;
      int global_ok = 1;
      MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      all_cases_ok = std::min(all_cases_ok, global_ok);

      nvshmem_free(device_combined);
      nvshmem_free(device_dispatch);
      nvshmem_free(device_send);

      if (pe == 0) {
        const double useful_gbytes_per_s = time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9
                                                               : 0.0;
        const double imbalance = static_cast<double>(plan.max_expert_tokens) / static_cast<double>(tokens);
        std::ostringstream extra;
        extra << "case=" << comm_playground::moe_routing_name(routing)
              << " routing=" << comm_playground::moe_routing_name(routing) << " tokens=" << tokens
              << " hidden=" << hidden << " top_k=1 max_expert_tokens=" << plan.max_expert_tokens
              << " expert_imbalance=" << imbalance << " useful_gbytes_per_s=" << useful_gbytes_per_s
              << " status=" << (global_ok ? "OK" : "ERROR") << " memory=device_symmetric";

        comm_playground::bench_report report;
        report.name = "cuda_nvshmem_moe";
        report.n = tokens;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = time_per_iter;
        report.min_s = min_time;
        report.max_s = max_time;
        report.valid = global_ok != 0;
        report.extra = extra.str();
        comm_playground::print_report(report);
      }
    }

    nvshmem_finalize();
    nvshmem_initialized = false;
    MPI_Finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << mpi_rank << ": " << error.what() << '\n';
    if (nvshmem_initialized) {
      nvshmem_global_exit(1);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
