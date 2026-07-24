#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe.hpp"
#include "oneccl.hpp"
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

bool is_point_to_point_unsupported(const std::string& message) {
  std::string lower(message.size(), '\0');
  std::transform(message.begin(), message.end(), lower.begin(),
                  [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  const bool unsupported = lower.find("not implemented") != std::string::npos ||
                           lower.find("not-implemented") != std::string::npos ||
                           lower.find("not_implemented") != std::string::npos ||
                           lower.find("unimplemented") != std::string::npos ||
                           lower.find("unsupported") != std::string::npos ||
                           lower.find("not supported") != std::string::npos;
  const bool point_to_point = lower.find("send") != std::string::npos || lower.find("recv") != std::string::npos ||
                              lower.find("point-to-point") != std::string::npos ||
                              lower.find("point to point") != std::string::npos ||
                              lower.find("pt2pt") != std::string::npos;
  return unsupported && point_to_point;
}

struct device_buffers {
  explicit device_buffers(sycl::queue& queue) : queue(queue) {}

  ~device_buffers() {
    if (send != nullptr) sycl::free(send, queue);
    if (dispatch != nullptr) sycl::free(dispatch, queue);
    if (combined != nullptr) sycl::free(combined, queue);
  }

  device_buffers(const device_buffers&) = delete;
  device_buffers& operator=(const device_buffers&) = delete;

  sycl::queue& queue;
  float* send = nullptr;
  float* dispatch = nullptr;
  float* combined = nullptr;
};

struct probe_buffers {
  explicit probe_buffers(sycl::queue& queue) : queue(queue) {}

  ~probe_buffers() {
    if (send != nullptr) sycl::free(send, queue);
    if (recv != nullptr) sycl::free(recv, queue);
  }

  probe_buffers(const probe_buffers&) = delete;
  probe_buffers& operator=(const probe_buffers&) = delete;

  sycl::queue& queue;
  float* send = nullptr;
  float* recv = nullptr;
};

void print_not_implemented(comm_playground::moe_routing routing, std::size_t tokens, std::size_t hidden,
                           std::size_t bytes, int ranks, int iterations, int warmup,
                           const char* reason = "point_to_point") {
  std::cout << "sycl_oneccl_moe n=" << tokens << " ranks=" << ranks << " bytes=" << bytes
            << " iters=" << iterations << " warmup=" << warmup
            << " time_per_iter_s=0 usec=0 min_usec=0 max_usec=0 gbytes_per_s=0 case="
            << comm_playground::moe_routing_name(routing)
            << " routing=" << comm_playground::moe_routing_name(routing) << " tokens=" << tokens
            << " hidden=" << hidden
            << " top_k=1 status=NOT_IMPLEMENTED reason=" << reason << " validation=SKIP\n";
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
          "usage: sycl_oneccl_moe <tokens_per_rank> [hidden] [iterations] [warmup] [routing_cases]");
    }
    const auto tokens = comm_playground::parse_moe_size_arg(argc, argv, 1, 16384U, "token count");
    const auto hidden = comm_playground::parse_moe_size_arg(argc, argv, 2, 256U, "hidden size");
    const auto iterations = comm_playground::parse_moe_positive_int_arg(argc, argv, 3, 100, "iteration count");
    const auto warmup = comm_playground::parse_moe_positive_int_arg(argc, argv, 4, 20, "warmup count");
    const auto routing_cases = comm_playground::parse_moe_routing_cases(argc, argv, 5);
    const auto payload_elements = comm_playground::moe_checked_multiply(tokens, hidden, "MoE payload");
    const auto bytes = comm_playground::moe_checked_multiply(
        comm_playground::moe_checked_multiply(2U, payload_elements, "MoE useful bytes"), sizeof(float),
        "MoE useful bytes");

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

    int local_probe_status = 0;
    std::string local_probe_error;
    {
      probe_buffers buffers(queue);
      buffers.send = sycl::malloc_device<float>(1, queue);
      buffers.recv = sycl::malloc_device<float>(1, queue);
      if (buffers.send == nullptr || buffers.recv == nullptr) {
        throw std::runtime_error("failed to allocate SYCL point-to-point probe memory");
      }

      std::vector<ccl::event> events;
      events.reserve(2);
      try {
        const int recv_peer = ranks == 1 ? rank : (rank - 1 + ranks) % ranks;
        const int send_peer = ranks == 1 ? rank : (rank + 1) % ranks;
        comm_playground::ccl_group_scope group;
        events.push_back(ccl::recv(buffers.recv, 1, ccl::datatype::float32, recv_peer, comm, stream));
        events.push_back(ccl::send(buffers.send, 1, ccl::datatype::float32, send_peer, comm, stream));
        group.end();
      } catch (const std::exception& error) {
        local_probe_error = error.what();
        local_probe_status = is_point_to_point_unsupported(local_probe_error) ? 1 : 2;
      } catch (...) {
        local_probe_error = "oneCCL point-to-point probe threw a non-standard exception";
        local_probe_status = 2;
      }

      for (auto& event : events) {
        try {
          event.wait();
        } catch (const std::exception& error) {
          const std::string message = error.what();
          const int status = is_point_to_point_unsupported(message) ? 1 : 2;
          if (status > local_probe_status) {
            local_probe_error = message;
            local_probe_status = status;
          }
        } catch (...) {
          local_probe_error = "oneCCL point-to-point probe wait threw a non-standard exception";
          local_probe_status = 2;
        }
      }
    }

    int local_probe_counts[2] = {local_probe_status == 1 ? 1 : 0, local_probe_status == 2 ? 1 : 0};
    int global_probe_counts[2] = {0, 0};
    MPI_Allreduce(local_probe_counts, global_probe_counts, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (global_probe_counts[1] != 0) {
      throw std::runtime_error(local_probe_status == 2 ? local_probe_error
                                                       : "oneCCL point-to-point probe failed on another rank");
    }
    if (global_probe_counts[0] == ranks) {
      if (rank == 0) {
        for (const auto routing : routing_cases) {
          print_not_implemented(routing, tokens, hidden, bytes, ranks, iterations, warmup);
        }
      }
      MPI_Finalize();
      return 0;
    }
    if (global_probe_counts[0] != 0) {
      throw std::runtime_error("oneCCL point-to-point probe reported inconsistent capability across ranks");
    }

    auto exchange_phase_unchecked = [&](float* send_buffer, const std::vector<int>& send_counts,
                                        const std::vector<int>& send_displacements, float* recv_buffer,
                                        const std::vector<int>& recv_counts,
                                        const std::vector<int>& recv_displacements) {
      std::vector<ccl::event> events;
      events.reserve(static_cast<std::size_t>(2 * ranks));
      comm_playground::ccl_group_scope group;
      for (int peer = 0; peer < ranks; ++peer) {
        const auto index = static_cast<std::size_t>(peer);
        if (recv_counts[index] > 0) {
          events.push_back(ccl::recv(recv_buffer + recv_displacements[index], recv_counts[index],
                                     ccl::datatype::float32, peer, comm, stream));
        }
      }
      for (int peer = 0; peer < ranks; ++peer) {
        const auto index = static_cast<std::size_t>(peer);
        if (send_counts[index] > 0) {
          events.push_back(ccl::send(send_buffer + send_displacements[index], send_counts[index],
                                     ccl::datatype::float32, peer, comm, stream));
        }
      }
      group.end();
      for (auto& event : events) {
        event.wait();
      }
    };

    int all_cases_ok = 1;
    for (const auto routing : routing_cases) {
      const auto plan = comm_playground::make_moe_plan(tokens, hidden, rank, ranks, routing);
      const auto host_send = comm_playground::pack_moe_send(plan);
      std::vector<float> host_dispatch(plan.recv_elements);
      std::vector<float> host_combined(plan.send_elements);

      {
        device_buffers buffers(queue);
        const auto dispatch_allocation = std::max<std::size_t>(plan.recv_elements, 1U);
        buffers.send = sycl::malloc_device<float>(plan.send_elements, queue);
        buffers.dispatch = sycl::malloc_device<float>(dispatch_allocation, queue);
        buffers.combined = sycl::malloc_device<float>(plan.send_elements, queue);
        if (buffers.send == nullptr || buffers.dispatch == nullptr || buffers.combined == nullptr) {
          throw std::runtime_error("failed to allocate SYCL device memory");
        }
        queue.copy(host_send.data(), buffers.send, plan.send_elements).wait();
        queue.memset(buffers.dispatch, 0, dispatch_allocation * sizeof(float)).wait();
        queue.memset(buffers.combined, 0, plan.send_elements * sizeof(float)).wait();

        MPI_Barrier(MPI_COMM_WORLD);
        const auto stats = comm_playground::run_benchmark(warmup, iterations, [&]() {
          exchange_phase_unchecked(buffers.send, plan.send_counts, plan.send_displacements, buffers.dispatch,
                                   plan.recv_counts, plan.recv_displacements);
          exchange_phase_unchecked(buffers.dispatch, plan.recv_counts, plan.recv_displacements, buffers.combined,
                                   plan.send_counts, plan.send_displacements);
        });

        double time_per_iter = 0.0;
        double min_time = 0.0;
        double max_time = 0.0;
        MPI_Reduce(&stats.avg_s, &time_per_iter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.min_s, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.max_s, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (plan.recv_elements > 0) {
          queue.copy(buffers.dispatch, host_dispatch.data(), plan.recv_elements).wait();
        }
        queue.copy(buffers.combined, host_combined.data(), plan.send_elements).wait();
        int local_ok = comm_playground::validate_moe_dispatch(host_dispatch.data(), plan) &&
                               comm_playground::validate_moe_combined(host_combined.data(), host_send)
                           ? 1
                           : 0;
        int global_ok = 1;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        all_cases_ok = std::min(all_cases_ok, global_ok);

        if (rank == 0) {
          const double useful_gbytes_per_s =
              time_per_iter > 0.0 ? static_cast<double>(bytes) / time_per_iter / 1.0e9 : 0.0;
          const double imbalance = static_cast<double>(plan.max_expert_tokens) / static_cast<double>(tokens);
          std::ostringstream extra;
          extra << "case=" << comm_playground::moe_routing_name(routing)
                << " routing=" << comm_playground::moe_routing_name(routing) << " tokens=" << tokens
                << " hidden=" << hidden << " top_k=1 max_expert_tokens=" << plan.max_expert_tokens
                << " expert_imbalance=" << imbalance << " useful_gbytes_per_s=" << useful_gbytes_per_s
                << " status=" << (global_ok ? "OK" : "ERROR");

          comm_playground::bench_report report;
          report.name = "sycl_oneccl_moe";
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
          comm_playground::print_report(report);
        }
      }
    }

    MPI_Finalize();
    return all_cases_ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
  } catch (...) {
    std::cerr << "rank " << rank << ": unknown fatal exception\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}
