#include <cuda_runtime.h>

#include <cstddef>

#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <shmem.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "oshmpi_space.h"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

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
  void* space = nullptr;
  float* device_send = nullptr;
  float* device_recv = nullptr;

  try {
    if (pes != 2) {
      throw std::runtime_error("pingpong requires exactly 2 PEs");
    }
    const int peer = pe == 0 ? 1 : 0;

    const auto max_elems = gpu_bench::parse_size_arg(argc, argv, 1U << 22U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto message_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_elems);

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    const auto symmetric_bytes = std::max<std::size_t>(4U * max_elems * sizeof(float), 1U << 20U);
    space = gpu_bench_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    device_send = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_elems * sizeof(float)));
    device_recv = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, max_elems * sizeof(float)));
    if (device_send == nullptr || device_recv == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric memory");
    }

    std::vector<float> host_send(max_elems);
    for (std::size_t i = 0; i < max_elems; ++i) {
      host_send[i] = static_cast<float>(i % 1024U);
    }
    check_cuda(cudaMemcpy(device_send, host_send.data(), max_elems * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy(send)");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(init)");
    shmem_barrier_all();

    // Barrier-based ping/pong handshake. OSHMPI point-to-point waits
    // (shmem_*_wait_until on a spin flag) can deadlock inter-node when passive
    // RMA needs target-side progress, so we synchronize with shmem_barrier_all
    // (a collective that always makes progress), as the other OSHMPI binaries do.
    // The reported latency therefore includes barrier overhead.
    for (std::size_t size : message_sizes) {
      const std::size_t bytes = size * sizeof(float);

      shmem_barrier_all();
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        if (pe == 0) {
          shmem_putmem(device_recv, device_send, bytes, peer);  // ping -> peer
          shmem_quiet();
          shmem_barrier_all();  // peer now holds the ping
          shmem_barrier_all();  // wait for the peer's pong to complete
        } else {
          shmem_barrier_all();  // ping has arrived
          shmem_putmem(device_recv, device_recv, bytes, peer);  // pong -> peer
          shmem_quiet();
          shmem_barrier_all();  // initiator now holds the pong
        }
      });

      if (pe == 0) {
        std::vector<float> host_recv(size);
        check_cuda(cudaMemcpy(host_recv.data(), device_recv, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(recv)");
        bool ok = true;
        for (std::size_t i = 0; i < size && ok; ++i) {
          ok = gpu_bench::nearly_equal(host_recv[i], host_send[i]);
        }

        gpu_bench::bench_report report;
        report.name = "oshmpi_pingpong";
        report.n = size;
        report.ranks = pes;
        report.bytes_per_iter = bytes;
        report.iterations = iterations;
        report.warmup = warmup;
        /* One-way latency = half the measured round trip. Unlike the
         * collectives, this is not reduced across ranks: the peer's own timings
         * cover a different window (it waits for the ping before replying), so
         * the initiator's round trip is the measurement by definition. */
        report.time_per_iter_s = 0.5 * stats.avg_s;
        report.min_s = 0.5 * stats.min_s;
        report.max_s = 0.5 * stats.max_s;
        gpu_bench::set_distribution(report, stats, 0.5);
        report.valid = ok;
        gpu_bench::print_report(report);
      }
    }

    shmem_barrier_all();
    shmem_free(device_recv);
    device_recv = nullptr;
    shmem_free(device_send);
    device_send = nullptr;
    gpu_bench_oshmpi_space_destroy(space);

    shmem_finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    // Allocation cleanup and space detach are collective and are unsafe when
    // another PE may still be inside the operation that failed locally.
    shmem_global_exit(1);
  }
}
