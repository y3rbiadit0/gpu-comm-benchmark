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

// Comm-only 1D halo exchange benchmark, OSHMPI (OpenSHMEM over MPI).
//
// Periodic ring with a swept halo width H, host-initiated. Each PE puts its
// boundary into both neighbours' halos, then synchronises with shmem_barrier_all.
// Point-to-point waits (shmem_*_wait_until on a spin flag) can deadlock inter-node
// when passive RMA needs target-side progress, so we use a barrier handshake (a
// collective that always makes progress), as the other OSHMPI binaries do. The
// timed loop therefore reflects neighbour latency plus barrier overhead.
//
// Symmetric device buffer (via the OSHMPI CUDA memory space), identical layout
// on every PE:
//
//   [ left_halo(cap) | interior(2*cap) | right_halo(cap) ]   cap = max halo width
//
// Symmetric addressing means recv_left / recv_right name the right slot on the
// neighbour too. Interior markers (exact in float) let each PE validate locally;
// per-PE timings are reduced to PE 0 through the symmetric heap. Reported GB/s is
// send+receive ("bus") bandwidth: 4 * H * sizeof(float).

namespace {

void check_cuda(cudaError_t status, const char* call) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
  }
}

__global__ void fill_interior_kernel(float* interior, std::size_t n_local, std::size_t half,
                                     float left_marker, float right_marker) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_local) {
    interior[i] = i < half ? left_marker : right_marker;
  }
}

}  // namespace

int main(int argc, char** argv) {
  shmem_init();

  const int pe = shmem_my_pe();
  const int pes = shmem_n_pes();
  bool space_created = false;
  void* space = nullptr;

  try {
    if (pes < 2) {
      throw std::runtime_error("ring halo exchange requires at least 2 PEs");
    }

    const auto max_halo = gpu_bench::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto halo_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_halo);

    const int left = (pe - 1 + pes) % pes;
    const int right = (pe + 1) % pes;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("no CUDA devices available");
    }
    check_cuda(cudaSetDevice(pe % device_count), "cudaSetDevice");

    const std::size_t cap = max_halo;
    const std::size_t n_local = 2U * cap;
    const std::size_t total = n_local + 2U * cap;
    const float left_marker = static_cast<float>(2 * (pe + 1));
    const float right_marker = static_cast<float>(2 * (pe + 1) + 1);
    const float expect_left = static_cast<float>(2 * (left + 1) + 1);
    const float expect_right = static_cast<float>(2 * (right + 1));

    const auto symmetric_bytes = std::max<std::size_t>(total * sizeof(float), 1U << 20U);
    space = gpu_bench_oshmpi_space_create(symmetric_bytes);
    if (space == nullptr) {
      throw std::runtime_error("failed to create OSHMPI CUDA memory space");
    }
    space_created = true;

    auto* buf = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, total * sizeof(float)));
    if (buf == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI CUDA symmetric memory");
    }
    auto* stat_avg = static_cast<double*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(double)));
    auto* stat_min = static_cast<double*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(double)));
    auto* stat_max = static_cast<double*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(double)));
    auto* oks = static_cast<int*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(int)));
    if (stat_avg == nullptr || stat_min == nullptr || stat_max == nullptr || oks == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI symmetric scratch");
    }

    check_cuda(cudaMemset(buf, 0, total * sizeof(float)), "cudaMemset(buf)");
    float* interior = buf + cap;
    {
      constexpr int block_size = 256;
      const auto grid_size = static_cast<int>((n_local + block_size - 1U) / block_size);
      fill_interior_kernel<<<grid_size, block_size>>>(interior, n_local, cap, left_marker, right_marker);
      check_cuda(cudaGetLastError(), "fill_interior_kernel");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(fill)");
    }
    shmem_barrier_all();

    std::vector<float> host_left;
    std::vector<float> host_right;

    for (const std::size_t halo : halo_sizes) {
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;
      const std::size_t halo_bytes = halo * sizeof(float);

      shmem_barrier_all();
      const auto stats = gpu_bench::run_benchmark(warmup, iterations, [&]() {
        // Put my right boundary into the right neighbour's left halo, and my
        // left boundary into the left neighbour's right halo.
        shmem_putmem(recv_left, send_right, halo_bytes, right);
        shmem_putmem(recv_right, send_left, halo_bytes, left);
        shmem_quiet();        // complete my puts before the barrier releases
        shmem_barrier_all();  // neighbours' halos have arrived once this returns
      });

      host_left.assign(halo, 0.0F);
      host_right.assign(halo, 0.0F);
      check_cuda(cudaMemcpy(host_left.data(), recv_left, halo_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv_left)");
      check_cuda(cudaMemcpy(host_right.data(), recv_right, halo_bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy(recv_right)");
      int local_ok = 1;
      for (std::size_t i = 0; i < halo; ++i) {
        if (!gpu_bench::nearly_equal(host_left[i], expect_left) ||
            !gpu_bench::nearly_equal(host_right[i], expect_right)) {
          local_ok = 0;
          break;
        }
      }

      shmem_double_p(&stat_avg[pe], stats.avg_s, 0);
      shmem_double_p(&stat_min[pe], stats.min_s, 0);
      shmem_double_p(&stat_max[pe], stats.max_s, 0);
      shmem_int_p(&oks[pe], local_ok, 0);
      shmem_quiet();
      shmem_barrier_all();

      if (pe == 0) {
        double max_avg = 0.0;
        double min_min = stat_min[0];
        double max_max = 0.0;
        int global_ok = 1;
        for (int source = 0; source < pes; ++source) {
          max_avg = std::max(max_avg, stat_avg[source]);
          min_min = std::min(min_min, stat_min[source]);
          max_max = std::max(max_max, stat_max[source]);
          global_ok = global_ok && oks[source];
        }

        gpu_bench::bench_report report;
        report.name = "oshmpi_halo_1d";
        report.n = halo;
        report.ranks = pes;
        report.bytes_per_iter = 4U * halo * sizeof(float);
        report.iterations = iterations;
        report.warmup = warmup;
        report.time_per_iter_s = max_avg;
        report.min_s = min_min;
        report.max_s = max_max;
        report.valid = global_ok != 0;
        report.extra = "halo_elems=" + std::to_string(halo) + " topology=ring bw=sendrecv sync=barrier";
        gpu_bench::print_report(report);
      }
      shmem_barrier_all();
    }

    shmem_free(oks);
    shmem_free(stat_max);
    shmem_free(stat_min);
    shmem_free(stat_avg);
    gpu_bench_oshmpi_space_destroy(space);
    space_created = false;

    shmem_finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    if (space_created) {
      gpu_bench_oshmpi_space_destroy(space);
    }
    shmem_global_exit(1);
  }
}
