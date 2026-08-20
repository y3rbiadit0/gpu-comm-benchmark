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
#include "stats/collective_shmem.hpp"
#include "oshmpi_space.h"
#include "report.hpp"
#include "timing.hpp"
#include "validation.hpp"

// Comm-only 1D halo exchange benchmark, OSHMPI (OpenSHMEM over MPI).
//
// Periodic ring with a swept halo width H, host-initiated. Each PE puts its
// boundary into both neighbours' halos, completes the CUDA work, then synchronises
// with shmem_barrier_all. Point-to-point waits (shmem_*_wait_until on a spin flag)
// can deadlock inter-node when passive RMA needs target-side progress, so we use a
// barrier handshake (a collective that always makes progress), as the other OSHMPI
// binaries do. Completion is per batch, not per exchange, so the timed loop
// reflects neighbour latency plus one device completion and one barrier per
// batch - the `isolated` case (one exchange per batch) is where that overhead
// shows up undivided.
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
  void* space = nullptr;

  try {
    if (pes < 2) {
      throw std::runtime_error("ring halo exchange requires at least 2 PEs");
    }

    const auto max_halo = gpu_bench::parse_size_arg(argc, argv, 1U << 20U);
    const auto iterations = gpu_bench::parse_positive_int_arg(argc, argv, 2, 100);
    const auto warmup = gpu_bench::parse_positive_int_arg(argc, argv, 3, 20);
    const auto halo_sizes = gpu_bench::parse_size_list_arg(argc, argv, 4, max_halo);
    const int batch_samples = gpu_bench::parse_positive_int_env("GPU_BENCH_BATCH_SAMPLES", 10);
    const int isolated_samples =
        gpu_bench::parse_positive_int_env("GPU_BENCH_ISOLATED_SAMPLES", 100);
    // One symmetric gather buffer serves both cases, so size it for the larger.
    const int max_samples = std::max(batch_samples, isolated_samples);

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
    auto* buf = static_cast<float*>(gpu_bench_oshmpi_space_malloc(space, total * sizeof(float)));
    if (buf == nullptr) {
      throw std::runtime_error("failed to allocate OSHMPI CUDA symmetric memory");
    }
    auto* sample_gather = static_cast<double*>(
        shmem_malloc(gpu_bench::collective_gather_elements(pes, max_samples) * sizeof(double)));
    auto* oks = static_cast<int*>(shmem_malloc(static_cast<std::size_t>(pes) * sizeof(int)));
    if (sample_gather == nullptr || oks == nullptr) {
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
    int failures = 0;

    for (const std::size_t halo : halo_sizes) {
      float* send_left = interior;
      float* send_right = interior + n_local - halo;
      float* recv_left = interior - halo;
      float* recv_right = interior + n_local;
      const std::size_t halo_bytes = halo * sizeof(float);

      for (const int batch_iters : gpu_bench::batch_iteration_counts(iterations)) {
        const int samples =
            gpu_bench::batch_samples_for(batch_iters, batch_samples, isolated_samples);
        const char* case_name = batch_iters == 1 ? "isolated" : "steady";
        int local_ok = 1;
        const auto stats = gpu_bench::run_batched_benchmark(
            warmup, batch_iters, samples,
            [&]() {
              check_cuda(cudaMemset(recv_left, 0xA5, halo_bytes), "cudaMemset(recv_left)");
              check_cuda(cudaMemset(recv_right, 0xA5, halo_bytes), "cudaMemset(recv_right)");
              check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(poison halos)");
              shmem_barrier_all();
            },
            [&](int count) {
              // Every exchange in a batch moves the same bytes into the same
              // slots, so the puts may all be in flight at once: the ring
              // dependency only has to hold at the batch boundary, where the
              // halos are read back and validated. Completing each exchange
              // individually - quiet, device sync and a global barrier per put
              // pair - is exactly what the `isolated` case (count == 1)
              // measures. Doing it here as well would leave OSHMPI the only
              // backend whose `steady` case cannot pipeline, and its bandwidth
              // column would then report barrier cost rather than transport.
              for (int i = 0; i < count; ++i) {
                shmem_putmem_nbi(recv_left, send_right, halo_bytes, right);
                shmem_putmem_nbi(recv_right, send_left, halo_bytes, left);
              }
              // CUDA-space RMA may enqueue device work that outlives
              // shmem_quiet. Close that work before the barrier and timer end.
              shmem_quiet();
              check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(halo batch)");
              shmem_barrier_all();
            },
            [&]() {
              host_left.assign(halo, 0.0F);
              host_right.assign(halo, 0.0F);
              check_cuda(cudaMemcpy(host_left.data(), recv_left, halo_bytes, cudaMemcpyDeviceToHost),
                         "cudaMemcpy(recv_left)");
              check_cuda(cudaMemcpy(host_right.data(), recv_right, halo_bytes, cudaMemcpyDeviceToHost),
                         "cudaMemcpy(recv_right)");
              for (std::size_t i = 0; i < halo; ++i) {
                if (!gpu_bench::nearly_equal(host_left[i], expect_left) ||
                    !gpu_bench::nearly_equal(host_right[i], expect_right)) {
                  local_ok = 0;
                  break;
                }
              }
            });
        const auto global = gpu_bench::collective_stats(stats, sample_gather, pe, pes);

        shmem_int_p(&oks[pe], local_ok, 0);
        shmem_quiet();
        shmem_barrier_all();

        int global_ok = local_ok;
        if (pe == 0) {
          global_ok = 1;
          for (int source = 0; source < pes; ++source) {
            global_ok = global_ok && oks[source];
          }
          oks[0] = global_ok;
          for (int target = 1; target < pes; ++target) {
            shmem_int_p(oks, global_ok, target);
          }
          shmem_quiet();
        }
        shmem_barrier_all();
        global_ok = oks[0];
        failures += global_ok == 0 ? 1 : 0;

        if (pe == 0) {
          gpu_bench::bench_report report;
          report.name = "oshmpi_halo_1d";
          report.n = halo;
          report.ranks = pes;
          report.bytes_per_iter = 4U * halo * sizeof(float);
          report.iterations = batch_iters;
          report.warmup = warmup;
          report.time_per_iter_s = global.avg_s;
          report.min_s = global.min_s;
          report.max_s = global.max_s;
          gpu_bench::set_distribution(report, global);
          report.valid = global_ok != 0;
          report.extra = "case=" + std::string(case_name) + " timing=batch batch_iters=" +
                         std::to_string(batch_iters) + " batch_samples=" +
                         std::to_string(samples) +
                          " submission=host-rma completion=quiet-device-sync-barrier halo_elems=" +
                         std::to_string(halo) + " topology=ring bw=sendrecv sync=barrier";
          gpu_bench::print_report(report);
        }
        shmem_barrier_all();
      }
    }

    shmem_free(oks);
    shmem_free(sample_gather);
    // The space allocation goes back before the space it came from does, as
    // OSHMPI's own CUDA-space test does.
    shmem_free(buf);
    gpu_bench_oshmpi_space_destroy(space);

    shmem_finalize();
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "PE " << pe << ": " << error.what() << '\n';
    // Space cleanup is collective and is unsafe when another PE may still be
    // inside the operation that failed locally.
    shmem_global_exit(1);
  }
}
